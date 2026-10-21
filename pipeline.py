"""Основной пайплайн обработки одного видео.

Вызывается из UI (ui/app.py). Возвращает строку статуса.
"""
import os
import glob
import shutil
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import cv2 as cv
from tqdm import tqdm
from safetensors.torch import load_file

import channels
from config import Config
from models import UNet, LSTMCNNLight
from io_utils import save_patch, save_class_summary_txt, append_class_column


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------- загрузка моделей -----------------------------

def load_seg_model(cfg: Config):
    m = UNet()
    m.load_state_dict(load_file(cfg.seg_weights))
    return m.to(DEVICE).eval()


def load_clf_model(cfg: Config):
    m = LSTMCNNLight(num_classes=cfg.num_classes)
    m.load_state_dict(load_file(cfg.clf_weights))
    return m.to(DEVICE).eval()


# ----------------------------- сегментация -----------------------------

class DataForSegment(torch.utils.data.Dataset):
    def __init__(self, folder_path, size, exts=(".tif",)):
        super().__init__()
        self.raw_files = sorted(os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(exts))
        self.size = size

    def __getitem__(self, index):
        data = Image.open(self.raw_files[index]).convert("L").resize(size=self.size)
        arr = np.array(data, np.float32)
        arr *= 1.0 / (arr.max() + 1e-8)
        return torch.from_numpy(arr).float().unsqueeze(0)

    def __len__(self):
        return len(self.raw_files)


@torch.no_grad()
def segment(model, video, h, w, masks_dir):
    os.makedirs(masks_dir, exist_ok=True)
    i = 0
    for batch in tqdm(video, desc="segment"):
        i += 1
        inputs = batch.to(DEVICE)
        y_pred = model(inputs).cpu()
        y_print = np.array(y_pred[0, 0, :, :])
        y_print = cv.resize(y_print, (w, h))
        y_mask = np.where(y_print > 0, 255, 0).astype("uint8")
        Image.fromarray(y_mask, mode="L").save(
            os.path.join(masks_dir, f"{i:04d}.bmp"), format="BMP"
        )


# ----------------------------- трекинг -----------------------------

class Trajectory:
    def __init__(self, id, frame_num, label, x, y, area):
        self.traj = [[id, frame_num, label, x, y, area, -1, -1, 0]]
        self.id = id

    def addToTraj(self, frame_num, label, x, y, area):
        v = float(np.sqrt((x - self.traj[-1][3]) ** 2 + (y - self.traj[-1][4]) ** 2))
        if self.traj[0][-2] == -1:
            self.traj[0][-2] = v
        self.traj.append([self.id, frame_num, label, x, y, area, -1, v, 0])

    def getFirstFrameNum(self):
        return self.traj[0][1]

    def getLastFrameNum(self):
        return self.traj[-1][1]

    def getFirstPoint(self):
        return self.traj[0][3], self.traj[0][4], self.traj[0][5]

    def getLastPoint(self):
        return self.traj[-1][3], self.traj[-1][4], self.traj[-1][5]

    def getS(self):
        return float(np.sqrt((self.traj[-1][3] - self.traj[0][3]) ** 2
                             + (self.traj[-1][4] - self.traj[0][4]) ** 2))

    def getMedianArrea(self):
        return float(np.median(np.asarray([p[5] for p in self.traj])))

    def getMedianVelocity(self):
        if self.getS() < np.sqrt(self.getMedianArrea()):
            return -1
        return float(np.median(np.asarray([p[-2] for p in self.traj])))

    def setChannelVelocity(self, v):
        for p in self.traj:
            p[-1] = v

    def nameTheType(self, t):
        for p in self.traj:
            p[6] = t

    def trajToNpArray(self):
        return np.asarray([[*p] for p in self.traj])


class TrajectoriesList:
    def __init__(self):
        self.trajList = []
        self.numOfTrajes = 0

    def addTraj(self, traj):
        self.trajList.append(traj)
        self.numOfTrajes += 1

    def findTrajectoriesWithFrameN(self, frameNum):
        return [t for t in self.trajList if t.getLastFrameNum() == frameNum]

    def defineTrajesTypes(self, x1, x2, ax1, ax2, traj_sensivity, capture_sensivity):
        for traj in self.trajList:
            xS, _, _ = traj.getFirstPoint()
            xE, _, _ = traj.getLastPoint()
            if xS < ax1 and xE < ax2:
                traj.nameTheType(0)
            elif xS > ax1 and xE <= x1 + traj_sensivity:
                traj.nameTheType(1)
            elif x2 - traj_sensivity <= xS <= x2 + traj_sensivity + capture_sensivity:
                traj.nameTheType(2)
            elif xS > x2:
                traj.nameTheType(3)

    def setVelocityById(self, id, v):
        for traj in self.trajList:
            if traj.id == id:
                traj.setChannelVelocity(v)
                return

    def getListAsArray(self):
        if not self.trajList:
            return np.array([]).reshape(0, 9)
        return np.vstack([t.trajToNpArray() for t in self.trajList])


# ----------------------------- анализ скоростей -----------------------------

def filter_and_align_outliers(help_outliers, deep=4):
    mask_in = help_outliers[0, :] > 0
    mask_out = help_outliers[1, :] > 0
    in_events = help_outliers[0, mask_in].astype(int)
    out_events = help_outliers[1, mask_out].astype(int)
    in_ids = help_outliers[2, mask_in].astype(int)
    out_ids = help_outliers[3, mask_out].astype(int)

    j = 0
    while j < min(len(in_events), len(out_events)):
        diff = out_events[j] - in_events[j]
        if diff <= 0:
            out_events = np.delete(out_events, j)
            out_ids = np.delete(out_ids, j)
            continue
        h = 0
        flag = True
        help_dif = []
        while flag and (j + h + 1) < len(in_events) and (j + h) < len(out_events):
            help_dif.append(out_events[j + h] - in_events[j + h + 1])
            h += 1
            if h == deep or help_dif[-1] <= 0:
                flag = False
        if h != 0:
            if help_dif[-1] > 0:
                in_events = np.delete(in_events, j)
                in_ids = np.delete(in_ids, j)
                continue
            else:
                j += 1
        else:
            j += 1

    m = min(len(in_events), len(out_events))
    if m == 0:
        return np.array([]).reshape(4, 0)
    return np.vstack((in_events[:m], out_events[:m], in_ids[:m], out_ids[:m]))


def calculate_velocities(sparse_outliers, points_x):
    if sparse_outliers.size == 0:
        return np.array([]).reshape(3, 0)
    dist = abs(points_x[1] - points_x[0])
    times = np.abs(sparse_outliers[1, :] - sparse_outliers[0, :])
    velocities = np.divide(dist, times, out=np.zeros_like(times, dtype=float), where=times != 0)
    return np.vstack((velocities, sparse_outliers[2, :], sparse_outliers[3, :]))


def capture_transitions(trajectoriesList, x1, x2, roi):
    ins, outs, idsin, idsout = [], [], [], []
    for traj in trajectoriesList.trajList:
        xlst, _, _ = traj.getLastPoint()
        flst = traj.getLastFrameNum()
        ffst = traj.getFirstFrameNum()
        xfst, _, _ = traj.getFirstPoint()
        if abs(x1 - xlst) <= roi:
            ins.append(flst)
            idsin.append(traj.id)
        elif abs(x2 - xfst) <= roi:
            outs.append(ffst)
            idsout.append(traj.id)
    if not ins or not outs:
        return np.array([]).reshape(3, 0)
    in_arr = np.asarray([ins, idsin])
    in_arr = in_arr[:, in_arr[0, :].argsort()]
    out_arr = np.asarray([outs, idsout])
    out_arr = out_arr[:, out_arr[0, :].argsort()]
    help_outliers = np.zeros((4, max(len(ins), len(outs))))
    for i in range(len(ins)):
        help_outliers[0, i] = in_arr[0, i]
        help_outliers[2, i] = in_arr[1, i]
    for i in range(len(outs)):
        help_outliers[1, i] = out_arr[0, i]
        help_outliers[3, i] = out_arr[1, i]
    sparse = filter_and_align_outliers(help_outliers)
    return calculate_velocities(sparse, [x1, x2])


def applyNsigmaRule(arr, sigmas=2):
    if arr.size == 0:
        return arr
    mu, sigma = np.mean(arr), np.std(arr)
    return arr[(arr >= mu - sigmas * sigma) & (arr <= mu + sigmas * sigma)]


# ----------------------------- классификация -----------------------------

@torch.no_grad()
def classify_trajectories(traj_root, model, cfg: Config):
    """traj_root = temp/trajes/<sample>/<id>/ ; возвращает {traj_id: class_idx}."""
    out = {}
    if not os.path.isdir(traj_root):
        return out
    model.eval()
    for name in sorted(os.listdir(traj_root), key=lambda x: int(x) if x.isdigit() else x):
        path = os.path.join(traj_root, name)
        if not os.path.isdir(path):
            continue
        frames = sorted(f for f in os.listdir(path) if f.endswith((".png", ".bmp")))
        if not frames:
            continue
        seq = []
        for f in frames:
            arr = np.array(Image.open(os.path.join(path, f)).convert("L"), np.float32)
            mn, mx = arr.min(), arr.max()
            arr = (arr - mn) / (mx - mn + 1e-8)
            seq.append(arr)
        x = torch.from_numpy(np.stack(seq)).float().unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,T,31,31)
        logits = model(x)
        out[int(name)] = int(logits.argmax(dim=1).item())
    return out


# ----------------------------- основной пайплайн -----------------------------

def ProcessVideo(vid_path, cfg: Config, seg_model, clf_model=None) -> str:
    sample_name = os.path.split(vid_path)[-1]
    exp_name = os.path.split(os.path.split(vid_path)[-2])[-1]
    output_path = os.path.join(cfg.outputs_dir, exp_name, sample_name)
    os.makedirs(output_path, exist_ok=True)

    masks_dir = os.path.join(cfg.temp_dir, "masks")
    trajes_dir = os.path.join(cfg.temp_dir, "trajes", sample_name)

    try:
        # 1. Границы канала
        channel_in, channel_out, alig_in, alig_out = channels.main(vid_path, cfg.frame_exts)

        # 2. Размеры кадра
        images = sorted([img for img in os.listdir(vid_path)
                 if img.endswith(cfg.frame_exts)])
        frame0 = cv.imread(os.path.join(vid_path, images[0]))
        height, width, _ = frame0.shape
        del frame0, images

        # 3. Сегментация
        dataset = DataForSegment(vid_path, cfg.seg_input_size, exts=cfg.frame_exts)
        loader = DataLoader(dataset, batch_size=cfg.seg_batch_size, shuffle=False)
        segment(seg_model, loader, height, width, masks_dir)

        # 4. Связные компоненты
        mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith((".png", ".bmp"))])
        data = []
        for name in tqdm(mask_files, desc="components"):
            img = cv.imread(os.path.join(masks_dir, name))
            img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
            output = cv.connectedComponentsWithStatsWithAlgorithm(
                img, connectivity=8, ltype=cv.CV_32S, ccltype=cv.CCL_BOLELLI)
            data.append(output)
        df = pd.DataFrame(data=data,
                          columns=["numlabels", "labels", "stats", "centroids"])
        del data

        areas, cells = [], []
        for index, row in df.iterrows():
            label = 0
            for s in row["stats"]:
                if label == 0:
                    label += 1
                    continue
                areas.append(s[4])
                cells.append([index, label, row["centroids"][label, 0],
                              row["centroids"][label, 1], s[4], s[0], s[1], s[2], s[3]])
                label += 1
        areas = np.asarray(areas)
        median_area = np.median(areas) if areas.size else 0
        del areas

        frames = df["labels"]
        cells_raw = np.asarray(cells)
        cells_clean = []
        for i, row in tqdm(enumerate(cells), desc="filter cells"):
            if median_area / row[4] > 4:
                mask = cv.inRange(frames[cells_raw[i, 0]], row[1], row[1])
                frames[cells_raw[i, 0]] = cv.bitwise_and(
                    frames[cells_raw[i, 0]], frames[cells_raw[i, 0]],
                    mask=cv.bitwise_not(mask))
            else:
                cells_clean.append(row)
        cells_clean = np.asarray(cells_clean) if cells_clean else np.array([]).reshape(0, 9)
        del cells, cells_raw, df

        # 5. Трекинг
        ListOfTraj = TrajectoriesList()
        id_counter = 0
        for i in tqdm(range(frames.size), desc="track"):
            subcell = cells_clean[cells_clean[:, 0] == i] if cells_clean.size else np.array([])
            if subcell.size == 0:
                continue
            prev = ListOfTraj.findTrajectoriesWithFrameN(i - 1)
            used = []
            if not prev:
                for row in subcell:
                    ListOfTraj.addTraj(Trajectory(id_counter, row[0], row[1], row[2], row[3], row[4]))
                    id_counter += 1
                continue
            for traj in prev:
                closest, closest2 = 99999, 99999
                best, best2 = -1, -2
                for idx, row in enumerate(subcell):
                    xt, yt, area = traj.getLastPoint()
                    r = np.sqrt((xt - row[2]) ** 2 + (yt - row[3]) ** 2)
                    bound = np.sqrt(row[4]) * 2
                    if r <= bound and r < closest and idx not in used:
                        closest2, best2 = closest, best
                        closest, best = r, idx
                    elif r <= bound and r < closest2 and r >= closest and idx not in used:
                        closest2, best2 = r, idx
                if best != -1:
                    if (best2 >= 0
                            and abs(subcell[best2][4] - area) < abs(subcell[best][4] - area)
                            and 0.5 < subcell[best2][4] / area < 2):
                        ListOfTraj.trajList[traj.id].addToTraj(
                            subcell[best2][0], subcell[best2][1], subcell[best2][2],
                            subcell[best2][3], subcell[best2][4])
                        used.append(best2)
                    elif 0.5 <= subcell[best][4] / area <= 2:
                        ListOfTraj.trajList[traj.id].addToTraj(
                            subcell[best][0], subcell[best][1], subcell[best][2],
                            subcell[best][3], subcell[best][4])
                        used.append(best)
            for idx, row in enumerate(subcell):
                if idx not in used:
                    ListOfTraj.addTraj(Trajectory(id_counter, row[0], row[1], row[2], row[3], row[4]))
                    id_counter += 1

        ListOfTraj.defineTrajesTypes(channel_in, channel_out, alig_in, alig_out,
                                     cfg.traj_sensitivity, cfg.capture_sensitivity)

        # 6. Скорости
        trajes_ids, vels = [], []
        for traj in ListOfTraj.trajList:
            v = traj.getMedianVelocity()
            if v > 0:
                trajes_ids.append(traj.id)
                vels.append(v)
        vels = applyNsigmaRule(np.asarray(vels)) if vels else np.array([])
        v_med = float(np.median(vels)) if vels.size else -1

        velocities = capture_transitions(ListOfTraj, channel_in, channel_out, cfg.capture_sensitivity)
        if velocities.size:
            velocities = velocities.T
            for row in velocities:
                scaled = row[0] * cfg.channels_area_coeff / v_med if v_med > 0 else 0.0
                ListOfTraj.setVelocityById(int(row[1]), scaled)
                ListOfTraj.setVelocityById(int(row[2]), scaled)

        finalList = ListOfTraj.getListAsArray()
        del ListOfTraj

        # 7. Сохранение output.csv
        np.savetxt(os.path.join(output_path, "output.csv"), finalList, delimiter=",")
        if velocities.size and v_med > 0:
            np.savetxt(os.path.join(output_path, "v.csv"),
                       velocities[:, 0] / (v_med * cfg.channels_area_coeff), delimiter=",")

        # 8. Нарезка патчей в temp/trajes/<sample>/<id>/
        images = sorted([img for img in os.listdir(vid_path)
                 if img.endswith(cfg.frame_exts)])
        for traj_id in trajes_ids:
            traj_path = os.path.join(trajes_dir, str(traj_id))
            for row in finalList:
                if row[0] == traj_id and row[-3] == 2:
                    os.makedirs(traj_path, exist_ok=True)
                    frame = cv.imread(os.path.join(vid_path, images[int(row[1])]))
                    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    save_patch(frame, int(row[3]), int(row[4]), traj_path,
                               int(row[1]), cfg.patch_size)

        # 9. Классификация
        if clf_model is not None:
            classes = classify_trajectories(trajes_dir, clf_model, cfg)
            if classes:
                append_class_column(os.path.join(output_path, "output.csv"),
                                    classes, cfg.class_names)
                save_class_summary_txt(output_path, classes, cfg.class_names)

        return f"Обработка завершена: {sample_name}"

    finally:
        # Удаляем весь temp целиком — маски и патчи больше не нужны
        if os.path.isdir(cfg.temp_dir):
            shutil.rmtree(cfg.temp_dir, ignore_errors=True)