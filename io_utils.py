import os
import cv2 as cv
import numpy as np


def save_patch(frame_gray, x, y, out_dir, frame_num, patch_size=31):
    """Вырезает патч patch_size x patch_size вокруг (x, y) и сохраняет PNG."""
    half = patch_size // 2
    h, w = frame_gray.shape
    if x - half < 0 or y - half < 0 or x + half + 1 > w or y + half + 1 > h:
        return
    patch = frame_gray[y - half:y + half + 1, x - half:x + half + 1]
    cv.imwrite(os.path.join(out_dir, f"{int(frame_num)}.png"), patch)


def save_class_summary_txt(output_path, classes, class_names):
    """classes: dict {traj_id: class_idx}; пишет class_summary.txt с человекочитаемым текстом."""
    counts = {i: 0 for i in class_names}
    for c in classes.values():
        counts[c] += 1
    lines = [f"{class_names[i]}: {counts[i]}" for i in sorted(class_names)]
    with open(os.path.join(output_path, "class_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def append_class_column(csv_path, classes, class_names):
    """Дописывает к output.csv колонку relax_class. classes: {traj_id: class_idx}."""
    data = np.loadtxt(csv_path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    header = "id,frame,label,x,y,area,traj_type,v_flow,v_channel,relax_class"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for row in data:
            traj_id = int(row[0])
            cls = class_names.get(classes.get(traj_id, -1), "unknown")
            f.write(",".join([f"{v:g}" for v in row] + [cls]) + "\n")