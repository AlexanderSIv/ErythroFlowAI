"""Детекция границ микроканала по морфологическому градиенту кадра."""
import os
import numpy as np
import cv2 as cv
from tqdm import tqdm
from sklearn.cluster import DBSCAN
import scipy


def find_channel(dist):
    h, _ = dist.shape
    w = 10
    Smax = 0
    position = 0
    for i in range(int(h / 3), int(2 * h / 3)):
        S = 0
        for j in range(-w, w):
            S = S + dist[i + j, 1]
        if S > Smax:
            Smax = S
            position = i
    return position


def get_ch_position(img):
    kernel3 = np.ones((3, 3), np.uint8)
    gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel3)
    h, w = gradient.shape
    dist = []
    for i in range(h):
        s = np.int32(0)
        for j in range(w):
            s = s + gradient[i, j]
        dist.append([i, s])
    dist = np.array(dist)
    channel = find_channel(dist)
    ch_distr = []
    for i in range(w):
        s = np.int32(0)
        for j in range(channel - 10, channel + 10):
            s = s + gradient[j, i]
        ch_distr.append([i, s])
    ch_distr = np.array(ch_distr)
    newsig = scipy.signal.medfilt(ch_distr[:, 1], kernel_size=101)
    newsig = w * (newsig - newsig.min()) / (newsig.max() - newsig.min())
    amp = newsig.max() - newsig.min()
    ch_distr[:, 1] = newsig
    db = DBSCAN(eps=amp / 20, min_samples=50).fit(ch_distr)
    labels = db.labels_
    topLabel = labels.max() + 1
    labelP = []
    labelsAvg = []
    for l in range(topLabel):
        currP = 0
        avg = 0
        for i in range(w):
            if labels[i] == l:
                currP = currP + 1
                avg = avg + newsig[i]
        avg = avg / currP
        labelP.append(currP)
        labelsAvg.append(avg)
    top1l = 0
    top2l = -2
    for l in range(topLabel):
        if labelsAvg[l] > labelsAvg[top1l]:
            top2l = top1l
            top1l = l
        if labelsAvg[l] > labelsAvg[top2l] and labelsAvg[l] < labelsAvg[top1l]:
            top2l = l
    if (labelP[top1l] * labelsAvg[top1l]) > (labelP[top2l] * labelsAvg[top2l]):
        thechannel = top1l
        alig = top2l
    else:
        thechannel = top2l
        alig = top1l
    channelx1 = -1
    channelx2 = 0
    aligx1 = -1
    aligx2 = 0
    for i in range(w):
        if labels[i] == thechannel:
            if channelx1 == -1:
                channelx1 = i
            channelx2 = i
        if labels[i] == alig:
            if aligx1 == -1:
                aligx1 = i
            aligx2 = i
    return channelx1, channelx2, aligx1, aligx2


def get_img(img_path):
    img = cv.imread(img_path)
    return cv.cvtColor(img, cv.COLOR_RGB2GRAY)


def multithr(image):
    x1, x2, ax1, ax2 = get_ch_position(get_img(image))
    return x1, x2, ax1, ax2


def get_frames(vid_path, exts=(".tif",)):
    return [os.path.join(vid_path, img) for img in sorted(os.listdir(vid_path))
            if img.endswith(exts)]


def main(vid_path, exts=(".tif",)):
    import concurrent.futures
    frames = get_frames(vid_path=vid_path, exts=exts)
    x1s, x2s, ax1s, ax2s = [], [], [], []
    with concurrent.futures.ProcessPoolExecutor() as pool:
        for x1, x2, ax1, ax2 in tqdm(pool.map(multithr, frames)):
            x1s.append(x1)
            x2s.append(x2)
            ax1s.append(ax1)
            ax2s.append(ax2)
    return (
        int(np.round(np.median(np.asarray(x1s)))),
        int(np.round(np.median(np.asarray(x2s)))),
        int(np.round(np.median(np.asarray(ax1s)))),
        int(np.round(np.median(np.asarray(ax2s)))),
    )


if __name__ == '__main__':
    # Прямой запуск модуля не предполагается.
    main()