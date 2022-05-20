import numpy as np
import json
import os
import cv2
from . import paths

paths = paths.paths


def keypoints_to_xy(keypoints_xyv):
    x = keypoints_xyv[0::3]
    y = keypoints_xyv[1::3]
    v = keypoints_xyv[2::3]
    xy = np.array(list(zip(x, y)))

    return xy


def load_img(path, name):
    img = cv2.imread(os.path.join(path, name))
    return img[:, :, ::-1]


def load_ann(path):
    with open(path) as f:
        ann = json.load(f)
    return ann


def get_iid_to_ann(val_ann_path, train_ann_path):
    if train_ann_path:
        train_ann = load_ann(train_ann_path)
    if val_ann_path:
        val_ann = load_ann(val_ann_path)

    iid_to_ann = {}
    iid_to_idx = {}

    if train_ann_path:
        for i, a in enumerate(train_ann['images']):
            iid_to_ann[a['id']] = []
            iid_to_idx[a['id']] = i
    if val_ann_path:
        for i, a in enumerate(val_ann['images']):
            iid_to_ann[a['id']] = []
            iid_to_idx[a['id']] = i

    if train_ann_path:
        for a in train_ann['annotations']:
            iid_to_ann[a['image_id']].append(a)
    if val_ann_path:
        for a in val_ann['annotations']:
            iid_to_ann[a['image_id']].append(a)

    return iid_to_ann, iid_to_idx
