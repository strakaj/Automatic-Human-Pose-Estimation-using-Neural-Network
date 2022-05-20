import cv2
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

colors = {'rankl': (0, 0, 1), 'rknee': (0, 0, 1), 'rhip': (0, 0, 1),
          'lankl': (1, 0, 0), 'lknee': (1, 0, 0), 'lhip': (1, 0, 0),
          'rwri': (1, 1, 0), 'relb': (1, 1, 0), 'rsho': (1, 1, 0),
          'lwri': (0, 1, 0), 'lelb': (0, 1, 0), 'lsho': (0, 1, 0),
          'head': (0, 1, 1), 'thorax': (0, 1, 1), 'upper_neck': (0, 1, 1),
          'pelvis': (1, 0, 1)}
'''
mpii_idx_to_jnt = {0: 'rankl', 1: 'rknee', 2: 'rhip', 5: 'lankl', 4: 'lknee', 3: 'lhip',
                   6: 'pelvis', 7: 'thorax', 8: 'upper_neck', 11: 'relb', 10: 'rwri', 9: 'head',
                   12: 'rsho', 13: 'lsho', 14: 'lelb', 15: 'lwri'}

jnt_connections = [
    (0, 1),
    (1, 2),
    (2, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    (5, 4),
    (4, 3),
    (3, 6),
    (10, 11),
    (11, 12),
    (12, 7),
    (15, 14),
    (14, 13),
    (13, 7)
]
'''


def load_image(image):
    if isinstance(image, list):
        images = []
        for im in image:
            path = os.path.join(im.image_path, im.image_name)
            img = cv2.imread(path)
            images.append(img[:, :, ::-1])
        return images
    else:
        path = os.path.join(image.image_path, image.image_name)
        img = cv2.imread(path)
        img = img[:, :, ::-1]
        return img


def show_image(image):
    im = load_image(image)
    plt.imshow(im)


def visualize_image_annotations(images, joints=True, connections=True, joints_name=False,
                                alpha=0.5, joints_r=5, connectins_w=5, figsize=(15, 10)):
    max_cols = 2
    images_num = 1
    if isinstance(images, list):
        images_num = len(images)
    else:
        images = [images]
    nrows = math.ceil(images_num / max_cols)
    ncols = max_cols
    if nrows == 1:
        ncols = images_num % (max_cols + 1)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, frameon=False, figsize=figsize)
    if not isinstance(axs, np.ndarray):
        axs = np.array(axs)
    axs = np.ndarray.flatten(axs)

    ax_idx = 0
    for image in images:
        ax = axs[ax_idx]
        ax_idx = ax_idx + 1
        img = load_image(image)
        text_overlay = []
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(str(image.imgidx))
        mpii_idx_to_jnt = image.idx_to_jnt
        jnt_connections = image.connections

        for points in image.points:

            for jnt in points:
                center = (jnt["x"], jnt["y"])
                jnt_id = jnt["id"]
                jnt_name = mpii_idx_to_jnt[jnt_id]
                if joints:
                    ax.add_patch(Circle(center, radius=joints_r, color=colors[jnt_name], fill=True, alpha=alpha))
                if joints_name:
                    text_overlay.append(
                        ax.text(x=jnt["x"], y=jnt["y"], s=mpii_idx_to_jnt[jnt_id], color=colors[jnt_name], fontsize=12))

            for j in jnt_connections:
                jnt_id1 = j[0]
                jnt_id2 = j[1]

                jnt1 = jnt_from_id(points, jnt_id1)
                jnt2 = jnt_from_id(points, jnt_id2)

                if jnt1 is None or jnt2 is None:
                    continue

                color = colors[mpii_idx_to_jnt[jnt_id1]]
                x = [jnt1["x"], jnt2["x"]]
                y = [jnt1["y"], jnt2["y"]]
                if connections:
                    ax.add_line(Line2D(x, y, lw=connectins_w, color=color, alpha=alpha))


def jnt_from_id(joints, jnt_id):
    for jnt in joints:
        if jnt["id"] == jnt_id:
            return jnt
    return None
