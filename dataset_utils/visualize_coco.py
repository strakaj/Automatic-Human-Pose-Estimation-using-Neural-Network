import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from dataset_utils.dataset_config.coco import dataset_info
from dataset_utils.utils_coco import image_idx_to_ann_idx, only_with_annotations, get_random_idx


keypoint_names = ['nose',
                'left_eye',
                'right_eye',
                'left_ear',
                'right_ear',
                'left_shoulder',
                'right_shoulder',
                'left_elbow',
                'right_elbow',
                'left_wrist',
                'right_wrist',
                'left_hip',
                'right_hip',
                'left_knee',
                'right_knee',
                'left_ankle',
                'right_ankle']

keypoint_connections = [[16, 14],
                        [14, 12],
                        [17, 15],
                        [15, 13],
                        [12, 13],
                        [6, 12],
                        [7, 13],
                        [6, 7],
                        [6, 8],
                        [7, 9],
                        [8, 10],
                        [9, 11],
                        [2, 3],
                        [1, 2],
                        [1, 3],
                        [2, 4],
                        [3, 5],
                        [4, 6],
                        [5, 7]]



def histograms(keypoints, keypounts_visible, keypounts_not_visible, nannotations):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax = axs[0]
    n, bins = np.histogram(nannotations, np.max(nannotations)-1)
    p2 = ax.bar(bins[:-1], n)
    tx = ax.set_xticks(bins[:-1])
    ax.set_title('Annotation in single image')
    ax.set_xlabel('Number of annotations in image')
    ax.set_ylabel('Number of images')


    joints = []
    for k in keypoints:
        joints.append(keypounts_visible[k]+keypounts_not_visible[k])

    ax = axs[1]
    ax.bar(keypoints, joints)
    plt.sca(ax)
    xt = plt.xticks(rotation='vertical')
    ax.set_title('Histogram of keypoints')
    ax.set_xlabel('Keypoint name')
    yl = ax.set_ylabel('Number of annotations')


def load_image(image, dataset_path):
    if isinstance(image, list):
        images = []
        for im in image:
            path = os.path.join(dataset_path, im["file_name"])
            img = cv2.imread(path)
            images.append(img[:, :, ::-1])
        return images
    else:
        path = os.path.join(dataset_path, image["file_name"])
        img = cv2.imread(path)
        img = img[:, :, ::-1]
        return img


def show_image(image, dataset_path):
    im = load_image(image, dataset_path)
    if isinstance(image, list):
        n = int(np.ceil(np.sqrt(len(im))))
        
        fig, axs = plt.subplots(nrows=n, ncols=n, figsize=(15, 10))
        axs = np.ndarray.flatten(axs)
        for i, img in enumerate(im):
            ax = axs[i]
            ax.set_axis_off()
            ax.imshow(img)
    else:
        plt.imshow(im)
        
def visualize_image_annotations(images, annotations, dataset_path, joints=True, connections=True, joints_name=False,
                                alpha=0.5, joints_r=5, connectins_w=5, figsize=(15, 10)):
    max_cols = 2
    images_num = 1
    if isinstance(images, list):
        images_num = len(images)
    else:
        images = [images]
        annotations = [annotations]
    nrows = np.ceil(images_num / max_cols)
    ncols = max_cols
    if nrows == 1:
        ncols = images_num % (max_cols + 1)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, frameon=False, figsize=figsize)
    if not isinstance(axs, np.ndarray):
        axs = np.array(axs)
    axs = np.ndarray.flatten(axs)
    
    ax_idx = 0
    
     
    for i, image in enumerate(images):
        ax = axs[ax_idx]
        ax_idx = ax_idx + 1
        img = load_image(image, dataset_path)
        text_overlay = []
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(str(image["id"]))
        
        image_anns = annotations[i]


        for anns in image_anns:
            kp = anns["keypoints"]
            
            for i in range(int(len(kp)/3)):
                x = kp[(3 * i)]
                y = kp[(3 * i) + 1]
                v = kp[(3 * i) + 2]
                if(v > 0):
                    center = (x, y)
                    kp_name = keypoint_names[i]
                    color = np.array(dataset_info["keypoint_info"][i]["color"])/255
                    if joints:
                        ax.add_patch(Circle(center, radius=joints_r, color=color, fill=True, alpha=alpha))
                    if joints_name:
                        text_overlay.append(
                            ax.text(x=x, y=y, s=kp_name, color=color, fontsize=12))
            
            for i,k in enumerate(keypoint_connections):
                kp_id1 = k[0]-1
                kp_id2 = k[1]-1

                kp1_x = kp[(3 * kp_id1)]
                kp1_y = kp[(3 * kp_id1) + 1]
                kp1_v = kp[(3 * kp_id1) + 2]
                
                kp2_x = kp[(3 * kp_id2)]
                kp2_y = kp[(3 * kp_id2) + 1]
                kp2_v = kp[(3 * kp_id2) + 2]
                
                if kp1_v == 0 or kp2_v == 0:
                    continue

                color = np.array(dataset_info["skeleton_info"][i]["color"])/255
                x = [kp1_x, kp2_x]
                y = [kp1_y, kp2_y]
                if connections:
                    ax.add_line(Line2D(x, y, lw=connectins_w, color=color, alpha=alpha))





def random_images(annotations, path, n):
    imgidx_to_annidx, id2idx = image_idx_to_ann_idx(annotations)

    idxs_len = only_with_annotations(imgidx_to_annidx)
    iamges_id = get_random_idx(list(idxs_len.keys()), n-1)
    am = np.argmax(list(idxs_len.values()))
    iamges_id.append(list(idxs_len.keys())[am])

    anns = []
    imgs = []
    for j in range(len(iamges_id)):
        imgs.append(annotations["images"][iamges_id[j]])

        anns.append([])
        idx = imgidx_to_annidx[iamges_id[j]]
        for i in idx:
            anns[j].append(annotations["annotations"][i])

    visualize_image_annotations(imgs, anns, path, joints_r=3, figsize=(15, len(iamges_id )*3), alpha=0.7)