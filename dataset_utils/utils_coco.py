import json
import numpy as np
import random as rnd


def load_annotations(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def image_idx_to_ann_idx(annotation):
    """
    imgid_to_annid - dict
        key = img_id
        value = [ann_ids]
    """
    imgidx_to_annidx = {}
    
    id2idx = {}
    for i, img in enumerate(annotation["images"]):
        imgidx_to_annidx[i] = []
        id2idx[img["id"]] = i
        
    for i, ann in enumerate(annotation["annotations"]):
        if(ann["num_keypoints"] > 0): #only annotations with keypoints
            imgidx_to_annidx[id2idx[ann["image_id"]]].append(i)
    
    return imgidx_to_annidx, id2idx

def get_image_annotations(iidx2aidx, img_idx, annotation):
    annotations = []
    for idx in iidx2aidx[img_idx]:
        annotations.append(annotation["annotations"][idx])
    return annotations

def get_stats(annotations):
    
    imgidx_to_annidx, id2idx = image_idx_to_ann_idx(annotations)
    
    num_images = len(imgidx_to_annidx)
    num_images_ann = 0
    nannotations = []
    keypoints = annotations["categories"][0]["keypoints"]
    nkeypoints = {}
    for k in keypoints:
        nkeypoints[k] = []
    
    for k in imgidx_to_annidx:
        annidx = imgidx_to_annidx[k]
        
        # image has annotations
        if annidx:
            num_images_ann += 1
            nannotations.append(len(annidx))
            
            # check annotation keypoints
            for idx in annidx:
                ann = annotations["annotations"][idx]
                kp = ann["keypoints"]
                for i, k in enumerate(keypoints):
                    kpidx = ((i+1)*3) - 1
                    visibility = kp[kpidx]
                    nkeypoints[k].append(visibility)
                    
    keypounts_not_labeled = {}
    keypounts_not_visible = {}
    keypounts_visible = {}
    
    for kp in nkeypoints:
        keypounts_not_labeled[kp] = nkeypoints[kp].count(0)
        keypounts_not_visible[kp] = nkeypoints[kp].count(1)
        keypounts_visible[kp] = nkeypoints[kp].count(2)
    
    visible = np.sum(list(keypounts_visible.values()))
    non_visible = np.sum(list(keypounts_not_visible.values()))
    

    avg_annotations = np.round(np.sum(nannotations)/num_images_ann,2)
    
    print("Number of images in dataset: {}".format(num_images))
    print("Number of images in dataset with annotation: {}".format(num_images_ann))
    print()
    print('Annotations: {}'.format(np.sum(nannotations)))
    print('Annotations average in one image : {}'.format(avg_annotations))
    print()
    print('Visible joints: {} ({:.2f}%)'.format(visible, visible/(visible+non_visible) * 100))
    print('Nonvisible joints: {} ({:.2f}%)'.format(non_visible, non_visible/(visible+non_visible) * 100))
    
    return keypoints, keypounts_visible, keypounts_not_visible, nannotations


def only_with_annotations(imgidx_to_annidx):
    ann = {}
    for i in imgidx_to_annidx:
        if(imgidx_to_annidx[i]):
            ann[i] = len(imgidx_to_annidx[i])
    
    return ann
        
def get_random_idx(idxs, n):
    idx = rnd.sample(range(0, len(idxs)), n)
    r_idxs = []
    for i in idx:
        r_idxs.append(idxs[i])
    return r_idxs

