import json
import cv2
import os
import numpy as np
import copy
import torch
import torch.utils.data as torch_data
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


def load_ann(path):
    with open(path) as f:
        ann = json.load(f)
    return ann


def load_img(path, name):
    img = cv2.imread(os.path.join(path, name))
    img = img[:, :, ::-1]
    return img


def show_img(img, ann=None):
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    ax.imshow(img)
    patches = []
    r = 3
    lw = 3
    if ann:
        for a in ann:
            bbox = a['bbox']
            patches.append(Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], color='red', linewidth=lw, fill=False))
            kp = a['keypoints']
            for i in range(0, len(kp), 3):
                if kp[i + 2] > 0:
                    patches.append(Circle((kp[i], kp[i + 1]), radius=r, color='red'))
        for p in patches:
            ax.add_patch(p)


def get_iid_to_ann(ann):
    iid_to_ann = {}
    if ann:
        for a in ann['images']:
            iid_to_ann[a['id']] = []
    if ann:
        for a in ann['annotations']:
            iid_to_ann[a['image_id']].append(a)
    return iid_to_ann


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def keypoints_annotations(num_joints, kp):
    joints_3d = np.zeros((num_joints, 2), dtype=float)
    joints_3d_vis = np.zeros((num_joints, 2), dtype=float)
    for ipt in range(num_joints):
        joints_3d[ipt, 0] = kp[ipt * 3 + 0]
        joints_3d[ipt, 1] = kp[ipt * 3 + 1]
        # joints_3d[ipt, 2] = kp[ipt * 3 + 2]
        t_vis = kp[ipt * 3 + 2]
        if t_vis > 1:
            t_vis = 1
        joints_3d_vis[ipt, 0] = t_vis
        joints_3d_vis[ipt, 1] = t_vis
        # joints_3d_vis[ipt, 2] = 0
    return joints_3d, joints_3d_vis


def keypoints_to_list(kp):
    kp_list = []
    for r in kp:
        kp_list.append(r[0])
        kp_list.append(r[1])
    return kp_list


def get_final_preds(coords, center, scale, width, height):
    """
    :param coords: (1, 17, 2)
    :param center: (1, 2)
    :param scale: (1, 2)
    :param width: 192
    :param height: 256
    :return:
    """
    preds = coords.copy()
    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(coords[i], center[i], scale[i], [width[i], height[i]])
    return preds


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)

    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def fliplr_joints(joints, joints_vis, matched_parts):
    """
    flip coords
    """
    # Change left-right pairs
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = \
            joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints * joints_vis, joints_vis


def fliplr_joints2(joints, matched_parts):
    """
    flip coords
    """
    # Change left-right pairs
    for b in range(joints.shape[0]):
        for pair in matched_parts:
            joints[b, pair[0], :], joints[b, pair[1], :] = \
                joints[b, pair[1], :], joints[b, pair[0], :].copy()

    return joints


def flip_back(inp, kp, flip):
    kp_n = []
    for i in range(inp.shape[0]):
        kp_o = kp[i] * [inp.shape[3], inp.shape[2]]
        im_o = np.zeros([inp.shape[2], inp.shape[3], 3])
        transformed = flip(image=im_o, keypoints=kp_o, class_labels=['person'])
        kp_n.append(transformed['keypoints'])
    return np.array(kp_n)


class COCODataset(torch_data.Dataset):
    def __init__(self, dataset_filename, img_path, transform=None, augmentations=None, mode='train',
                 relative_positions=False, return_vis=False, crop_w=192, crop_h=256):
        dataset_ann = load_ann(dataset_filename)
        self.data_type = type(dataset_ann)
        self.mode = mode
        self.relative_positions = relative_positions
        self.return_vis = return_vis
        self.image_width = crop_w
        self.image_height = crop_h
        self.image_size = (self.image_width, self.image_height)
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.dataset_joints = 17
        self.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                           [9, 10], [11, 12], [13, 14], [15, 16]]
        self.pixel_std = 200
        self.data = []

        self.transform = transform
        self.augmentations = augmentations

        if (mode == 'train') or (isinstance(dataset_ann, dict)):
            iid_to_ann = get_iid_to_ann(dataset_ann)
            for img in dataset_ann["images"]:
                iid = img['id']
                anns = iid_to_ann[iid]
                for ann in anns:
                    if ann['num_keypoints'] > 0:
                        data = ann
                        if self.augmentations:
                            center, scale = None, None
                        else:
                            center, scale = self._box2cs(ann['bbox'])
                        data['crop_box'] = ann['bbox']
                        data['file_path'] = os.path.join(img_path, img['file_name'])
                        data['center'] = center
                        data['scale'] = scale
                        data['num_keypoints'] = ann['num_keypoints']
                        data['score'] = 1
                        data['image_id'] = ann['image_id']
                        data['joints'], data['joints_vis'] = keypoints_annotations(self.dataset_joints,
                                                                                   ann['keypoints'])
                        data['id'] = ann['id']
                        self.data.append(data)
        else:
            for ann in dataset_ann:
                data = dict()
                center, scale = self._box2cs(ann['bbox'])
                data['crop_box'] = ann['bbox']
                file_name = '%012d.jpg' % ann['image_id']
                data['file_path'] = os.path.join(img_path, file_name)
                data['center'] = center
                data['scale'] = scale
                data['score'] = ann['score']
                data['image_id'] = ann['image_id']
                data['id'] = ann['image_id']
                self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        img = cv2.imread(data['file_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        c = copy.deepcopy(data['center'])
        s = copy.deepcopy(data['scale'])
        r = 0

        if self.mode == 'train':
            joints = copy.deepcopy(data['joints'])
            joints_vis = copy.deepcopy(data['joints_vis'])

        # apply augmentations
        if self.augmentations:
            transformed = self.augmentations(image=img,
                                             keypoints=data['joints'],
                                             bboxes=[data['bbox']],
                                             class_labels=['person'])
            if len(transformed['bboxes']) > 0:
                bbx = np.array(transformed['bboxes'][0])
                img = transformed['image']
                c, s = self._box2cs(bbx)
                joints = copy.deepcopy(np.array(transformed['keypoints']))
            else:
                c, s = self._box2cs(data['bbox'])
                data['center'] = c
                data['scale'] = s
                img = cv2.imread(data['file_path'])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                c = data['center']
                s = data['scale']

        trans = get_affine_transform(c, s, r, self.image_size)
        img_inp = cv2.warpAffine(
            img,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        if self.mode == 'train':
            for i in range(self.dataset_joints):
                if data['joints_vis'][i, 0] > 0.0:
                    joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)
                else:
                    joints[i, 0:2] = [0, 0]

            if self.relative_positions:
                joints[:, 0:2] /= [self.image_width, self.image_height]

                if self.augmentations:
                    for trs in transformed['replay']['transforms']:
                        if trs['__class_fullname__'] == 'HorizontalFlip' or trs['__class_fullname__'] == 'VerticalFlip':
                            if trs['applied']:
                                joints, joints_vis = fliplr_joints(
                                    joints, joints_vis, self.flip_pairs)

        if self.transform:
            img_trans = self.transform(img_inp)
        else:
            img_trans = img_inp

        if self.mode == 'test':
            meta_data = {'scale': s,
                         'center': c,
                         'height': self.image_height,
                         'width': self.image_width,
                         'image_id': data['image_id'],
                         'score': data['score'],
                         'file_path': data['file_path'],
                         'id': data['id']
                         }
            return img_trans.float(), meta_data

        if self.return_vis:
            return img_trans.float(), torch.tensor(joints).float(), torch.tensor(joints_vis).float()
        return img_trans.float(), torch.tensor(joints).float()

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros([2], dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale


class COCODatasetPredict(torch_data.Dataset):
    def __init__(self, annotations, images, transform=None, crop_w=192, crop_h=256):
        dataset_ann = annotations
        self.images = images
        self.image_width = crop_w
        self.image_height = crop_h
        self.image_size = (self.image_width, self.image_height)
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.dataset_joints = 17
        self.pixel_std = 200
        self.data = []

        self.transform = transform

        for ann in dataset_ann:
            data = dict()
            crop_box = ann['bbox']
            center, scale = self._box2cs(ann['bbox'])
            data['crop_box'] = crop_box
            data['center'] = center
            data['scale'] = scale
            data['image_id'] = ann['image_id']
            self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        img = self.images[data['image_id']]
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        c = data['center']
        s = data['scale']
        r = 0

        trans = get_affine_transform(c, s, r, self.image_size)
        img_inp = cv2.warpAffine(
            img,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        if self.transform:
            img_trans = self.transform(img_inp)
        else:
            img_trans = img_inp

        meta_data = {'scale': s,
                     'center': c,
                     'height': self.image_height,
                     'width': self.image_width,
                     'image_id': data['image_id'],
                     'bbox': data['crop_box']
                     }

        return img_trans, meta_data

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros([2], dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale
