import json
import os
import random as rnd
import numpy as np
import cv2


class Dataset:

    def __init__(self, paths, only_train=1):
        self.annotations_path = os.path.join(paths["HOME_PATH"], paths["DATASET_ANNOTATIONS_PATH"])
        self.iamge_path = os.path.join(paths["HOME_PATH"], paths["DATASET_IMAGES_PATH"])
        self.annotations = {}
        self.images = []
        self.imgidx_to_idx = {}

        self.categories = []
        self.categori_names = []
        self.activiti_names = []
        self.activiti_imgidx = {}

        self.load_annotations()
        self.add_images(only_train)

        self.remove_without_ann()
        self.get_categories()

    def load_annotations(self):
        with open(self.annotations_path, 'r') as f:
            self.annotations = json.load(f)

    def add_images(self, only_train):
        img_train = []
        if only_train == 1:
            img_train.append(1)
        elif only_train == 0:
            img_train.append(0)
        else:
            img_train.append(1)
            img_train.append(0)

        for imgidx in range(len(self.annotations['annolist'])):

            if self.annotations['img_train'][imgidx] in img_train:
                self.add_image(imgidx,
                               self.annotations['annolist'][imgidx],
                               self.annotations['img_train'][imgidx],
                               self.annotations['single_person'][imgidx],
                               self.annotations['act'][imgidx])

    def add_image(self, imgidx, annolist, img_train, single_person, act):
        new_image = Image(imgidx, self.iamge_path, annolist, img_train, single_person, act)
        self.images.append(new_image)

    def remove_without_ann(self):
        new_images = []
        for img in self.images:
            if img.has_annotations:
                new_images.append(img)
        self.images = new_images

    def get_image_by_id(self, imgidx):
        if isinstance(imgidx, list):
            imgs = []
            for i in imgidx:
                imgs.append(self.images[self.imgidx_to_idx[i]])
            return imgs
        else:
            return self.images[self.imgidx_to_idx[imgidx]]

    def get_random_images(self, num):
        ind = rnd.sample(range(0, len(self.images)), num)
        return [self.images[i] for i in ind]

    def get_percent_imgidx_from(self, activity_name, percent, random=True):
        if percent > 1:
            percent = percent / 100
        if activity_name in self.activiti_imgidx:
            act_len = len(self.activiti_imgidx[activity_name])
            num = int(np.floor(act_len * percent))
            if num == 0 and act_len >= 1:
                num = 1
            if random:
                return self.get_random_imgindx_form(activity_name, num)
            else:
                step = int(np.floor(act_len / num))
                idx = self.activiti_imgidx[activity_name][::step]
                return idx[:num]
        else:
            print('No activity: {}'.format(activity_name))
            return None

    def get_random_imgindx_form(self, activity_name, num):
        if activity_name in self.activiti_imgidx:
            if num > len(self.activiti_imgidx[activity_name]):
                print('Only {} images in {} activity'.format(len(self.activiti_imgidx[activity_name]), activity_name))
                num = len(self.activiti_imgidx[activity_name])
            ind = rnd.sample(range(0, len(self.activiti_imgidx[activity_name])), num)
            return [self.activiti_imgidx[activity_name][i] for i in ind]
        else:
            print('No activity: {}'.format(activity_name))
            return None

    def get_random_images_from(self, activity_name, num):
        if activity_name in self.activiti_imgidx:
            ind = self.get_random_imgindx_form(activity_name, num)
            imgs = self.get_image_by_id(ind)
            return imgs
        else:
            print('No activity: {}'.format(activity_name))
            return None

    def get_categories(self):
        cats = []
        acts = []
        act_imgidx = {'no_act': []}
        imi_to_i = {}
        i = 0
        for img in self.images:
            cat = img.cat_name
            act = img.act_name
            aid = img.act_id
            if img.imgidx:
                imi_to_i[img.imgidx] = i
                i += 1
            if not cat or not act:
                act_imgidx['no_act'].append(img.imgidx)
                continue
            if cat not in cats:
                self.categories.append({"supercategory": cat, "id": aid, "name": act})
                cats.append(cat)
                acts.append(act)
                act_imgidx[act] = [img.imgidx]
            else:
                if act not in acts:
                    self.categories.append({"supercategory": cat, "id": aid, "name": act})
                    acts.append(act)
                    act_imgidx[act] = [img.imgidx]
                else:
                    act_imgidx[act].append(img.imgidx)

        self.imgidx_to_idx = imi_to_i
        self.activiti_imgidx = act_imgidx
        self.categori_names = cats
        self.activiti_names = acts


class Image:
    keypoints = ['head', 'upper_neck', 'thorax',
                 'lsho', 'rsho',
                 'lelb', 'relb',
                 'lwri', 'rwri',
                 'pelvis', 'lhip', 'rhip',
                 'lknee', 'rknee',
                 'lankl', 'rankl']

    idx_to_jnt = {0: 'rankl', 1: 'rknee', 2: 'rhip', 5: 'lankl', 4: 'lknee', 3: 'lhip',
                  6: 'pelvis', 7: 'thorax', 8: 'upper_neck', 11: 'relb', 10: 'rwri', 9: 'head',
                  12: 'rsho', 13: 'lsho', 14: 'lelb', 15: 'lwri'}

    connections = [[0, 1], [1, 2], [2, 6], [6, 7], [7, 8],
                   [8, 9], [5, 4], [4, 3], [3, 6], [10, 11],
                   [11, 12], [12, 7], [15, 14], [14, 13],
                   [13, 7]]

    def __init__(self, imgidx, image_path, annolist, img_train, single_person, act):
        self.imgidx = imgidx
        self.image_name = annolist['image']['name']
        self.image_path = image_path

        self.subset = img_train
        self.separated_individuals_ridx = single_person

        self.has_annotations = True
        self.head_rect = []
        self.points = []
        self.scale = []
        self.position = []
        self.annorect = []
        self.process_annorect(annolist['annorect'])

        self.cat_name = act['cat_name']
        self.act_name = act['act_name']
        self.act_id = act['act_id']

    def process_annorect(self, annorect):
        if isinstance(annorect, list):
            self.annorect = annorect
        else:
            self.annorect = [annorect]

        if not self.annorect:
            # print('image id {} no annotations'.format(self.imgidx)
            self.has_annotations = False

        for ann in self.annorect:
            if self.subset == 1:
                if 'annopoints' in ann.keys() and 'scale' in ann.keys() and 'objpos' in ann.keys():
                    if isinstance(ann['annopoints'], dict):
                        self.head_rect.append({'x1': ann['x1'], 'y1': ann['y1'], 'x2': ann['x2'], 'y2': ann['y2']})
                        if isinstance(ann['annopoints']['point'], list):
                            self.points.append(ann['annopoints']['point'])
                        else:
                            self.points.append([ann['annopoints']['point']])
                        if not self.points:
                            self.has_annotations = False
                        self.scale.append(ann['scale'])
                        self.position.append(ann['objpos'])
                    else:
                        # print('image id {} no annotpoints {}'.format(self.imgidx, ann['annopoints']))
                        self.has_annotations = False
                else:
                    # print('image id {} missing annotations, provided annotations {}'.format(self.imgidx, ann.keys()))
                    self.has_annotations = False

            if self.subset == 0:
                if 'scale' in ann.keys() and 'objpos' in ann.keys():
                    self.scale.append(ann['scale'])
                    self.position.append(ann['objpos'])
                else:
                    # print('image id {} missing annotations, provided annotations {}'.format(self.imgidx, ann.keys()))
                    self.has_annotations = False

    def keypoints_to_coco(self):
        keypoints = []
        for joints in self.points:
            kp = [0] * len(self.keypoints) * 3
            for joint in joints:
                name = self.idx_to_jnt[joint['id']]
                i = self.keypoints.index(name)
                x = joint['x']
                y = joint['y']
                vis = joint['is_visible']
                if vis:
                    v = 2
                else:
                    v = 1
                kp[i * 3] = x
                kp[(i * 3) + 1] = y
                kp[(i * 3) + 2] = v
            keypoints.append(kp)
        return keypoints

    def to_coco_format(self):
        path = os.path.join(self.image_path, self.image_name)
        img = cv2.imread(path)
        height, width, _ = img.shape
        kp = self.keypoints_to_coco()

        image = {
            'file_name': self.image_name,
            'height': height,
            'width': width,
            'id': self.imgidx
        }
        annotations = []

        for i in range(len(self.points)):
            hr = self.head_rect[i]
            annotation = {
                "id": ((i + 1) * 100000) + self.imgidx,
                "image_id": self.imgidx,
                "category_id": self.act_id,
                "bbox": [hr['x1'], hr['y1'], hr['x2'] - hr['x1'], hr['y2'] - hr['y1']],
                "keypoints": kp[i]
            }
            annotations.append(annotation)

        return image, annotations
