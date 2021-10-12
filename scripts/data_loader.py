import os
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2
from skimage import io, transform


class MPIIDataset(Dataset):
    def __init__(self, imgidx, dataset, one_person=True, transform=None):
        self.dataset = dataset
        self.one_person = one_person
        self.transform = transform
        self.root_path = dataset.iamge_path
        if one_person:
            self.imgidx = []
            for i in imgidx:
                img = dataset.images[dataset.imgidx_to_idx[i]]
                if len(img.annorect) == 1:
                    self.imgidx.append(i)
        else:
            self.imgidx = imgidx

    def __len__(self):
        return len(self.imgidx)

    def __getitem__(self, idx):
        imgidx = self.imgidx[idx]
        image_data = self.dataset.images[self.dataset.imgidx_to_idx[imgidx]]
        image_path = os.path.join(image_data.image_path, image_data.image_name)
        image = cv2.imread(image_path)
        image = image[:, :, ::-1]
        label = np.array(image_data.keypoints_to_coco())

        if self.transform:
            (image, label) = self.transform((image, label))

        return image, label


class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, keypoints = sample

        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        for i in range(int(len(keypoints[0]) / 3)):

            keypoints[0][3 * i] = keypoints[0][3 * i] * (new_w / w) / self.output_size[0]
            keypoints[0][(3 * i) + 1] = keypoints[0][(3 * i) + 1] * (new_h / h) / self.output_size[1]

            if keypoints[0][(3 * i) + 2] == 0:
                keypoints[0][(3 * i) + 2] = -1
            else:
                keypoints[0][(3 * i) + 2] = 1

        return img, keypoints


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, keypoints = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return (torch.from_numpy(image).float(),
                torch.from_numpy(keypoints).float())
