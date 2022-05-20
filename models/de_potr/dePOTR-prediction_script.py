import argparse
import torch
import h5py
import json
import os
import numpy as np

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from deformable_potr.models.deformable_potr import DeformablePOTR
from deformable_potr.models.backbone import build_backbone
from deformable_potr.models.deformable_transformer import build_deformable_transformer

from coco_dataset.coco_dataset import COCODataset, get_final_preds, flip_back, fliplr_joints2
from coco_dataset.coco_evaluate import evaluate, _print_name_value

from pycocotools.coco import COCO
import albumentations as A

# Arguments
parser = argparse.ArgumentParser("DETR HPOES Standalone Annotations Script", add_help=False)
parser.add_argument("--weights_file", type=str, default="out/checkpoint.pth",
                    help="Path to the pretrained model's chekpoint (.pth)")
parser.add_argument("--output_file", type=str, default="out.h5", help="Path to the .h5 file to write into")
parser.add_argument("--device", default="cuda", help="Device to be used")
parser.add_argument("--batch_size", default=1, type=int, help="Batch size")

parser.add_argument('--transformations', default=['totensor', 'normalize'], type=str, nargs='*')
parser.add_argument('--relative_positions', action='store_true', help='')
parser.add_argument('--crop_w', default=192, type=int)
parser.add_argument('--crop_h', default=256, type=int)

parser.add_argument('--data_path',
                    default="/auto/plzen1/home/strakajk/Datasets/COCO/person_detection_results/COCO_val2017_detections_AP_H_56_person.json",
                    type=str, help="Path to the training dataset coco annotation file.")
parser.add_argument('--images_path', default="/auto/plzen1/home/strakajk/Datasets/COCO/val2017",
                    type=str, help="Path to the image folder.")

parser.add_argument('--image_set', default="test",
                    type=str, help="test or val")
parser.add_argument('--validation_ann_path',
                    default="/auto/plzen1/home/strakajk/Datasets/COCO/annotations/person_keypoints_val2017.json",
                    type=str, help="")
parser.add_argument('--res_name', default="", type=str, help="")
parser.add_argument('--flip_test', action='store_true', help='')

args = parser.parse_args()
device = torch.device(args.device)

# Load the input data and checkpoint
print("Loading the input data and checkpoints.")
checkpoint = torch.load(args.weights_file, map_location=device)

all_preds_out = []

# Construct the model from the loaded data
model = DeformablePOTR(
    build_backbone(checkpoint["args"]),
    build_deformable_transformer(checkpoint["args"]),
    num_queries=checkpoint["args"].num_queries,
    num_feature_levels=checkpoint["args"].num_feature_levels
)
model.load_state_dict(checkpoint["model"])
model.eval()
model.to(device)

print("Constructed model successfully.")

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

trans = []
if 'totensor' in args.transformations:
    trans.append(transforms.ToTensor())
elif 'normalize' in args.transformations:
    trans.append(normalize)

dataset_test = COCODataset(
    args.data_path,
    args.images_path,
    transforms.Compose(trans),
    mode='test',
    relative_positions=args.relative_positions,
    crop_w=args.crop_w,
    crop_h=args.crop_h)
data_loader = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=0, shuffle=False)

num_samples = len(dataset_test)
num_joints = checkpoint["args"].num_queries
all_preds = np.zeros((num_samples, num_joints, 3), dtype=np.float32)
all_boxes = np.zeros((num_samples, 6))
file_path = []
idx = 0
img_size = dataset_test.image_size

# Make predictions
for i, (samples, meta_data) in enumerate(data_loader):
    print(i)
    samples = samples.to(device, dtype=torch.float32)

    results = model(samples).detach().cpu().numpy()

    if args.relative_positions:
        results = results * img_size

    if args.flip_test:
        flip = A.ReplayCompose([A.HorizontalFlip(p=1)],
                               keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

        # this part is ugly, because pytorch has not supported negative index
        input_flipped = np.flip(samples.cpu().numpy(), 3).copy()
        input_flipped = torch.from_numpy(input_flipped).cuda()
        output_flipped = model(input_flipped).detach().cpu().numpy()
        output_flipped = flip_back(samples, output_flipped, flip)
        output_flipped = fliplr_joints2(output_flipped, dataset_test.flip_pairs)

        results = (results + output_flipped) * 0.5

    # ---------------------- predictions in coco format -------------------------- #
    preds = get_final_preds(results,
                            meta_data['center'].numpy(),
                            meta_data['scale'].numpy(),
                            meta_data['width'].numpy(),
                            meta_data['height'].numpy())

    num_images = samples.shape[0]

    s = meta_data['scale'].numpy()
    c = meta_data['center'].numpy()
    a = np.prod(s * 200, 1)
    score = meta_data['score'].numpy()
    file_path.extend(meta_data['file_path'])

    maxvals = np.ones([num_images, num_joints, 1])
    for j, sc in enumerate(score):
        maxvals[j, :, :] = maxvals[j, :, :] * sc

    all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
    all_preds[idx:idx + num_images, :, 2:3] = maxvals
    all_boxes[idx:idx + num_images, 0:2] = c
    all_boxes[idx:idx + num_images, 2:4] = s
    all_boxes[idx:idx + num_images, 4] = a
    all_boxes[idx:idx + num_images, 5] = score

    idx += num_images
    # ------------------------------------------------------------------------- #

    for j in range(preds.shape[0]):
        dct = {'joints': preds[j].tolist(), 'image_id': int(meta_data['image_id'][j]), 'id': int(meta_data['id'][j])}
        all_preds_out.append(dct)

# ---------------------- create coco predictions -------------------------- #
val_path = args.validation_ann_path
coco = COCO(val_path)
cats = [cat['name'] for cat in coco.loadCats(coco.getCatIds())]
print(cats)
classes = ['__background__'] + cats
print(classes)
image_set = args.image_set
cfg = {'NUM_JOINTS': num_joints, 'IN_VIS_THRE': 0.2, 'OKS_THRE': 0.9, 'CLASSES': classes, 'IMAGE_SET': image_set,
       'COCO': coco}
folder = os.path.dirname(args.output_file)
name_values, perf_indicator = evaluate(cfg,
                                       all_preds,
                                       folder,
                                       all_boxes,
                                       file_path, res_name=args.res_name)

full_arch_name = 'resnet'
if isinstance(name_values, list):
    for name_value in name_values:
        _print_name_value(name_value, full_arch_name)
else:
    _print_name_value(name_values, full_arch_name)
# ------------------------------------------------------------------------- #

print("Predictions were made.")

folder = os.path.dirname(args.output_file)
with open(os.path.join(folder, 'output_data' + args.res_name + '.json'), 'w') as f:
    json.dump(all_preds_out, f)

print("Data was successfully structured and saved.")
