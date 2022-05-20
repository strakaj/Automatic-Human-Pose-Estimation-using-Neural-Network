import albumentations as A

transform = A.Compose(
    [A.HorizontalFlip(p=0.5), 
     A.RandomBrightnessContrast(p=0.6), 
     A.RGBShift(p=0.3),
     #A.ShiftScaleRotate(p=0.4, rotate_limit=10, border_mode=1), 
     A.Rotate (p=0.4, limit=10),
     A.GaussianBlur(p=0.5), 
     A.GaussNoise(p=0.5, var_limit=(10.0, 50.0))
     ],
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
    bbox_params=A.BboxParams(format='coco', label_fields=['class_labels'])
)