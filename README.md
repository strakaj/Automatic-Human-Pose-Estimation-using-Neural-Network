# Automatic Human Pose Estimation using Neural Network

|               |                    |
|---------------|--------------------|
| ![v2](assets/v2.gif) | ![v1](assets/v1.gif) |


This repository contains codes used in the master’s thesis: **Automatic Human Pose Estimation using Neural Network**.

In `model` directory are two models. Models are created as modifications of the DePOTR model: [POTR](https://github.com/mhruz/POTR). Results can be found [here](https://github.com/strakaj/Automatic-Human-Pose-Estimation-using-Neural-Network/tree/main/scripts). 

## Installation
Create conda environment for models:
```
conda create -n de_potr python=3.7 pip
conda activate de_potr
```

Install package dependencies:

```
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
or
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.1 -c pytorch
pip install -r models/requirements.txt
```

Build CUDA operator:
```
cd ./models/de_potr/deformable_potr/models/ops
sh ./make.sh
```

## Data

- Data for training and validation can be downloaded from [COCO dataset]( https://cocodataset.org/#download).

- Pretrained weights for both models are available at [GoogleDrive](https://drive.google.com/drive/folders/1sZ5pej78TXvcp41wpdM3O6anq0J3vG0g?usp=sharing).

- Detected person results for test and validation set  was downloade from [GoogleDrive](https://drive.google.com/drive/folders/1fRUDNUDxe9fjqcRZ2bnF_TKMlO0nB_dk)
