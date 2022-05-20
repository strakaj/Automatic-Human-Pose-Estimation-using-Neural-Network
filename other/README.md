# Human Pose Estimation Methods

## DeepPose: Human Pose Estimation via Deep Neural Networks [arXiv](https://arxiv.org/pdf/1312.4659.pdf)
![DeepPose: Human Pose Estimation via Deep Neural Networks](assets/DeepPose.png)
- First CNN-based model
- Predicts body joints position
- 2 stages - first stage predicts joints position, second stage crops parts of the image around a joint and refine its position

## Efficient Object Localization Using Convolutional Networks [arXiv](https://arxiv.org/pdf/1411.4280.pdf)
![Efficient Object Localization Using Convolutional Networks](assets/EOL.png)
- Generates heatmaps
- Heatmap predicts the probability of the joint at each pixel - better than direct regression
- Detection on a variety of image scales
- Model consists of a module for heatmap generation, a module to crop convolution features at joints location, and a module for fine-tuning

## Convolutional Pose Machines [arXiv](https://arxiv.org/pdf/1602.00134.pdf)
![Convolutional Pose Machines](assets/CPM.png)
- Consists of an image feature computation module followed by a prediction module
- Prediction module can be repeated
- Input for the first module is image and output are heatmaps, input for prediction modules are heatmaps from the previous stage, and original input image

## Human Pose Estimation with Iterative Error Feedback  [arXiv](https://arxiv.org/pdf/1507.06550)
![Human Pose Estimation with Iterative Error Feedback](assets/IEF.png)
- Model predicts what is wrong with the current estimates and correct them iteratively
- Input for the model is an image and estimated heatmaps from the previous step, output is a correction of joints position

## Stacked Hourglass Networks for Human Pose Estimation [arXiv](https://arxiv.org/pdf/1603.06937.pdf)
![Stacked Hourglass Networks for Human Pose Estimation](assets/StackedHourglass.png)
- Consist of multiple steps of pooling layers and upsampling layers
- Captures information on multiple scales
- Uses skip connections between downsampling and upsampling layers

## Simple Baselines for Human Pose Estimation and Tracking [arXiv](https://arxiv.org/pdf/1804.06208.pdf)
![Simple Baselines for Human Pose Estimation and Tracking](assets/Baseline.png)
- Simple and effective
- Consist of a ResNet and deconvolutional layers
- ResNet downsample the image to low resolution, deconvolution upsample feature maps back to high resolution
- No skip connections between ResNet and deconvolution layers

## Deep High-Resolution Representation Learning for Human Pose Estimation  [arXiv](https://arxiv.org/pdf/1902.09212.pdf)
![Deep High-Resolution Representation Learning for Human Pose Estimation](assets/HRNet.png)
- Maintains a high-resolution and adds parallel subnetworks with lower resolution 
- Parallel branches are connected

## Comparison

|                  |  MPII | COCO |
|------------------|:-----:|:----:|
| DeepPose         | -     |   -  |
| EOL              | 82.0  |   -  |
| CPM              | 87.95 |   -  |
| IEF              | 81.3  |   -  |
| StackedHourglass | 90.9  |   -  |
| Baseline         | 91.5  | 79   |
| HRNet            |   -   | 80.5 |

# Results
## SimpleBaseline2D - results
**test-set**

| Model        | AP    | AP .5 | AP .75 | AP (M) | AP (L) | AR    | AR .5 | AR .75 | AR (M) | AR (L) |
|--------------|-------|-------|--------|--------|--------|-------|-------|--------|--------|--------|
| Pretrained   | 0.700 | 0.909 | 0.779  | 0.668  | 0.758  | 0.756 | 0.945 | 0.830  | 0.715  | 0.813  |
| Trained 30ep | 0.565 | 0.842 | 0.616  | 0.536  | 0.618  | 0.627 | 0.883 | 0.680  | 0.585  | 0.683  |


**val-set**

| Model        | AP    | AP .5 | AP .75 | AP (M) | AP (L) | AR    | AR .5 | AR .75 | AR (M) | AR (L) |
|--------------|-------|-------|--------|--------|--------|-------|-------|--------|--------|--------|
| Original     | 0.704 | 0.886 | 0.783  | 0.671  | 0.772  | 0.763 | 0.929 | 0.834  | 0.721  | 0.824  |
| Pretrained   | 0.724 | 0.915 | 0.804  | 0.697  | 0.765  | 0.756 | 0.930 | 0.823  | 0.723  | 0.804  |
| Trained 30ep | 0.574 | 0.832 | 0.630  | 0.546  | 0.614  | 0.609 | 0.847 | 0.661  | 0.576  | 0.657  |




## MMPose - results
**test-set**

| Model        | AP    | AP .5 | AP .75 | AP (M) | AP (L) | AR    | AR .5 | AR .75 | AR (M) | AR (L) |
|--------------|-------|-------|--------|--------|--------|-------|-------|--------|--------|--------|
| Pretrained   | 0.711 | 0.916 | 0.796  | 0.677  | 0.765  | 0.766 | 0.950 | 0.842  | 0.724  | 0.822  |


**val-set**

| Model        | AP    | AP .5 | AP .75 | AP (M) | AP (L) | AR    | AR .5 | AR .75 | AR (M) | AR (L) |
|--------------|-------|-------|--------|--------|--------|-------|-------|--------|--------|--------|
| Original     | 0.718 | 0.898 | 0.795  |        |        | 0.773 | 0.937 |        |        |        |
| Pretrained   | 0.718 | 0.898 | 0.795  | 0.646  | 0.745  | 0.773 | 0.937 | 0.841  | 0.729  | 0.837  |


## TransPose - results

**val-set**

| Model        | AP    | AP .5 | AP .75 | AP (M) | AP (L) | AR    | AR .5 | AR .75 | AR (M) | AR (L) |
|--------------|-------|-------|--------|--------|--------|-------|-------|--------|--------|--------|
| Original     | 0.717 | 0.889 | 0.788  | 0.680  | 0.786  | 0.771 | 0.930 | 0.836  | 0.727  | 0.835  |
| Pretrained   | 0.751 | 0.926 | 0.826  | 0.719  | 0.796  | 0.778 | 0.932 | 0.841  | 0.743  | 0.830  |
