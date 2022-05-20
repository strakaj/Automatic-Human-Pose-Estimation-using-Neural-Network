### Full dataset

![](03-04.png)

### Epoch 60 AP results
|     | AP    | AP .5 | AP .75 | AP (M) | AP (L) | AR    | AR .5 | AR .75 | AR (M) | AR (L) |
|-----|-------|-------|--------|--------|--------|-------|-------|--------|--------|--------|
| lin | 0.483 | 0.773 | 0.526  | 0.479  | 0.523  | 0.617 | 0.870 | 0.680  | 0.583  | 0.665  |
| cos | 0.492 | 0.777 | 0.536  | 0.479  | 0.542  | 0.623 | 0.872 | 0.686  | 0.583  | 0.679  |


![](ap.png)


### Results analysis
http://www.vision.caltech.edu/mronchi/papers/ICCV17_PoseErrorDiagnosis_PAPER.pdf
#### OKS
|                        |                       |
|------------------------|-----------------------|
| ![](oks/oks_gauss.jpg) | ![](oks/oks_full.jpg) |



| model | all                        | medium                      | large                       |
|-------|----------------------------|-----------------------------|-----------------------------|
| lin   | ![](oks/lin60_oks_all.png) | ![](oks/lin60_oks_medium.png) | ![](oks/lin60_oks_large.png)  |
| cos   | ![](oks/cos60_oks_all.png)  | ![](oks/cos60_oks_medium.png) | ![](oks/cos60_oks_large.png) |

#### Error
##### lin
| OKS = 0.95                        | OKS = 0.9                        | OKS = 0.85                     |
|-----------------------------------|----------------------------------|--------------------------------|
| ![](errors/lin60_err_0.95.png)    | ![](errors/lin60_err_0.9.png)    | ![](errors/lin60_err_0.85.png) |
| OKS = 0.8                         | OKS = 0.75                       | OKS = 0.5                     |
| ![](errors/lin60_err_0.8.png)     | ![](errors/lin60_err_0.75.png)   | ![](errors/lin60_err_0.5.png) |

##### cos
| OKS = 0.95                        | OKS = 0.9                        | OKS = 0.85                     |
|-----------------------------------|----------------------------------|--------------------------------|
| ![](errors/cos60_err_0.95.png)    | ![](errors/cos60_err_0.9.png)    | ![](errors/cos60_err_0.85.png) |
| OKS = 0.8                         | OKS = 0.75                       | OKS = 0.5                     |
| ![](errors/cos60_err_0.8.png)     | ![](errors/cos60_err_0.75.png)   | ![](errors/cos60_err_0.5.png) |


### Baseline, Transpose
| Model      | all                                      | OKS = 0.95                                   | OKS = 0.75                                   | OKS = 0.5                                   |
|------------|------------------------------------------|----------------------------------------------|----------------------------------------------|---------------------------------------------|
| Baseline   | ![](oks/baseline/baseline_oks_all.png)   | ![](errors/baseline/baseline_err_0.95.png)   | ![](errors/baseline/baseline_err_0.75.png)   | ![](errors/baseline/baseline_err_0.5.png)   |
| Transpose  | ![](oks/transpose/transpose_oks_all.png) | ![](errors/transpose/transpose_err_0.95.png) | ![](errors/transpose/transpose_err_0.75.png) | ![](errors/transpose/transpose_err_0.5.png) |
| DEPOTR cos | ![](oks/cos60_oks_all.png)               | ![](errors/cos60_err_0.95.png)               | ![](errors/cos60_err_0.75.png)               | ![](errors/cos60_err_0.5.png)               |


### YOLOv5 + DePOTR

![](gifs/v3_kpls.gif)
![](gifs/v1_kpls.gif) 
![](gifs/v2_kpls.gif) 



