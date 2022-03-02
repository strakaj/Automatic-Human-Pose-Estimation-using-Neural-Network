
|                      |                      |
|----------------------|----------------------|
| ![](gifs/v3_bb.gif)  | ![](gifs/v3_kpl.gif) | 
| ![](gifs/v1_bb.gif)  | ![](gifs/v1_kpl.gif) |
| ![](gifs/v2_bb.gif)  | ![](gifs/v2_kpl.gif) |




### Full dataset

![](00-01-ep50.png)

### Epoch 50 AP results
|                          | AP    | AP .5 | AP .75 | AP (M) | AP (L) | AR    | AR .5 | AR .75 | AR (M) | AR (L) |
|--------------------------|-------|-------|--------|--------|--------|-------|-------|--------|--------|--------|
| full_dataset_const       | 0.427 | 0.752 | 0.448  | 0.424  | 0.462  | 0.557 | 0.852 | 0.617  | 0.520  | 0.610  |
| full_dataset_warm_up-lin | 0.503 | 0.784 | 0.552  | 0.496  | 0.547  | 0.632 | 0.877 | 0.696  | 0.595  | 0.685  |


![](ap.png)

### Epoch 30 vs. 40 average pixel error

![](hist0_err_30_40.png)

![](hist2_err_30_40.png)



### Epoch 40 vs. 50 average pixel error

![](hist0_err_40_50.png)

![](hist2_err_40_50.png)