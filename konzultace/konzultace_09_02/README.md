## Medium dataset

- ### 06, 07, 08
  - různé konstanty učení
  - absolutní souřadnice bodů
  - zvýšení batch size z 2 na 32



![](medium_dataset_06-07-08.png)


- ### 10, 12, 14, 15, 17
  - relativní souřadnice bodů
  - 12 vs 17
    - 12 trénováno s konstantou učení 0.001
    - 17 začíná s konstantou učení 0.0001, která během 5 epoch roste na 0.001, poté zůstává konstantní

**val-set**

| Medium dataset 10 | AP    | AP .5 | AP .75 | AP (M) | AP (L) | AR    | AR .5 | AR .75 | AR (M) | AR (L) |
|-------------------|-------|-------|--------|--------|--------|-------|-------|--------|--------|--------|
|                   | 0.044 | 0.189 | 0.004  | 0.069  | 0.040  | 0.133 | 0.410 | 0.054  | 0.121  | 0.151  |

![](medium_dataset_10-12-14-15-17.png)

## Full dataset


![](full_dataset_06-07.png)

**val-set**

| Full dataset 6 | AP    | AP .5 | AP .75 | AP (M) | AP (L) | AR    | AR .5 | AR .75 | AR (M) | AR (L) |
|----------------|-------|-------|--------|--------|--------|-------|-------|--------|--------|--------|
|                | 0.088 | 0.327 | 0.017  | 0.115  | 0.087  | 0.197 | 0.537 | 0.104  | 0.181  | 0.209  |

