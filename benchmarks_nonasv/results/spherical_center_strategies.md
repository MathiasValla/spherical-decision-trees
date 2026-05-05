# Small PMLB Spherical Tree Benchmark

Primary score is balanced accuracy for classification and R2 for regression.
Secondary score is accuracy for classification and RMSE for regression.

- CV folds: 3
- Forest estimators: 50
- Spherical center candidates per node: 8
- Spherical radius candidates per center: 64
- Spherical center strategies: random, target, hybrid

## breast_cancer (classification)

| model                         |   score_mean |   score_std |   secondary_mean |   fit_time_mean_s |   predict_time_mean_s |
|:------------------------------|-------------:|------------:|-----------------:|------------------:|----------------------:|
| SphericalRandomForest[hybrid] |       0.6226 |      0.0191 |           0.7272 |            0.1106 |                0.0063 |
| SphericalRandomForest[target] |       0.6205 |      0.0601 |           0.7238 |            0.1047 |                0.0065 |
| SphericalRandomForest[random] |       0.6191 |      0.0222 |           0.7273 |            0.1192 |                0.0073 |
| SphericalTree[target]         |       0.6153 |      0.0418 |           0.6784 |            0.0042 |                0.0007 |
| SphericalTree[hybrid]         |       0.6026 |      0.0386 |           0.6609 |            0.0041 |                0.0007 |
| RandomForest                  |       0.6014 |      0.0690 |           0.7167 |            0.0706 |                0.0032 |
| ObliqueRandomForest           |       0.5955 |      0.0492 |           0.7133 |            0.0767 |                0.0035 |
| ExtraTrees                    |       0.5813 |      0.0182 |           0.6888 |            0.0643 |                0.0048 |
| SphericalTree[random]         |       0.5761 |      0.0563 |           0.6573 |            0.0033 |                0.0006 |
| CART                          |       0.5482 |      0.0422 |           0.6468 |            0.0025 |                0.0005 |

## iris (classification)

| model                         |   score_mean |   score_std |   secondary_mean |   fit_time_mean_s |   predict_time_mean_s |
|:------------------------------|-------------:|------------:|-----------------:|------------------:|----------------------:|
| ObliqueRandomForest           |       0.9534 |      0.0117 |           0.9533 |            0.0538 |                0.0019 |
| SphericalRandomForest[hybrid] |       0.9534 |      0.0117 |           0.9533 |            0.0882 |                0.0033 |
| SphericalRandomForest[random] |       0.9534 |      0.0117 |           0.9533 |            0.0668 |                0.0054 |
| SphericalRandomForest[target] |       0.9534 |      0.0117 |           0.9533 |            0.0477 |                0.0039 |
| ExtraTrees                    |       0.9469 |      0.0110 |           0.9467 |            0.0465 |                0.0037 |
| RandomForest                  |       0.9469 |      0.0230 |           0.9467 |            0.0430 |                0.0016 |
| CART                          |       0.9338 |      0.0117 |           0.9333 |            0.0023 |                0.0005 |
| SphericalTree[hybrid]         |       0.8807 |      0.0913 |           0.8800 |            0.0027 |                0.0003 |
| SphericalTree[target]         |       0.8807 |      0.0913 |           0.8800 |            0.0041 |                0.0007 |
| SphericalTree[random]         |       0.8681 |      0.0498 |           0.8667 |            0.0027 |                0.0003 |

## prnn_crabs (classification)

| model                         |   score_mean |   score_std |   secondary_mean |   fit_time_mean_s |   predict_time_mean_s |
|:------------------------------|-------------:|------------:|-----------------:|------------------:|----------------------:|
| ObliqueRandomForest           |       0.9850 |      0.0003 |           0.9850 |            0.1162 |                0.0040 |
| ExtraTrees                    |       0.9652 |      0.0170 |           0.9651 |            0.0991 |                0.0034 |
| RandomForest                  |       0.9202 |      0.0375 |           0.9202 |            0.1113 |                0.0047 |
| SphericalRandomForest[hybrid] |       0.9198 |      0.0319 |           0.9199 |            0.1357 |                0.0057 |
| SphericalRandomForest[target] |       0.8947 |      0.0310 |           0.8948 |            0.1598 |                0.0109 |
| SphericalRandomForest[random] |       0.8852 |      0.0384 |           0.8848 |            0.3555 |                0.0113 |
| CART                          |       0.8699 |      0.0185 |           0.8699 |            0.0052 |                0.0007 |
| SphericalTree[target]         |       0.8454 |      0.0218 |           0.8451 |            0.0065 |                0.0020 |
| SphericalTree[hybrid]         |       0.8348 |      0.0386 |           0.8351 |            0.0074 |                0.0006 |
| SphericalTree[random]         |       0.8048 |      0.1072 |           0.8054 |            0.0061 |                0.0014 |

## 1027_ESL (regression)

| model                         |   score_mean |   score_std |   secondary_mean |   fit_time_mean_s |   predict_time_mean_s |
|:------------------------------|-------------:|------------:|-----------------:|------------------:|----------------------:|
| ObliqueRandomForest           |       0.8405 |      0.0149 |           0.5667 |            0.0611 |                0.0038 |
| SphericalRandomForest[random] |       0.8353 |      0.0258 |           0.5754 |            0.0970 |                0.0057 |
| SphericalRandomForest[target] |       0.8280 |      0.0334 |           0.5871 |            0.0807 |                0.0072 |
| SphericalRandomForest[hybrid] |       0.8246 |      0.0292 |           0.5936 |            0.0865 |                0.0070 |
| RandomForest                  |       0.8217 |      0.0249 |           0.5983 |            0.0622 |                0.0029 |
| ExtraTrees                    |       0.8168 |      0.0153 |           0.6073 |            0.0410 |                0.0047 |
| CART                          |       0.7627 |      0.0297 |           0.6885 |            0.0032 |                0.0006 |
| SphericalTree[random]         |       0.7518 |      0.0485 |           0.7065 |            0.0031 |                0.0003 |
| SphericalTree[hybrid]         |       0.7209 |      0.0354 |           0.7482 |            0.0066 |                0.0016 |
| SphericalTree[target]         |       0.7165 |      0.0820 |           0.7477 |            0.0033 |                0.0005 |

## 192_vineyard (regression)

| model                         |   score_mean |   score_std |   secondary_mean |   fit_time_mean_s |   predict_time_mean_s |
|:------------------------------|-------------:|------------:|-----------------:|------------------:|----------------------:|
| ObliqueRandomForest           |       0.5347 |      0.0286 |           2.9413 |            0.0417 |                0.0028 |
| RandomForest                  |       0.5110 |      0.0910 |           2.9974 |            0.0369 |                0.0023 |
| SphericalRandomForest[random] |       0.5077 |      0.0858 |           3.0318 |            0.0453 |                0.0039 |
| ExtraTrees                    |       0.4985 |      0.0926 |           3.0324 |            0.0244 |                0.0023 |
| SphericalRandomForest[target] |       0.4920 |      0.0480 |           3.0721 |            0.0492 |                0.0037 |
| SphericalRandomForest[hybrid] |       0.4837 |      0.0690 |           3.0868 |            0.0478 |                0.0038 |
| SphericalTree[hybrid]         |       0.4324 |      0.1768 |           3.1843 |            0.0027 |                0.0005 |
| CART                          |       0.2810 |      0.1612 |           3.6119 |            0.0020 |                0.0003 |
| SphericalTree[random]         |       0.2568 |      0.4639 |           3.6719 |            0.0021 |                0.0004 |
| SphericalTree[target]         |       0.2398 |      0.2719 |           3.7100 |            0.0026 |                0.0005 |
