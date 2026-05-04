# Small PMLB Spherical Tree Benchmark

Primary score is balanced accuracy for classification and R2 for regression.
Secondary score is accuracy for classification and RMSE for regression.

- CV folds: 3
- Forest estimators: 50
- Spherical center candidates per node: 8
- Spherical radius candidates per center: 64

## breast_cancer (classification)

| model                 |   score_mean |   score_std |   secondary_mean |   fit_time_mean_s |   predict_time_mean_s |
|:----------------------|-------------:|------------:|-----------------:|------------------:|----------------------:|
| SphericalRandomForest |       0.6191 |      0.0222 |           0.7273 |            0.0650 |                0.0057 |
| RandomForest          |       0.6014 |      0.0690 |           0.7167 |            0.0417 |                0.0026 |
| ObliqueRandomForest   |       0.5955 |      0.0492 |           0.7133 |            0.0522 |                0.0024 |
| ExtraTrees            |       0.5813 |      0.0182 |           0.6888 |            0.0403 |                0.0026 |
| SphericalTree         |       0.5761 |      0.0563 |           0.6573 |            0.0035 |                0.0004 |
| CART                  |       0.5482 |      0.0422 |           0.6468 |            0.0023 |                0.0004 |

## iris (classification)

| model                 |   score_mean |   score_std |   secondary_mean |   fit_time_mean_s |   predict_time_mean_s |
|:----------------------|-------------:|------------:|-----------------:|------------------:|----------------------:|
| ObliqueRandomForest   |       0.9534 |      0.0117 |           0.9533 |            0.0435 |                0.0020 |
| SphericalRandomForest |       0.9534 |      0.0117 |           0.9533 |            0.0574 |                0.0103 |
| ExtraTrees            |       0.9469 |      0.0110 |           0.9467 |            0.0462 |                0.0034 |
| RandomForest          |       0.9469 |      0.0230 |           0.9467 |            0.0463 |                0.0022 |
| CART                  |       0.9338 |      0.0117 |           0.9333 |            0.0028 |                0.0003 |
| SphericalTree         |       0.8681 |      0.0498 |           0.8667 |            0.0023 |                0.0003 |

## prnn_crabs (classification)

| model                 |   score_mean |   score_std |   secondary_mean |   fit_time_mean_s |   predict_time_mean_s |
|:----------------------|-------------:|------------:|-----------------:|------------------:|----------------------:|
| ObliqueRandomForest   |       0.9850 |      0.0003 |           0.9850 |            0.0520 |                0.0032 |
| ExtraTrees            |       0.9652 |      0.0170 |           0.9651 |            0.0434 |                0.0043 |
| RandomForest          |       0.9202 |      0.0375 |           0.9202 |            0.0509 |                0.0021 |
| SphericalRandomForest |       0.8852 |      0.0384 |           0.8848 |            0.0856 |                0.0056 |
| CART                  |       0.8699 |      0.0185 |           0.8699 |            0.0030 |                0.0008 |
| SphericalTree         |       0.8048 |      0.1072 |           0.8054 |            0.0029 |                0.0005 |

## 1027_ESL (regression)

| model                 |   score_mean |   score_std |   secondary_mean |   fit_time_mean_s |   predict_time_mean_s |
|:----------------------|-------------:|------------:|-----------------:|------------------:|----------------------:|
| ObliqueRandomForest   |       0.8405 |      0.0149 |           0.5667 |            0.0484 |                0.0037 |
| SphericalRandomForest |       0.8353 |      0.0258 |           0.5754 |            0.0850 |                0.0045 |
| RandomForest          |       0.8217 |      0.0249 |           0.5983 |            0.0372 |                0.0032 |
| ExtraTrees            |       0.8168 |      0.0153 |           0.6073 |            0.0409 |                0.0059 |
| CART                  |       0.7627 |      0.0297 |           0.6885 |            0.0025 |                0.0003 |
| SphericalTree         |       0.7518 |      0.0485 |           0.7065 |            0.0043 |                0.0003 |

## 192_vineyard (regression)

| model                 |   score_mean |   score_std |   secondary_mean |   fit_time_mean_s |   predict_time_mean_s |
|:----------------------|-------------:|------------:|-----------------:|------------------:|----------------------:|
| ObliqueRandomForest   |       0.5347 |      0.0286 |           2.9413 |            0.0420 |                0.0032 |
| RandomForest          |       0.5110 |      0.0910 |           2.9974 |            0.0348 |                0.0023 |
| SphericalRandomForest |       0.5077 |      0.0858 |           3.0318 |            0.0286 |                0.0029 |
| ExtraTrees            |       0.4985 |      0.0926 |           3.0324 |            0.0229 |                0.0019 |
| CART                  |       0.2810 |      0.1612 |           3.6119 |            0.0015 |                0.0003 |
| SphericalTree         |       0.2568 |      0.4639 |           3.6719 |            0.0024 |                0.0006 |
