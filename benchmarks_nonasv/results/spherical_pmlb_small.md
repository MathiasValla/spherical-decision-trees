# Small PMLB Spherical Tree Benchmark

Primary score is balanced accuracy for classification and R2 for regression.
Secondary score is accuracy for classification and RMSE for regression.

- CV folds: 3
- Forest estimators: 50
- Spherical center candidates per node: 8
- Spherical radius candidates per center: 64
- Spherical center strategies: default

## breast_cancer (classification)

| model                 |   score_mean |   score_std |   secondary_mean |   fit_time_mean_s |   predict_time_mean_s |
|:----------------------|-------------:|------------:|-----------------:|------------------:|----------------------:|
| SphericalRandomForest |       0.6226 |      0.0191 |           0.7272 |            0.0654 |                0.0032 |
| SphericalTree         |       0.6153 |      0.0418 |           0.6784 |            0.0042 |                0.0005 |
| RandomForest          |       0.6014 |      0.0690 |           0.7167 |            0.0447 |                0.0026 |
| ObliqueRandomForest   |       0.5955 |      0.0492 |           0.7133 |            0.0525 |                0.0035 |
| ExtraTrees            |       0.5813 |      0.0182 |           0.6888 |            0.0579 |                0.0029 |
| CART                  |       0.5482 |      0.0422 |           0.6468 |            0.0021 |                0.0003 |

## iris (classification)

| model                 |   score_mean |   score_std |   secondary_mean |   fit_time_mean_s |   predict_time_mean_s |
|:----------------------|-------------:|------------:|-----------------:|------------------:|----------------------:|
| ObliqueRandomForest   |       0.9534 |      0.0117 |           0.9533 |            0.0440 |                0.0018 |
| SphericalRandomForest |       0.9534 |      0.0117 |           0.9533 |            0.0476 |                0.0032 |
| ExtraTrees            |       0.9469 |      0.0110 |           0.9467 |            0.0278 |                0.0020 |
| RandomForest          |       0.9469 |      0.0230 |           0.9467 |            0.0387 |                0.0019 |
| CART                  |       0.9338 |      0.0117 |           0.9333 |            0.0018 |                0.0003 |
| SphericalTree         |       0.8807 |      0.0913 |           0.8800 |            0.0016 |                0.0004 |

## prnn_crabs (classification)

| model                 |   score_mean |   score_std |   secondary_mean |   fit_time_mean_s |   predict_time_mean_s |
|:----------------------|-------------:|------------:|-----------------:|------------------:|----------------------:|
| ObliqueRandomForest   |       0.9850 |      0.0003 |           0.9850 |            0.0490 |                0.0019 |
| ExtraTrees            |       0.9652 |      0.0170 |           0.9651 |            0.0222 |                0.0020 |
| RandomForest          |       0.9202 |      0.0375 |           0.9202 |            0.0434 |                0.0020 |
| SphericalRandomForest |       0.9198 |      0.0319 |           0.9199 |            0.0735 |                0.0053 |
| CART                  |       0.8699 |      0.0185 |           0.8699 |            0.0028 |                0.0003 |
| SphericalTree         |       0.8454 |      0.0218 |           0.8451 |            0.0033 |                0.0003 |

## 1027_ESL (regression)

| model                 |   score_mean |   score_std |   secondary_mean |   fit_time_mean_s |   predict_time_mean_s |
|:----------------------|-------------:|------------:|-----------------:|------------------:|----------------------:|
| ObliqueRandomForest   |       0.8405 |      0.0149 |           0.5667 |            0.0530 |                0.0038 |
| SphericalRandomForest |       0.8353 |      0.0258 |           0.5754 |            0.0582 |                0.0056 |
| RandomForest          |       0.8217 |      0.0249 |           0.5983 |            0.0419 |                0.0024 |
| ExtraTrees            |       0.8168 |      0.0153 |           0.6073 |            0.0221 |                0.0021 |
| CART                  |       0.7627 |      0.0297 |           0.6885 |            0.0028 |                0.0004 |
| SphericalTree         |       0.7518 |      0.0485 |           0.7065 |            0.0039 |                0.0005 |

## 192_vineyard (regression)

| model                 |   score_mean |   score_std |   secondary_mean |   fit_time_mean_s |   predict_time_mean_s |
|:----------------------|-------------:|------------:|-----------------:|------------------:|----------------------:|
| ObliqueRandomForest   |       0.5347 |      0.0286 |           2.9413 |            0.0334 |                0.0029 |
| RandomForest          |       0.5110 |      0.0910 |           2.9974 |            0.0360 |                0.0025 |
| SphericalRandomForest |       0.5077 |      0.0858 |           3.0318 |            0.0338 |                0.0045 |
| ExtraTrees            |       0.4985 |      0.0926 |           3.0324 |            0.0225 |                0.0026 |
| CART                  |       0.2810 |      0.1612 |           3.6119 |            0.0018 |                0.0003 |
| SphericalTree         |       0.2568 |      0.4639 |           3.6719 |            0.0021 |                0.0004 |
