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
| SphericalRandomForest |       0.6228 |      0.0467 |           0.7272 |            7.2711 |                0.4406 |
| SphericalTree         |       0.6192 |      0.0522 |           0.6852 |            0.6250 |                0.0082 |
| RandomForest          |       0.6014 |      0.0690 |           0.7167 |            0.2475 |                0.0068 |
| ObliqueRandomForest   |       0.5955 |      0.0492 |           0.7133 |            0.2273 |                0.0175 |
| ExtraTrees            |       0.5813 |      0.0182 |           0.6888 |            0.1078 |                0.0093 |
| CART                  |       0.5482 |      0.0422 |           0.6468 |            0.0067 |                0.0012 |

## iris (classification)

| model                 |   score_mean |   score_std |   secondary_mean |   fit_time_mean_s |   predict_time_mean_s |
|:----------------------|-------------:|------------:|-----------------:|------------------:|----------------------:|
| ObliqueRandomForest   |       0.9534 |      0.0117 |           0.9533 |            0.2033 |                0.0292 |
| ExtraTrees            |       0.9469 |      0.0110 |           0.9467 |            0.2470 |                0.0199 |
| RandomForest          |       0.9469 |      0.0230 |           0.9467 |            0.2255 |                0.0065 |
| CART                  |       0.9338 |      0.0117 |           0.9333 |            0.0077 |                0.0012 |
| SphericalRandomForest |       0.9269 |      0.0577 |           0.9267 |            5.3556 |                0.0997 |
| SphericalTree         |       0.8738 |      0.0309 |           0.8733 |            0.3354 |                0.0049 |

## prnn_crabs (classification)

| model                 |   score_mean |   score_std |   secondary_mean |   fit_time_mean_s |   predict_time_mean_s |
|:----------------------|-------------:|------------:|-----------------:|------------------:|----------------------:|
| ObliqueRandomForest   |       0.9850 |      0.0003 |           0.9850 |            0.2272 |                0.0127 |
| ExtraTrees            |       0.9652 |      0.0170 |           0.9651 |            0.1406 |                0.0207 |
| RandomForest          |       0.9202 |      0.0375 |           0.9202 |            0.2581 |                0.0113 |
| SphericalRandomForest |       0.8950 |      0.0269 |           0.8949 |           14.8487 |                0.1812 |
| CART                  |       0.8699 |      0.0185 |           0.8699 |            0.0133 |                0.0014 |
| SphericalTree         |       0.8249 |      0.0452 |           0.8247 |            0.4710 |                0.0065 |

## 1027_ESL (regression)

| model                 |   score_mean |   score_std |   secondary_mean |   fit_time_mean_s |   predict_time_mean_s |
|:----------------------|-------------:|------------:|-----------------:|------------------:|----------------------:|
| SphericalRandomForest |       0.8424 |      0.0313 |           0.5625 |            5.0698 |                0.4790 |
| ObliqueRandomForest   |       0.8405 |      0.0149 |           0.5667 |            0.1202 |                0.0093 |
| RandomForest          |       0.8217 |      0.0249 |           0.5983 |            0.1209 |                0.0095 |
| ExtraTrees            |       0.8168 |      0.0153 |           0.6073 |            0.0841 |                0.0076 |
| CART                  |       0.7627 |      0.0297 |           0.6885 |            0.0070 |                0.0011 |
| SphericalTree         |       0.6982 |      0.0869 |           0.7730 |            0.3798 |                0.0137 |

## 192_vineyard (regression)

| model                 |   score_mean |   score_std |   secondary_mean |   fit_time_mean_s |   predict_time_mean_s |
|:----------------------|-------------:|------------:|-----------------:|------------------:|----------------------:|
| ObliqueRandomForest   |       0.5347 |      0.0286 |           2.9413 |            0.0845 |                0.0064 |
| SphericalRandomForest |       0.5193 |      0.0548 |           2.9859 |            1.1746 |                0.0484 |
| RandomForest          |       0.5110 |      0.0910 |           2.9974 |            0.0738 |                0.0051 |
| ExtraTrees            |       0.4985 |      0.0926 |           3.0324 |            0.0661 |                0.0105 |
| SphericalTree         |       0.3516 |      0.2265 |           3.3914 |            0.0502 |                0.0023 |
| CART                  |       0.2810 |      0.1612 |           3.6119 |            0.0038 |                0.0025 |
