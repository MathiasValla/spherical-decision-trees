# Full PMLB Spherical/Oblique Tree Benchmark

Primary score is balanced accuracy for classification and R2 for regression.
Secondary score is accuracy for classification and RMSE for regression.
Dataset-level summary includes only datasets with all requested model/fold rows.

- CV folds: 3
- Forest estimators: 50
- Spherical center candidates per node: 8
- Spherical radius candidates per center: 64
- Last resume max samples per dataset: 1000
- Dataset timeout: 600
- Model timeout: 120
- Complete datasets summarized: 249
- Skipped/error datasets: 35

## Sample Usage

| task           | capped        |   n_datasets |
|:---------------|:--------------|-------------:|
| classification | full rows     |          116 |
| classification | sample capped |           14 |
| regression     | full rows     |           95 |
| regression     | sample capped |           24 |

## Model Summary

| task           | model                 |   n_datasets |   score_rank_mean |   fit_time_rank_mean |   score_mean |   fit_time_mean_s |   predict_time_mean_s |
|:---------------|:----------------------|-------------:|------------------:|---------------------:|-------------:|------------------:|----------------------:|
| classification | RandomForest          |          130 |            2.6346 |               4.1769 |       0.7564 |            0.3378 |                0.0205 |
| classification | SphericalRandomForest |          130 |            2.7808 |               5.8154 |       0.7559 |            1.8903 |                0.0706 |
| classification | ObliqueRandomForest   |          130 |            2.8846 |               4.9769 |       0.7557 |            0.4335 |                0.0362 |
| classification | CART                  |          130 |            3.7038 |               1.6846 |       0.7355 |            0.0745 |                0.0033 |
| classification | ObliqueTree           |          130 |            4.0769 |               2.0538 |       0.7296 |            0.1036 |                0.0030 |
| classification | SphericalTree         |          130 |            4.9192 |               2.2923 |       0.6830 |            0.1383 |                0.0048 |
| regression     | RandomForest          |          119 |            1.7395 |               4.4874 |       0.7237 |            0.1374 |                0.0023 |
| regression     | ObliqueRandomForest   |          119 |            1.7731 |               5.1681 |       0.7042 |            0.1633 |                0.0031 |
| regression     | SphericalRandomForest |          119 |            3.0924 |               5.3445 |       0.5744 |            0.1102 |                0.0039 |
| regression     | CART                  |          119 |            3.7857 |               2.0672 |       0.4937 |            0.0057 |                0.0003 |
| regression     | ObliqueTree           |          119 |            4.8193 |               1.9160 |       0.3566 |            0.0066 |                0.0003 |
| regression     | SphericalTree         |          119 |            5.7899 |               2.0168 |      -0.0538 |            0.0045 |                0.0003 |

## First Errors

| task           | dataset                 | stage   |   model |   fold | error_type   | error                      |
|:---------------|:------------------------|:--------|--------:|-------:|:-------------|:---------------------------|
| classification | australian              | fetch   |     nan |    nan | ValueError   | Dataset not found in PMLB. |
| classification | auto                    | fetch   |     nan |    nan | ValueError   | Dataset not found in PMLB. |
| classification | breast                  | fetch   |     nan |    nan | ValueError   | Dataset not found in PMLB. |
| classification | breast_cancer_wisconsin | fetch   |     nan |    nan | ValueError   | Dataset not found in PMLB. |
| classification | breast_w                | fetch   |     nan |    nan | ValueError   | Dataset not found in PMLB. |
| classification | buggyCrx                | fetch   |     nan |    nan | ValueError   | Dataset not found in PMLB. |
| classification | car                     | fetch   |     nan |    nan | ValueError   | Dataset not found in PMLB. |
| classification | cleve                   | fetch   |     nan |    nan | ValueError   | Dataset not found in PMLB. |
| classification | cleveland               | fetch   |     nan |    nan | ValueError   | Dataset not found in PMLB. |
| classification | cleveland_nominal       | fetch   |     nan |    nan | ValueError   | Dataset not found in PMLB. |
| classification | cmc                     | fetch   |     nan |    nan | ValueError   | Dataset not found in PMLB. |
| classification | colic                   | fetch   |     nan |    nan | ValueError   | Dataset not found in PMLB. |
| classification | contraceptive           | fetch   |     nan |    nan | ValueError   | Dataset not found in PMLB. |
| classification | credit_a                | fetch   |     nan |    nan | ValueError   | Dataset not found in PMLB. |
| classification | credit_g                | fetch   |     nan |    nan | ValueError   | Dataset not found in PMLB. |
| classification | crx                     | fetch   |     nan |    nan | ValueError   | Dataset not found in PMLB. |
| classification | diabetes                | fetch   |     nan |    nan | ValueError   | Dataset not found in PMLB. |
| classification | flare                   | fetch   |     nan |    nan | ValueError   | Dataset not found in PMLB. |
| classification | german                  | fetch   |     nan |    nan | ValueError   | Dataset not found in PMLB. |
| classification | glass                   | fetch   |     nan |    nan | ValueError   | Dataset not found in PMLB. |
| classification | heart_c                 | fetch   |     nan |    nan | ValueError   | Dataset not found in PMLB. |
| classification | heart_h                 | fetch   |     nan |    nan | ValueError   | Dataset not found in PMLB. |
| classification | heart_statlog           | fetch   |     nan |    nan | ValueError   | Dataset not found in PMLB. |
| classification | horse_colic             | fetch   |     nan |    nan | ValueError   | Dataset not found in PMLB. |
| classification | house_votes_84          | fetch   |     nan |    nan | ValueError   | Dataset not found in PMLB. |
| classification | hungarian               | fetch   |     nan |    nan | ValueError   | Dataset not found in PMLB. |
| classification | pima                    | fetch   |     nan |    nan | ValueError   | Dataset not found in PMLB. |
| classification | prnn_fglass             | fetch   |     nan |    nan | ValueError   | Dataset not found in PMLB. |
| classification | solar_flare_1           | fetch   |     nan |    nan | ValueError   | Dataset not found in PMLB. |
| classification | solar_flare_2           | fetch   |     nan |    nan | ValueError   | Dataset not found in PMLB. |
