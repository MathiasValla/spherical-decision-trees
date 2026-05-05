# Spherical Heuristic Analysis

This report extends the PMLB benchmark regime analysis with target and predictor
descriptors. The response variable `spherical_wins` means that the best
spherical model beats the best non-spherical model in the same comparison set.

Target scale for regression is included because it was requested, but it should
not be interpreted causally: regression target values can be arbitrarily rescaled
without changing the prediction problem, and the benchmark scores use R2.

## Outputs

- `spherical_pmlb_dataset_descriptors.csv`
- `spherical_pmlb_heuristic_enriched.csv`
- `spherical_pmlb_heuristic_rules.csv`
- `spherical_pmlb_heuristic_context_summary.csv`
- `spherical_pmlb_heuristic_fold_margins.csv`
- `figures/spherical_heuristic_top_rules.png`

- `figures/spherical_heuristic_poor_rules.png`


## Strong Practical Wins

`consistent_practical_win` requires the spherical model to beat the selected
non-spherical comparator on every fold, to exceed a practical margin threshold
(0.03 balanced-accuracy/R2 points for
classification, 0.05 for regression), and to be
larger than two fold-level standard errors. With only three folds this is not a
formal significance test; it is a conservative practical screen.

| comparison   | task           |   n_datasets |   consistent_practical_wins |   large_margin_wins |   consistent_practical_losses |   large_margin_losses |
|:-------------|:---------------|-------------:|----------------------------:|--------------------:|------------------------------:|----------------------:|
| forests      | classification |          130 |                          10 |                   7 |                            15 |                    13 |
| forests      | regression     |          119 |                           0 |                   0 |                            58 |                    47 |
| overall      | classification |          130 |                           8 |                   5 |                            22 |                    20 |
| overall      | regression     |          119 |                           0 |                   0 |                            58 |                    47 |
| trees        | classification |          130 |                           7 |                   5 |                            60 |                    53 |
| trees        | regression     |          119 |                           0 |                   2 |                            80 |                    88 |

## Top Strong Cases

| comparison   | task           | dataset                                     | best_spherical_model   | best_non_spherical_model   |   fold_margin_mean |   fold_margin_min |   n_used_samples |   n_features |   n_over_p | target_balance        | target_value_scale   | predictor_profile                                |
|:-------------|:---------------|:--------------------------------------------|:-----------------------|:---------------------------|-------------------:|------------------:|-----------------:|-------------:|-----------:|:----------------------|:---------------------|:-------------------------------------------------|
| forests      | classification | parity5+5                                   | SphericalRandomForest  | RandomForest               |              0.317 |             0.181 |             1124 |           10 |    112.400 | balanced_binary       | not_regression       | mostly_binary;sparse                             |
| overall      | classification | parity5+5                                   | SphericalRandomForest  | ObliqueTree                |              0.188 |             0.125 |             1124 |           10 |    112.400 | balanced_binary       | not_regression       | mostly_binary;sparse                             |
| trees        | classification | balance_scale                               | SphericalTree          | ObliqueTree                |              0.136 |             0.080 |              625 |            4 |    156.250 | moderate_multiclass   | not_regression       | mostly_discrete;balanced                         |
| forests      | classification | analcatdata_boxing1                         | SphericalRandomForest  | RandomForest               |              0.103 |             0.003 |              120 |            3 |     40.000 | balanced_binary       | not_regression       | mostly_continuous                                |
| trees        | classification | ring                                        | SphericalTree          | CART                       |              0.102 |             0.087 |             5000 |           20 |    250.000 | balanced_binary       | not_regression       | mostly_continuous;balanced                       |
| overall      | classification | balance_scale                               | SphericalTree          | ObliqueRandomForest        |              0.092 |             0.010 |              625 |            4 |    156.250 | moderate_multiclass   | not_regression       | mostly_discrete;balanced                         |
| forests      | classification | nursery                                     | SphericalRandomForest  | RandomForest               |              0.087 |             0.064 |             5000 |            8 |    625.000 | unbalanced_multiclass | not_regression       | mostly_discrete                                  |
| trees        | classification | analcatdata_cyyoung9302                     | SphericalTree          | CART                       |              0.076 |             0.021 |               92 |           10 |      9.200 | moderate_binary       | not_regression       | mostly_continuous;scale_unbalanced;redundant     |
| overall      | classification | analcatdata_cyyoung9302                     | SphericalTree          | RandomForest               |              0.066 |             0.030 |               92 |           10 |      9.200 | moderate_binary       | not_regression       | mostly_continuous;scale_unbalanced;redundant     |
| trees        | classification | twonorm                                     | SphericalTree          | ObliqueTree                |              0.064 |             0.049 |             5000 |           20 |    250.000 | balanced_binary       | not_regression       | mostly_continuous;balanced                       |
| trees        | classification | tic_tac_toe                                 | SphericalTree          | CART                       |              0.062 |             0.021 |              958 |            9 |    106.444 | moderate_binary       | not_regression       | mostly_discrete;balanced                         |
| overall      | classification | tic_tac_toe                                 | SphericalRandomForest  | RandomForest               |              0.061 |             0.022 |              958 |            9 |    106.444 | moderate_binary       | not_regression       | mostly_discrete;balanced                         |
| forests      | classification | tic_tac_toe                                 | SphericalRandomForest  | RandomForest               |              0.061 |             0.022 |              958 |            9 |    106.444 | moderate_binary       | not_regression       | mostly_discrete;balanced                         |
| forests      | classification | krkopt                                      | SphericalRandomForest  | RandomForest               |              0.060 |             0.057 |            28056 |            6 |   4676.000 | unbalanced_multiclass | not_regression       | mostly_discrete;balanced                         |
| overall      | classification | analcatdata_boxing1                         | SphericalRandomForest  | CART                       |              0.057 |             0.022 |              120 |            3 |     40.000 | balanced_binary       | not_regression       | mostly_continuous                                |
| forests      | classification | car_evaluation                              | SphericalRandomForest  | ObliqueRandomForest        |              0.056 |             0.024 |             1728 |            6 |    288.000 | unbalanced_multiclass | not_regression       | mostly_discrete                                  |
| forests      | classification | allrep                                      | SphericalRandomForest  | RandomForest               |              0.049 |             0.041 |             3772 |           29 |    130.069 | unbalanced_multiclass | not_regression       | mostly_continuous;sparse;scale_unbalanced;skewed |
| forests      | classification | GAMETES_Epistasis_2_Way_20atts_0.4H_EDM_1_1 | SphericalRandomForest  | ObliqueRandomForest        |              0.049 |             0.043 |             1600 |           20 |     80.000 | balanced_binary       | not_regression       | mostly_discrete;sparse                           |
| overall      | classification | GAMETES_Epistasis_2_Way_20atts_0.4H_EDM_1_1 | SphericalRandomForest  | ObliqueRandomForest        |              0.049 |             0.043 |             1600 |           20 |     80.000 | balanced_binary       | not_regression       | mostly_discrete;sparse                           |
| trees        | classification | analcatdata_asbestos                        | SphericalTree          | ObliqueTree                |              0.046 |             0.010 |               83 |            3 |     27.667 | balanced_binary       | not_regression       | mostly_continuous;scale_unbalanced;redundant     |
| trees        | classification | yeast                                       | SphericalTree          | CART                       |              0.040 |             0.017 |             1479 |            8 |    184.875 | unbalanced_multiclass | not_regression       | mostly_continuous                                |
| forests      | classification | monk1                                       | SphericalRandomForest  | RandomForest               |              0.038 |             0.027 |              556 |            6 |     92.667 | balanced_binary       | not_regression       | mostly_discrete                                  |
| forests      | classification | mofn_3_7_10                                 | SphericalRandomForest  | RandomForest               |              0.038 |             0.036 |             1324 |           10 |    132.400 | moderate_binary       | not_regression       | mostly_binary;sparse                             |
| overall      | classification | krkopt                                      | SphericalRandomForest  | CART                       |              0.037 |             0.025 |            28056 |            6 |   4676.000 | unbalanced_multiclass | not_regression       | mostly_discrete;balanced                         |
| overall      | classification | GAMETES_Epistasis_2_Way_20atts_0.1H_EDM_1_1 | SphericalRandomForest  | ObliqueTree                |              0.036 |             0.017 |             1600 |           20 |     80.000 | balanced_binary       | not_regression       | mostly_discrete;sparse                           |

## Strong Practical Losses

The same fold screen can be read in the opposite direction:
`consistent_practical_loss` requires the best spherical model to lose to the
best non-spherical comparator on every fold, by at least the practical margin,
and by more than two fold-level standard errors.

| comparison   | task           |   n_datasets |   consistent_practical_wins |   large_margin_wins |   consistent_practical_losses |   large_margin_losses |
|:-------------|:---------------|-------------:|----------------------------:|--------------------:|------------------------------:|----------------------:|
| forests      | classification |          130 |                          10 |                   7 |                            15 |                    13 |
| forests      | regression     |          119 |                           0 |                   0 |                            58 |                    47 |
| overall      | classification |          130 |                           8 |                   5 |                            22 |                    20 |
| overall      | regression     |          119 |                           0 |                   0 |                            58 |                    47 |
| trees        | classification |          130 |                           7 |                   5 |                            60 |                    53 |
| trees        | regression     |          119 |                           0 |                   2 |                            80 |                    88 |

## Top Strong Loss Cases

| comparison   | task       | dataset             | best_spherical_model   | best_non_spherical_model   |   fold_margin_mean |   fold_margin_min |   n_used_samples |   n_features |   n_over_p |   target_balance | target_value_scale   | predictor_profile          |
|:-------------|:-----------|:--------------------|:-----------------------|:---------------------------|-------------------:|------------------:|-----------------:|-------------:|-----------:|-----------------:|:---------------------|:---------------------------|
| trees        | regression | 626_fri_c2_500_50   | SphericalTree          | CART                       |             -1.569 |            -1.660 |              500 |           50 |     10.000 |              nan | medium_target_values | mostly_continuous;balanced |
| trees        | regression | 637_fri_c1_500_50   | SphericalTree          | CART                       |             -1.540 |            -1.699 |              500 |           50 |     10.000 |              nan | low_target_values    | mostly_continuous;balanced |
| trees        | regression | 588_fri_c4_1000_100 | SphericalTree          | CART                       |             -1.536 |            -1.746 |             1000 |          100 |     10.000 |              nan | low_target_values    | mostly_continuous;balanced |
| trees        | regression | 605_fri_c2_250_25   | SphericalTree          | CART                       |             -1.455 |            -1.511 |              250 |           25 |     10.000 |              nan | medium_target_values | mostly_continuous;balanced |
| trees        | regression | 622_fri_c2_1000_50  | SphericalTree          | CART                       |             -1.357 |            -1.494 |             1000 |           50 |     20.000 |              nan | medium_target_values | mostly_continuous;balanced |
| trees        | regression | 643_fri_c2_500_25   | SphericalTree          | CART                       |             -1.349 |            -1.454 |              500 |           25 |     20.000 |              nan | medium_target_values | mostly_continuous;balanced |
| trees        | regression | 582_fri_c1_500_25   | SphericalTree          | CART                       |             -1.335 |            -1.566 |              500 |           25 |     20.000 |              nan | medium_target_values | mostly_continuous;balanced |
| trees        | regression | 618_fri_c3_1000_50  | SphericalTree          | CART                       |             -1.319 |            -1.528 |             1000 |           50 |     20.000 |              nan | low_target_values    | mostly_continuous;balanced |
| trees        | regression | 583_fri_c1_1000_50  | SphericalTree          | CART                       |             -1.302 |            -1.431 |             1000 |           50 |     20.000 |              nan | medium_target_values | mostly_continuous;balanced |
| trees        | regression | 645_fri_c3_500_50   | SphericalTree          | CART                       |             -1.276 |            -1.317 |              500 |           50 |     10.000 |              nan | low_target_values    | mostly_continuous;balanced |
| trees        | regression | 620_fri_c1_1000_25  | SphericalTree          | CART                       |             -1.273 |            -1.363 |             1000 |           25 |     40.000 |              nan | medium_target_values | mostly_continuous;balanced |
| trees        | regression | 607_fri_c4_1000_50  | SphericalTree          | CART                       |             -1.271 |            -1.543 |             1000 |           50 |     20.000 |              nan | low_target_values    | mostly_continuous;balanced |
| trees        | regression | 616_fri_c4_500_50   | SphericalTree          | CART                       |             -1.248 |            -1.560 |              500 |           50 |     10.000 |              nan | low_target_values    | mostly_continuous;balanced |
| trees        | regression | 592_fri_c4_1000_25  | SphericalTree          | CART                       |             -1.239 |            -1.513 |             1000 |           25 |     40.000 |              nan | low_target_values    | mostly_continuous;balanced |
| trees        | regression | 589_fri_c2_1000_25  | SphericalTree          | CART                       |             -1.211 |            -1.350 |             1000 |           25 |     40.000 |              nan | medium_target_values | mostly_continuous;balanced |
| trees        | regression | 633_fri_c0_500_25   | SphericalTree          | CART                       |             -1.210 |            -1.352 |              500 |           25 |     20.000 |              nan | low_target_values    | mostly_continuous;balanced |
| trees        | regression | 586_fri_c3_1000_25  | SphericalTree          | CART                       |             -1.170 |            -1.202 |             1000 |           25 |     40.000 |              nan | medium_target_values | mostly_continuous;balanced |
| trees        | regression | 644_fri_c4_250_25   | SphericalTree          | CART                       |             -1.077 |            -1.346 |              250 |           25 |     10.000 |              nan | low_target_values    | mostly_continuous;balanced |
| trees        | regression | 647_fri_c1_250_10   | SphericalTree          | CART                       |             -1.060 |            -1.270 |              250 |           10 |     25.000 |              nan | medium_target_values | mostly_continuous          |
| trees        | regression | 650_fri_c0_500_50   | SphericalTree          | CART                       |             -1.020 |            -1.307 |              500 |           50 |     10.000 |              nan | medium_target_values | mostly_continuous;balanced |
| trees        | regression | 584_fri_c4_500_25   | SphericalTree          | CART                       |             -1.011 |            -1.140 |              500 |           25 |     20.000 |              nan | medium_target_values | mostly_continuous;balanced |
| trees        | regression | 593_fri_c1_1000_10  | SphericalTree          | CART                       |             -0.964 |            -1.025 |             1000 |           10 |    100.000 |              nan | medium_target_values | mostly_continuous          |
| trees        | regression | 598_fri_c0_1000_25  | SphericalTree          | CART                       |             -0.949 |            -1.011 |             1000 |           25 |     40.000 |              nan | medium_target_values | mostly_continuous;balanced |
| trees        | regression | 603_fri_c0_250_50   | SphericalTree          | CART                       |             -0.944 |            -1.354 |              250 |           50 |      5.000 |              nan | medium_target_values | mostly_continuous;balanced |
| trees        | regression | 590_fri_c0_1000_50  | SphericalTree          | CART                       |             -0.943 |            -1.021 |             1000 |           50 |     20.000 |              nan | medium_target_values | mostly_continuous;balanced |

## Heuristic Rules: Spherical Forests, Classification

| rule                                                          |   support |   baseline_win_rate |   spherical_win_rate |   lift_vs_baseline |   mean_spherical_margin |   consistent_practical_win_rate |
|:--------------------------------------------------------------|----------:|--------------------:|---------------------:|-------------------:|------------------------:|--------------------------------:|
| 10 < p <= 50 & sparse predictors & balanced target            |         9 |               0.338 |                0.889 |              2.626 |                   0.011 |                           0.111 |
| 10 < p <= 50 & sparse predictors & target entropy >= 0.9      |         9 |               0.338 |                0.889 |              2.626 |                   0.011 |                           0.111 |
| 10 < p <= 50 & mostly discrete predictors & sparse predictors |        13 |               0.338 |                0.846 |              2.500 |                   0.012 |                           0.077 |
| n/p >= 50 & mostly discrete predictors & sparse predictors    |        12 |               0.338 |                0.833 |              2.462 |                   0.039 |                           0.250 |
| 10 < p <= 50 & sparse predictors & binary classification      |        11 |               0.338 |                0.818 |              2.417 |                   0.016 |                           0.091 |
| p <= 10 & n/p >= 50 & mostly discrete predictors              |        15 |               0.338 |                0.800 |              2.364 |                   0.045 |                           0.467 |
| n/p >= 50 & sparse predictors & balanced target               |        10 |               0.338 |                0.800 |              2.364 |                   0.041 |                           0.200 |
| n/p >= 50 & sparse predictors & min class share >= 0.35       |        10 |               0.338 |                0.800 |              2.364 |                   0.041 |                           0.200 |

## Heuristic Rules: Spherical Trees, Classification

| rule                                                         |   support |   baseline_win_rate |   spherical_win_rate |   lift_vs_baseline |   mean_spherical_margin |   consistent_practical_win_rate |
|:-------------------------------------------------------------|----------:|--------------------:|---------------------:|-------------------:|------------------------:|--------------------------------:|
| n/p >= 100 & balanced predictors & target entropy >= 0.9     |         9 |               0.238 |                0.778 |              3.262 |                   0.004 |                           0.333 |
| n/p >= 50 & balanced predictors & balanced target            |         8 |               0.238 |                0.750 |              3.145 |                   0.003 |                           0.250 |
| 10 < p <= 50 & n/p >= 100 & balanced target                  |        11 |               0.238 |                0.636 |              2.669 |                   0.013 |                           0.182 |
| n <= 1000 & p <= 10 & balanced predictors                    |         8 |               0.238 |                0.625 |              2.621 |                   0.014 |                           0.250 |
| 10 < p <= 50 & balanced predictors & balanced target         |         8 |               0.238 |                0.625 |              2.621 |                   0.003 |                           0.250 |
| n <= 1000 & mostly discrete predictors & balanced predictors |         8 |               0.238 |                0.625 |              2.621 |                   0.002 |                           0.250 |
| 10 < p <= 50 & n/p >= 50 & balanced predictors               |         8 |               0.238 |                0.625 |              2.621 |                   0.001 |                           0.250 |
| n > 1000 & n/p >= 100 & balanced target                      |        15 |               0.238 |                0.533 |              2.237 |                   0.007 |                           0.133 |

## Heuristic Rules: Spherical Forests, Regression

_No rows._

## Heuristic Rules: Spherical Trees, Regression

_No rows._

## Poor-Performance Rules: Spherical Forests, Classification

| rule                                                                      |   support |   baseline_win_rate |   spherical_win_rate |   lift_vs_baseline |   mean_spherical_margin |   consistent_practical_loss_rate |
|:--------------------------------------------------------------------------|----------:|--------------------:|---------------------:|-------------------:|------------------------:|---------------------------------:|
| n > 1000 & p > 50 & balanced target                                       |        12 |               0.338 |                0.000 |              0.000 |                  -0.102 |                            0.500 |
| n > 1000 & p > 50 & target entropy >= 0.9                                 |        12 |               0.338 |                0.000 |              0.000 |                  -0.102 |                            0.500 |
| p > 50 & binary classification                                            |         9 |               0.338 |                0.000 |              0.000 |                  -0.098 |                            0.444 |
| n > 1000 & p > 50                                                         |        14 |               0.338 |                0.000 |              0.000 |                  -0.092 |                            0.500 |
| high predictor correlation & balanced target                              |        12 |               0.338 |                0.000 |              0.000 |                  -0.087 |                            0.417 |
| high predictor correlation & target entropy >= 0.9                        |        12 |               0.338 |                0.000 |              0.000 |                  -0.087 |                            0.417 |
| redundant predictors & high predictor correlation & balanced target       |        12 |               0.338 |                0.000 |              0.000 |                  -0.087 |                            0.417 |
| redundant predictors & high predictor correlation & target entropy >= 0.9 |        12 |               0.338 |                0.000 |              0.000 |                  -0.087 |                            0.417 |

## Poor-Performance Rules: Spherical Trees, Classification

| rule                                                     |   support |   baseline_win_rate |   spherical_win_rate |   lift_vs_baseline |   mean_spherical_margin |   consistent_practical_loss_rate |
|:---------------------------------------------------------|----------:|--------------------:|---------------------:|-------------------:|------------------------:|---------------------------------:|
| 10 < p <= 50 & skewed predictors & unbalanced target     |         8 |               0.238 |                0.000 |              0.000 |                  -0.217 |                            1.000 |
| 10 < p <= 50 & skewed predictors & target entropy < 0.75 |         8 |               0.238 |                0.000 |              0.000 |                  -0.217 |                            1.000 |
| 10 < p <= 50 & n/p >= 100 & sparse predictors            |         8 |               0.238 |                0.000 |              0.000 |                  -0.213 |                            1.000 |
| n/p >= 100 & sparse predictors & skewed predictors       |         8 |               0.238 |                0.000 |              0.000 |                  -0.213 |                            1.000 |
| 10 < p <= 50 & sparse predictors & unbalanced target     |         8 |               0.238 |                0.000 |              0.000 |                  -0.209 |                            1.000 |
| n > 1000 & 10 < p <= 50 & skewed predictors              |         9 |               0.238 |                0.000 |              0.000 |                  -0.205 |                            1.000 |
| 10 < p <= 50 & n/p >= 50 & skewed predictors             |         9 |               0.238 |                0.000 |              0.000 |                  -0.205 |                            1.000 |
| 10 < p <= 50 & n/p >= 100 & skewed predictors            |         9 |               0.238 |                0.000 |              0.000 |                  -0.205 |                            1.000 |

## Poor-Performance Rules: Spherical Forests, Regression

| rule                                                      |   support |   baseline_win_rate |   spherical_win_rate |   lift_vs_baseline |   mean_spherical_margin |   consistent_practical_loss_rate |
|:----------------------------------------------------------|----------:|--------------------:|---------------------:|-------------------:|------------------------:|---------------------------------:|
| 10 < p <= 50 & balanced predictors & low target values    |        10 |               0.126 |                0.000 |              0.000 |                  -0.483 |                            1.000 |
| 10 < p <= 50 & balanced predictors                        |        27 |               0.126 |                0.000 |              0.000 |                  -0.476 |                            1.000 |
| 10 < p <= 50 & balanced predictors & weakly skewed target |        27 |               0.126 |                0.000 |              0.000 |                  -0.476 |                            1.000 |
| 10 < p <= 50 & balanced predictors & target CV >= 1       |        27 |               0.126 |                0.000 |              0.000 |                  -0.476 |                            1.000 |
| 10 < p <= 50 & p/n >= 0.10 & balanced predictors          |        12 |               0.126 |                0.000 |              0.000 |                  -0.467 |                            1.000 |
| 10 < p <= 50 & weakly skewed target & target CV >= 1      |        28 |               0.126 |                0.000 |              0.000 |                  -0.460 |                            0.964 |
| p/n >= 0.10 & balanced predictors                         |        14 |               0.126 |                0.000 |              0.000 |                  -0.453 |                            1.000 |
| p/n >= 0.10 & balanced predictors & weakly skewed target  |        14 |               0.126 |                0.000 |              0.000 |                  -0.453 |                            1.000 |

## Poor-Performance Rules: Spherical Trees, Regression

| rule                                                      |   support |   baseline_win_rate |   spherical_win_rate |   lift_vs_baseline |   mean_spherical_margin |   consistent_practical_loss_rate |
|:----------------------------------------------------------|----------:|--------------------:|---------------------:|-------------------:|------------------------:|---------------------------------:|
| 10 < p <= 50 & balanced predictors & low target values    |        10 |               0.059 |                0.000 |              0.000 |                  -1.185 |                            1.000 |
| 10 < p <= 50 & balanced predictors                        |        27 |               0.059 |                0.000 |              0.000 |                  -1.141 |                            1.000 |
| 10 < p <= 50 & balanced predictors & weakly skewed target |        27 |               0.059 |                0.000 |              0.000 |                  -1.141 |                            1.000 |
| 10 < p <= 50 & balanced predictors & target CV >= 1       |        27 |               0.059 |                0.000 |              0.000 |                  -1.141 |                            1.000 |
| 10 < p <= 50 & weakly skewed target & target CV >= 1      |        28 |               0.059 |                0.000 |              0.000 |                  -1.108 |                            1.000 |
| 10 < p <= 50 & low target values                          |        11 |               0.059 |                0.000 |              0.000 |                  -1.096 |                            1.000 |
| 10 < p <= 50 & low target values & weakly skewed target   |        11 |               0.059 |                0.000 |              0.000 |                  -1.096 |                            1.000 |
| 10 < p <= 50 & low target values & target CV >= 1         |        11 |               0.059 |                0.000 |              0.000 |                  -1.096 |                            1.000 |

## Practical Reading

- The strongest repeatable signal is classification with discrete/binary
  predictors and moderate dimensionality. This fits the geometric motivation:
  spherical splits can isolate compact interaction regions that axis-aligned
  splits need several levels to approximate.
- Spherical forests are the more promising default than single spherical trees.
  They keep many of the classification gains while reducing the brittleness of a
  single center/radius choice.
- Regression target magnitude is not a useful heuristic by itself. Any apparent
  high/low target-value rule should be treated as a dataset-family artifact,
  because target scaling is arbitrary under R2.
- Rows and predictors alone do not produce a clean monotone rule. The useful
  heuristic is conditional: low-to-moderate p, enough observations to estimate
  centers, discrete/binary or radially clustered predictors, and a classification
  target that is not extremely imbalanced.

## Candidate Heuristic

Use the spherical forest as a serious candidate when all or most of the following
hold:

1. The task is classification.
2. `p <= 50`, preferably `p <= 10` for the strongest practical gains.
3. `n/p >= 50`, so centers and radii are estimated from a reasonably dense cloud.
4. Predictors are mostly discrete, mostly binary, or sparse, or prior knowledge
   suggests compact/radial class regions.
5. The target is balanced or only moderately imbalanced. Unbalanced multiclass
   can still work in forests, but unbalanced binary tasks did not show a strong
   signal here.

For single spherical trees, be more conservative: the favorable cases are
classification datasets with high `n/p`, low-to-moderate `p`, and either balanced
predictors/targets or known radial geometry. The tree wins are less stable than
the forest wins.

Avoid treating regression target magnitude as a model-selection criterion. In
this benchmark, high and low target-value groups both had weak spherical win
rates and no consistent practical regression wins.

## Candidate Anti-Heuristic

Expect spherical methods to perform poorly when one or more of the following
hold:

1. The task is regression, especially with `10 < p <= 50`; the observed margins
   are strongly negative and target magnitude does not rescue the method.
2. The task is classification with many predictors (`p > 50`, and especially
   `p > 200`) relative to the number of observations. Spherical splits then
   search centers/radii in a space where distances are noisy and local spheres
   become hard to estimate.
3. Predictors are mostly continuous with scale imbalance or redundancy but no
   clear compact/radial class structure. In those cases, oblique or axis-aligned
   splits often approximate the boundary more directly.
4. Binary classification is strongly target-imbalanced. This benchmark had a
   weak spherical signal for unbalanced binary tasks.
5. Single spherical trees should be avoided more often than spherical forests:
   the failure rates are higher because one bad center/radius choice propagates
   down the whole tree.

## Context Summary

The full context summary is in `spherical_pmlb_heuristic_context_summary.csv`.
The mined rules are in `spherical_pmlb_heuristic_rules.csv`.
