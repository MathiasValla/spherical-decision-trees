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

## Strong Practical Wins

`consistent_practical_win` requires the spherical model to beat the selected
non-spherical comparator on every fold, to exceed a practical margin threshold
(0.03 balanced-accuracy/R2 points for
classification, 0.05 for regression), and to be
larger than two fold-level standard errors. With only three folds this is not a
formal significance test; it is a conservative practical screen.

| comparison   | task           |   n_datasets |   consistent_practical_wins |   large_margin_wins |
|:-------------|:---------------|-------------:|----------------------------:|--------------------:|
| forests      | classification |          130 |                          10 |                   7 |
| forests      | regression     |          119 |                           0 |                   0 |
| overall      | classification |          130 |                           8 |                   5 |
| overall      | regression     |          119 |                           0 |                   0 |
| trees        | classification |          130 |                           7 |                   5 |
| trees        | regression     |          119 |                           0 |                   2 |

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

## Context Summary

The full context summary is in `spherical_pmlb_heuristic_context_summary.csv`.
The mined rules are in `spherical_pmlb_heuristic_rules.csv`.
