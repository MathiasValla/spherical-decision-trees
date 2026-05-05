# Spherical PMLB Regime Analysis

This analysis uses `spherical_pmlb_full.csv` and `spherical_pmlb_full_summary.csv`.
Classification winners are selected by mean balanced accuracy; regression winners
are selected by mean R2. The regime maps use `n_used_samples`, not always the
original PMLB row count, because several large datasets were capped during the
benchmark.

## Outputs

- `figures/spherical_regime_overall.png`
- `figures/spherical_regime_trees.png`
- `figures/spherical_regime_forests.png`
- `figures/spherical_margin_trees.png`
- `figures/spherical_margin_forests.png`

Additional tables:

- `spherical_pmlb_regime_table.csv`: dataset-level winners, margins, metadata, and model scores.
- `spherical_pmlb_regime_winner_counts.csv`: winner counts by comparison and task.
- `spherical_pmlb_regime_spherical_summary.csv`: spherical win rates and margin summaries.
- `spherical_pmlb_regime_grid_counts.csv`: winner counts in coarse row/predictor bands.
- `spherical_pmlb_spherical_wins.csv`: all positive spherical-margin datasets.

## Empirical Takeaways

- The current spherical signal is mainly a classification signal: spherical
  models are positive-margin overall winners on 30.0% of
  classification datasets versus 12.6% of regression datasets.
- Ensembling helps substantially. Among forests on classification tasks,
  spherical random forests win 45 datasets, compared
  with 51 for classical random forests and
  34 for oblique random forests. Positive-margin
  spherical forest wins have median size 900 rows and
  10 predictors.
- Single spherical trees win 31 classification
  tree-only comparisons (23.8%). These wins are mostly in
  small-to-medium, low-dimensional problems: the median positive-margin spherical
  tree win has 625 rows and 10
  predictors.
- There is no clean monotone frontier in rows and predictors alone. The regime
  maps show patchy wins, which suggests that the geometry of the response
  boundary is the missing explanatory variable. The row/predictor map is useful
  as a first diagnostic, but the next analysis should add shape diagnostics such
  as class overlap, radial separability, interaction strength, and manifold or
  cluster structure.

## Winner Counts

| comparison   | task           | winner           |   n_wins |
|:-------------|:---------------|:-----------------|---------:|
| forests      | classification | Random forest    |       51 |
| forests      | classification | Spherical forest |       45 |
| forests      | classification | Oblique forest   |       34 |
| forests      | regression     | Random forest    |       63 |
| forests      | regression     | Oblique forest   |       41 |
| forests      | regression     | Spherical forest |       15 |
| overall      | classification | Random forest    |       34 |
| overall      | classification | Spherical forest |       32 |
| overall      | classification | Oblique forest   |       31 |
| overall      | classification | CART             |       21 |
| overall      | classification | Spherical tree   |        7 |
| overall      | classification | Oblique tree     |        5 |
| overall      | regression     | Random forest    |       63 |
| overall      | regression     | Oblique forest   |       40 |
| overall      | regression     | Spherical forest |       15 |
| overall      | regression     | CART             |        1 |
| trees        | classification | CART             |       63 |
| trees        | classification | Oblique tree     |       36 |
| trees        | classification | Spherical tree   |       31 |
| trees        | regression     | CART             |       95 |
| trees        | regression     | Oblique tree     |       17 |
| trees        | regression     | Spherical tree   |        7 |

## Spherical Advantage Summary

Here, a spherical win means the best spherical model in the comparison set beats
the best non-spherical model in the same set.

| comparison   | task           |   n_datasets |   n_spherical_wins |   spherical_win_rate |   median_spherical_margin |   mean_spherical_margin |   median_rows_all |   median_features_all |   median_rows_when_spherical_wins |   median_features_when_spherical_wins |   median_margin_when_spherical_wins |
|:-------------|:---------------|-------------:|-------------------:|---------------------:|--------------------------:|------------------------:|------------------:|----------------------:|----------------------------------:|--------------------------------------:|------------------------------------:|
| forests      | classification |          130 |                 44 |                 33.8 |                   -0.0053 |                 -0.0123 |             958.5 |                    14 |                             899.5 |                                  10.5 |                              0.0206 |
| forests      | regression     |          119 |                 15 |                 12.6 |                   -0.0623 |                 -0.1585 |             500   |                    10 |                            1000   |                                   5   |                              0.0057 |
| overall      | classification |          130 |                 39 |                 30   |                   -0.0074 |                 -0.0216 |             958.5 |                    14 |                             556   |                                  10   |                              0.0168 |
| overall      | regression     |          119 |                 15 |                 12.6 |                   -0.0623 |                 -0.1585 |             500   |                    10 |                            1000   |                                   5   |                              0.0057 |
| trees        | classification |          130 |                 31 |                 23.8 |                   -0.042  |                 -0.0662 |             958.5 |                    14 |                             625   |                                  10   |                              0.0281 |
| trees        | regression     |          119 |                  7 |                  5.9 |                   -0.4712 |                 -0.5573 |             500   |                    10 |                             100   |                                  10   |                              0.1279 |

## Largest Positive Spherical Margins

| comparison   | task           | dataset                      | best_spherical_model   | best_non_spherical_model   |   spherical_margin |   n_used_samples |   n_features |
|:-------------|:---------------|:-----------------------------|:-----------------------|:---------------------------|-------------------:|-----------------:|-------------:|
| trees        | regression     | 542_pollution                | Spherical tree         | CART                       |             0.5018 |               60 |           15 |
| trees        | classification | parity5                      | Spherical tree         | Oblique tree               |             0.4167 |               32 |            5 |
| forests      | classification | parity5+5                    | Spherical forest       | Random forest              |             0.3173 |             1124 |           10 |
| trees        | regression     | 656_fri_c1_100_5             | Spherical tree         | CART                       |             0.3024 |              100 |            5 |
| trees        | regression     | 527_analcatdata_election2000 | Spherical tree         | CART                       |             0.1781 |               67 |           14 |
| trees        | classification | balance_scale                | Spherical tree         | Oblique tree               |             0.1364 |              625 |            4 |
| trees        | regression     | 1595_poker                   | Spherical tree         | Oblique tree               |             0.1279 |             1000 |           10 |
| forests      | classification | analcatdata_boxing1          | Spherical forest       | Random forest              |             0.1035 |              120 |            3 |
| trees        | classification | ring                         | Spherical tree         | CART                       |             0.1025 |             5000 |           20 |
| forests      | classification | nursery                      | Spherical forest       | Random forest              |             0.0868 |             5000 |            8 |
| trees        | classification | analcatdata_cyyoung9302      | Spherical tree         | CART                       |             0.0764 |               92 |           10 |
| forests      | regression     | 594_fri_c2_100_5             | Spherical forest       | Oblique forest             |             0.068  |              100 |            5 |
| forests      | classification | dis                          | Spherical forest       | Random forest              |             0.0674 |             3772 |           29 |
| trees        | classification | hayes_roth                   | Spherical tree         | CART                       |             0.0654 |              160 |            4 |
| trees        | classification | twonorm                      | Spherical tree         | Oblique tree               |             0.064  |             5000 |           20 |

## Interpretation Notes

- `overall` asks whether any spherical model beats CART, random forest, oblique
  tree, and oblique forest on the same dataset.
- `trees` isolates single-tree behavior: spherical tree versus CART and oblique
  tree.
- `forests` isolates ensemble behavior: spherical random forest versus random
  forest and oblique random forest.
- The pale background bins in the regime maps show the modal winner in each
  log-spaced row/predictor cell; the points are the actual dataset winners.
