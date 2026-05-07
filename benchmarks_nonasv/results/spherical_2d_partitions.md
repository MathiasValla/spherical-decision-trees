# Spherical 2D Partition Figures

These figures are generated for complete benchmark datasets with exactly
two numerical predictors, including the synthetic toy classification
datasets. Each model is fitted with the
same spherical tree hyperparameters used in the full PMLB benchmark:
`n_center_candidates=500`,
`radius_candidates=None`,
`center_strategy=target_radial`.

The displayed coordinates are standardized because spherical splits are
distance-based and the benchmark pipeline standardizes predictors before
fitting.

Circles are the explicit split frontiers. The right-hand panel adds a
light fill for the terminal leaf regions induced by those frontiers;
the left-hand panel only shades the two root children for orientation.
The axes are centered on the observed data cloud, so far-field circles
may appear as low-curvature arcs clipped by the plotting window.

| task           | dataset                 |   n_used_samples |   n_features | feature_1           | feature_2           |   node_count |   internal_node_count |   drawn_circle_count | figure                                                                    | splits                                                                         |
|:---------------|:------------------------|-----------------:|-------------:|:--------------------|:--------------------|-------------:|----------------------:|---------------------:|:--------------------------------------------------------------------------|:-------------------------------------------------------------------------------|
| classification | prnn_synth              |              250 |            2 | xs                  | ys                  |           35 |                    17 |                   17 | figures/spherical_2d_partitions/classification_prnn_synth.png             | spherical_2d_partition_splits/classification_prnn_synth_splits.csv             |
| classification | toy_gaussian_quantiles  |              100 |            2 | Feature #0          | Feature #1          |            5 |                     2 |                    2 | figures/spherical_2d_partitions/classification_toy_gaussian_quantiles.png | spherical_2d_partition_splits/classification_toy_gaussian_quantiles_splits.csv |
| classification | toy_moons               |              100 |            2 | Feature #0          | Feature #1          |            9 |                     4 |                    4 | figures/spherical_2d_partitions/classification_toy_moons.png              | spherical_2d_partition_splits/classification_toy_moons_splits.csv              |
| classification | toy_xor                 |              200 |            2 | Feature #0          | Feature #1          |           13 |                     6 |                    6 | figures/spherical_2d_partitions/classification_toy_xor.png                | spherical_2d_partition_splits/classification_toy_xor_splits.csv                |
| regression     | 192_vineyard            |               52 |            2 | lugs_1989           | lugs_1990           |           63 |                    31 |                   31 | figures/spherical_2d_partitions/regression_192_vineyard.png               | spherical_2d_partition_splits/regression_192_vineyard_splits.csv               |
| regression     | 228_elusage             |               55 |            2 | average_temperature | month               |           93 |                    46 |                   46 | figures/spherical_2d_partitions/regression_228_elusage.png                | spherical_2d_partition_splits/regression_228_elusage_splits.csv                |
| regression     | 519_vinnie              |              380 |            2 | year                | field_goal_attempts |          147 |                    73 |                   73 | figures/spherical_2d_partitions/regression_519_vinnie.png                 | spherical_2d_partition_splits/regression_519_vinnie_splits.csv                 |
| regression     | 523_analcatdata_neavote |              100 |            2 | Party               | Bills               |           11 |                     5 |                    5 | figures/spherical_2d_partitions/regression_523_analcatdata_neavote.png    | spherical_2d_partition_splits/regression_523_analcatdata_neavote_splits.csv    |
| regression     | 663_rabe_266            |              120 |            2 | col_1               | col_2               |          207 |                   103 |                  103 | figures/spherical_2d_partitions/regression_663_rabe_266.png               | spherical_2d_partition_splits/regression_663_rabe_266_splits.csv               |
| regression     | 712_chscase_geyser1     |              222 |            2 | col_1               | col_2               |          285 |                   142 |                  142 | figures/spherical_2d_partitions/regression_712_chscase_geyser1.png        | spherical_2d_partition_splits/regression_712_chscase_geyser1_splits.csv        |
| regression     | banana                  |             1000 |            2 | At1                 | At2                 |          121 |                    60 |                   60 | figures/spherical_2d_partitions/regression_banana.png                     | spherical_2d_partition_splits/regression_banana_splits.csv                     |
