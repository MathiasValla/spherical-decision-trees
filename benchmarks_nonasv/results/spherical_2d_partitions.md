# Spherical 2D Partition Figures

These figures are generated for benchmark datasets with exactly two
continuous-looking numerical predictors. Each model is fitted with the
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

| task           | dataset             |   n_used_samples |   n_features | feature_1           | feature_2   |   node_count |   internal_node_count |   drawn_circle_count | figure                                                             | splits                                                                  |
|:---------------|:--------------------|-----------------:|-------------:|:--------------------|:------------|-------------:|----------------------:|---------------------:|:-------------------------------------------------------------------|:------------------------------------------------------------------------|
| classification | prnn_synth          |              250 |            2 | xs                  | ys          |           35 |                    17 |                   17 | figures/spherical_2d_partitions/classification_prnn_synth.png      | spherical_2d_partition_splits/classification_prnn_synth_splits.csv      |
| regression     | 192_vineyard        |               52 |            2 | lugs_1989           | lugs_1990   |           63 |                    31 |                   31 | figures/spherical_2d_partitions/regression_192_vineyard.png        | spherical_2d_partition_splits/regression_192_vineyard_splits.csv        |
| regression     | 228_elusage         |               55 |            2 | average_temperature | month       |           93 |                    46 |                   46 | figures/spherical_2d_partitions/regression_228_elusage.png         | spherical_2d_partition_splits/regression_228_elusage_splits.csv         |
| regression     | 712_chscase_geyser1 |              222 |            2 | col_1               | col_2       |          285 |                   142 |                  142 | figures/spherical_2d_partitions/regression_712_chscase_geyser1.png | spherical_2d_partition_splits/regression_712_chscase_geyser1_splits.csv |
| regression     | banana              |             1000 |            2 | At1                 | At2         |          121 |                    60 |                   60 | figures/spherical_2d_partitions/regression_banana.png              | spherical_2d_partition_splits/regression_banana_splits.csv              |
