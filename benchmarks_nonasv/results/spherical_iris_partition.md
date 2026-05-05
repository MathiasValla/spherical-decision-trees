# Iris Spherical Partition

This example fits a depth-2 spherical decision tree on Iris petal
length and petal width after standardization. The displayed
circles are the actual split rules in the standardized two-dimensional
covariate space.

At each internal node, the left child receives observations satisfying
`(x_1 - c_1)^2 + (x_2 - c_2)^2 <= r^2`; the right child receives the
outside of that sphere.

## Split Table

|   node |   depth |   left_child_inside |   right_child_outside |   n_node_samples |   center_petal_length_standardized |   center_petal_width_standardized |   radius_standardized |
|-------:|--------:|--------------------:|----------------------:|-----------------:|-----------------------------------:|----------------------------------:|----------------------:|
|      0 |       0 |                   1 |                     2 |              150 |                             -1.305 |                            -1.255 |                 1.061 |
|      2 |       1 |                   3 |                     4 |              100 |                             -0.147 |                            -0.262 |                 1.06  |

## Outputs

- `figures/spherical_iris_partition.png`
- `spherical_iris_splits.csv`
