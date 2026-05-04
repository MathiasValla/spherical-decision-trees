# distutils: language=c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

cimport numpy as cnp

from .._lib.sklearn.tree._tree cimport Node
from .._lib.sklearn.utils._typedefs cimport float32_t, float64_t, intp_t
from ._oblique_tree cimport ObliqueTree


cdef class SphericalTree(ObliqueTree):
    """Tree backend whose node vectors are hypersphere centers."""

    cdef float32_t _compute_feature(
        self,
        const float32_t[:, :] X_ndarray,
        intp_t sample_index,
        Node *node
    ) noexcept nogil:
        cdef float32_t distance = 0.0
        cdef float32_t diff
        cdef intp_t node_id = node - self.nodes
        cdef intp_t j
        cdef intp_t feature_index

        for j in range(0, self.proj_vec_indices[node_id].size()):
            feature_index = self.proj_vec_indices[node_id][j]
            diff = X_ndarray[sample_index, feature_index] - self.proj_vec_weights[node_id][j]
            distance += diff * diff

        return distance

    cdef void _compute_feature_importances(
        self,
        float64_t[:] importances,
        Node* node
    ) noexcept nogil:
        cdef Node* nodes = self.nodes
        cdef Node* left
        cdef Node* right
        cdef intp_t node_id = node - self.nodes
        cdef intp_t i, feature_index
        cdef float64_t impurity_decrease
        cdef float64_t share

        left = &nodes[node.left_child]
        right = &nodes[node.right_child]
        impurity_decrease = (
            node.weighted_n_node_samples * node.impurity -
            left.weighted_n_node_samples * left.impurity -
            right.weighted_n_node_samples * right.impurity
        )

        if self.proj_vec_indices[node_id].size() == 0:
            return

        share = impurity_decrease / self.proj_vec_indices[node_id].size()
        for i in range(0, self.proj_vec_indices[node_id].size()):
            feature_index = self.proj_vec_indices[node_id][i]
            importances[feature_index] += share
