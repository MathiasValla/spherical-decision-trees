# distutils: language = c++

from .._lib.sklearn.tree._tree cimport Node
from .._lib.sklearn.utils._typedefs cimport float32_t, float64_t, intp_t
from ._oblique_tree cimport ObliqueTree


cdef class SphericalTree(ObliqueTree):
    cdef float32_t _compute_feature(
        self,
        const float32_t[:, :] X_ndarray,
        intp_t sample_index,
        Node *node
    ) noexcept nogil
    cdef void _compute_feature_importances(
        self,
        float64_t[:] importances,
        Node* node
    ) noexcept nogil
