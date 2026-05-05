# distutils: language = c++

from libcpp.vector cimport vector

from .._lib.sklearn.tree._criterion cimport Criterion
from .._lib.sklearn.tree._splitter cimport SplitRecord, Splitter
from .._lib.sklearn.tree._tree cimport ParentInfo
from .._lib.sklearn.utils._typedefs cimport (
    float32_t,
    float64_t,
    int8_t,
    intp_t,
    uint8_t,
)


cdef class SphericalSplitter(Splitter):
    cdef const float32_t[:, :] X
    cdef vector[vector[float32_t]] center_values
    cdef vector[vector[intp_t]] center_indices
    cdef public intp_t n_center_candidates
    cdef public intp_t sphere_features
    cdef public intp_t radius_candidates
    cdef public intp_t center_strategy
    cdef public bint is_classification
    cdef public intp_t n_classes

    cdef int node_reset(
        self,
        intp_t start,
        intp_t end,
        float64_t* weighted_n_node_samples
    ) except -1 nogil

    cdef intp_t pointer_size(self) noexcept nogil

    cdef bint _center_has_feature(
        self,
        intp_t center_i,
        intp_t feature
    ) noexcept nogil

    cdef intp_t _center_kind(
        self,
        intp_t center_i
    ) noexcept nogil

    cdef bint _target_center_value(
        self,
        intp_t center_i,
        intp_t start,
        intp_t end,
        const intp_t[:] samples,
        intp_t feature,
        float64_t y_mean,
        float32_t* value_out
    ) noexcept nogil

    cdef void sample_center(
        self,
        intp_t center_i,
        intp_t start,
        intp_t end,
        const intp_t[:] samples,
    ) noexcept nogil

    cdef void compute_squared_distances(
        self,
        intp_t start,
        intp_t end,
        const intp_t[:] samples,
        float32_t[:] feature_values,
        vector[float32_t]* center_values,
        vector[intp_t]* center_indices,
    ) noexcept nogil

    cdef int node_split(
        self,
        ParentInfo* parent,
        SplitRecord* split,
    ) except -1 nogil


cdef class BestSphericalSplitter(SphericalSplitter):
    cdef int node_split(
        self,
        ParentInfo* parent,
        SplitRecord* split,
    ) except -1 nogil
