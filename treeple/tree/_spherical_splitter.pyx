# distutils: language=c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

import numpy as np

from cython.operator cimport dereference as deref
from libcpp.vector cimport vector

from .._lib.sklearn.tree._criterion cimport Criterion
from .._lib.sklearn.tree._utils cimport rand_int
from .._lib.sklearn.utils._typedefs cimport uint32_t
from ._oblique_splitter cimport ObliqueSplitRecord
from ._sklearn_splitter cimport sort


cdef float64_t INFINITY = np.inf
cdef float32_t FEATURE_THRESHOLD = 1e-7


cdef inline void _init_split(ObliqueSplitRecord* self, intp_t start_pos) noexcept nogil:
    self.impurity_left = INFINITY
    self.impurity_right = INFINITY
    self.pos = start_pos
    self.feature = 0
    self.threshold = 0.
    self.improvement = -INFINITY
    self.missing_go_to_left = False
    self.n_missing = 0


cdef class SphericalSplitter(Splitter):
    """Base splitter for inside/outside hypersphere rules."""

    def __cinit__(
        self,
        Criterion criterion,
        intp_t max_features,
        intp_t min_samples_leaf,
        float64_t min_weight_leaf,
        object random_state,
        const int8_t[:] monotonic_cst,
        intp_t n_center_candidates,
        intp_t radius_candidates,
        *argv
    ):
        self.criterion = criterion
        self.n_samples = 0
        self.n_features = 0

        self.max_features = max_features
        self.sphere_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.random_state = random_state
        self.monotonic_cst = monotonic_cst

        self.n_center_candidates = max(n_center_candidates, 1)
        self.radius_candidates = radius_candidates
        self.center_values = vector[vector[float32_t]](self.n_center_candidates)
        self.center_indices = vector[vector[intp_t]](self.n_center_candidates)

    def __reduce__(self):
        """Enable pickling the splitter."""
        return (type(self),
                (
                    self.criterion,
                    self.max_features,
                    self.min_samples_leaf,
                    self.min_weight_leaf,
                    self.random_state,
                    self.monotonic_cst.base if self.monotonic_cst is not None else None,
                    self.n_center_candidates,
                    self.radius_candidates,
                ), self.__getstate__())

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const uint8_t[::1] missing_values_in_feature_mask,
    ) except -1:
        Splitter.init(self, X, y, sample_weight, missing_values_in_feature_mask)
        self.X = X
        self.sphere_features = min(self.max_features, self.n_features)
        return 0

    cdef int node_reset(
        self,
        intp_t start,
        intp_t end,
        float64_t* weighted_n_node_samples
    ) except -1 nogil:
        Splitter.node_reset(self, start, end, weighted_n_node_samples)

        cdef intp_t i
        for i in range(self.n_center_candidates):
            self.center_values[i].clear()
            self.center_indices[i].clear()
        return 0

    cdef intp_t pointer_size(self) noexcept nogil:
        return sizeof(ObliqueSplitRecord)

    cdef bint _center_has_feature(
        self,
        intp_t center_i,
        intp_t feature
    ) noexcept nogil:
        cdef intp_t j
        for j in range(self.center_indices[center_i].size()):
            if self.center_indices[center_i][j] == feature:
                return True
        return False

    cdef void sample_center(
        self,
        intp_t center_i,
        intp_t start,
        intp_t end,
        const intp_t[:] samples,
    ) noexcept nogil:
        cdef intp_t n_node_samples = end - start
        cdef intp_t n_features = self.n_features
        cdef intp_t n_center_features = min(self.sphere_features, n_features)
        cdef uint32_t* random_state = &self.rand_r_state

        cdef intp_t j, p, feat, sample_a, sample_b
        cdef float64_t weighted_sum
        cdef float64_t weighted_total
        cdef float32_t value

        self.center_values[center_i].clear()
        self.center_indices[center_i].clear()

        if n_center_features == n_features:
            for j in range(n_features):
                self.center_indices[center_i].push_back(j)
        else:
            while self.center_indices[center_i].size() < n_center_features:
                feat = rand_int(0, n_features, random_state)
                if not self._center_has_feature(center_i, feat):
                    self.center_indices[center_i].push_back(feat)

        sample_a = samples[start + rand_int(0, n_node_samples, random_state)]
        sample_b = samples[start + rand_int(0, n_node_samples, random_state)]

        for j in range(self.center_indices[center_i].size()):
            feat = self.center_indices[center_i][j]

            if center_i == 0:
                weighted_sum = 0.0
                weighted_total = 0.0
                for p in range(start, end):
                    if self.sample_weight is not None:
                        weighted_sum += self.X[samples[p], feat] * self.sample_weight[samples[p]]
                        weighted_total += self.sample_weight[samples[p]]
                    else:
                        weighted_sum += self.X[samples[p], feat]
                if self.sample_weight is not None and weighted_total > 0.0:
                    value = <float32_t>(weighted_sum / weighted_total)
                else:
                    value = <float32_t>(weighted_sum / n_node_samples)
            elif center_i % 2 == 1:
                value = self.X[sample_a, feat]
            else:
                value = <float32_t>(
                    self.X[sample_a, feat] / 2.0 + self.X[sample_b, feat] / 2.0
                )

            self.center_values[center_i].push_back(value)

    cdef void compute_squared_distances(
        self,
        intp_t start,
        intp_t end,
        const intp_t[:] samples,
        float32_t[:] feature_values,
        vector[float32_t]* center_values,
        vector[intp_t]* center_indices,
    ) noexcept nogil:
        cdef intp_t p, j, feat
        cdef float32_t diff
        cdef float32_t dist

        for p in range(start, end):
            dist = 0.0
            for j in range(center_indices.size()):
                feat = deref(center_indices)[j]
                diff = self.X[samples[p], feat] - deref(center_values)[j]
                dist += diff * diff
            feature_values[p] = dist

    cdef int node_split(
        self,
        ParentInfo* parent_record,
        SplitRecord* split,
    ) except -1 nogil:
        return 0


cdef class BestSphericalSplitter(SphericalSplitter):
    cdef int node_split(
        self,
        ParentInfo* parent_record,
        SplitRecord* split,
    ) except -1 nogil:
        cdef ObliqueSplitRecord* spherical_split = <ObliqueSplitRecord*>(split)

        cdef intp_t[::1] samples = self.samples
        cdef intp_t start = self.start
        cdef intp_t end = self.end
        cdef float32_t[::1] feature_values = self.feature_values
        cdef intp_t min_samples_leaf = self.min_samples_leaf

        cdef ObliqueSplitRecord best_split, current_split
        cdef float64_t current_proxy_improvement = -INFINITY
        cdef float64_t best_proxy_improvement = -INFINITY
        cdef float64_t impurity = parent_record.impurity

        cdef intp_t center_i, p, partition_end, j
        cdef intp_t n_valid_thresholds, threshold_i, radius_step
        cdef float32_t temp_d, diff

        _init_split(&best_split, end)

        for center_i in range(self.n_center_candidates):
            self.sample_center(center_i, start, end, samples)
            if self.center_values[center_i].empty():
                continue

            current_split.feature = center_i
            current_split.proj_vec_weights = &self.center_values[center_i]
            current_split.proj_vec_indices = &self.center_indices[center_i]

            self.compute_squared_distances(
                start,
                end,
                samples,
                feature_values,
                &self.center_values[center_i],
                &self.center_indices[center_i],
            )
            sort(&feature_values[start], &samples[start], end - start)

            n_valid_thresholds = 0
            p = start
            while p < end:
                while (
                    p + 1 < end
                    and feature_values[p + 1] <= feature_values[p] + FEATURE_THRESHOLD
                ):
                    p += 1
                p += 1

                if p < end and (
                    (p - start) >= min_samples_leaf and (end - p) >= min_samples_leaf
                ):
                    n_valid_thresholds += 1

            if n_valid_thresholds == 0:
                continue

            radius_step = 1
            if self.radius_candidates > 0 and n_valid_thresholds > self.radius_candidates:
                radius_step = max(n_valid_thresholds // self.radius_candidates, 1)

            self.criterion.reset()
            threshold_i = 0
            p = start
            while p < end:
                while (
                    p + 1 < end
                    and feature_values[p + 1] <= feature_values[p] + FEATURE_THRESHOLD
                ):
                    p += 1
                p += 1

                if p < end:
                    current_split.pos = p

                    if (((current_split.pos - start) < min_samples_leaf) or
                            ((end - current_split.pos) < min_samples_leaf)):
                        continue

                    threshold_i += 1
                    if (
                        radius_step > 1
                        and threshold_i != 1
                        and threshold_i != n_valid_thresholds
                        and threshold_i % radius_step != 0
                    ):
                        continue

                    self.criterion.update(current_split.pos)
                    if self.check_postsplit_conditions() == 1:
                        continue

                    current_proxy_improvement = self.criterion.proxy_impurity_improvement()

                    if current_proxy_improvement > best_proxy_improvement:
                        best_proxy_improvement = current_proxy_improvement
                        current_split.threshold = (
                            feature_values[p - 1] / 2.0 + feature_values[p] / 2.0
                        )

                        if (
                            (current_split.threshold == feature_values[p]) or
                            (current_split.threshold == INFINITY) or
                            (current_split.threshold == -INFINITY)
                        ):
                            current_split.threshold = feature_values[p - 1]

                        best_split = current_split

        if best_split.pos < end:
            partition_end = end
            p = start

            while p < partition_end:
                temp_d = 0.0
                for j in range(best_split.proj_vec_indices.size()):
                    diff = self.X[samples[p], deref(best_split.proj_vec_indices)[j]] - \
                        deref(best_split.proj_vec_weights)[j]
                    temp_d += diff * diff

                if temp_d <= best_split.threshold:
                    p += 1
                else:
                    partition_end -= 1
                    samples[p], samples[partition_end] = samples[partition_end], samples[p]

            self.criterion.reset()
            self.criterion.update(best_split.pos)
            self.criterion.children_impurity(&best_split.impurity_left,
                                             &best_split.impurity_right)
            best_split.improvement = self.criterion.impurity_improvement(
                impurity, best_split.impurity_left, best_split.impurity_right)

        deref(spherical_split).proj_vec_indices = best_split.proj_vec_indices
        deref(spherical_split).proj_vec_weights = best_split.proj_vec_weights
        deref(spherical_split).feature = best_split.feature
        deref(spherical_split).pos = best_split.pos
        deref(spherical_split).threshold = best_split.threshold
        deref(spherical_split).improvement = best_split.improvement
        deref(spherical_split).impurity_left = best_split.impurity_left
        deref(spherical_split).impurity_right = best_split.impurity_right

        parent_record.n_constant_features = 0
        return 0
