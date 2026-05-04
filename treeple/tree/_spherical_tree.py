from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import check_random_state
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


@dataclass
class _SphereSplit:
    features: np.ndarray
    center: np.ndarray
    radius_squared: float
    improvement: float
    weighted_impurity_decrease: float
    left_impurity: float
    right_impurity: float


@dataclass
class _SphereNode:
    node_id: int
    depth: int
    n_node_samples: int
    weighted_n_node_samples: float
    impurity: float
    prediction: object
    value: np.ndarray | float
    is_leaf: bool = True
    features: np.ndarray | None = None
    center: np.ndarray | None = None
    radius_squared: float | None = None
    left_child: int = -1
    right_child: int = -1
    impurity_decrease: float = 0.0


class _BaseSphericalDecisionTree(BaseEstimator):
    """Base implementation for greedy hypersphere-split decision trees."""

    _task = None

    def __init__(
        self,
        *,
        criterion,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        min_impurity_decrease=0.0,
        n_center_candidates=16,
        radius_candidates=None,
        random_state=None,
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.n_center_candidates = n_center_candidates
        self.radius_candidates = radius_candidates
        self.random_state = random_state

    def _fit_tree(self, X, y, sample_weight=None):
        self._validate_hyperparameters()
        self.random_state_ = check_random_state(self.random_state)
        self.n_features_in_ = X.shape[1]
        self.max_features_ = self._resolve_max_features(self.max_features, self.n_features_in_)
        self.min_samples_split_ = self._resolve_min_samples(
            self.min_samples_split, X.shape[0], "min_samples_split"
        )
        self.min_samples_leaf_ = self._resolve_min_samples(
            self.min_samples_leaf, X.shape[0], "min_samples_leaf"
        )

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0], dtype=np.float64)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if sample_weight.shape != (X.shape[0],):
                raise ValueError("sample_weight must have shape (n_samples,).")
            if np.any(sample_weight < 0):
                raise ValueError("sample_weight cannot contain negative weights.")

        self._total_weight = float(np.sum(sample_weight))
        self.min_weight_leaf_ = self.min_weight_fraction_leaf * self._total_weight
        self._feature_importances = np.zeros(self.n_features_in_, dtype=np.float64)
        self._nodes: list[_SphereNode] = []

        sample_indices = np.arange(X.shape[0])
        self.root_ = self._build_node(X, y, sample_weight, sample_indices, depth=0)
        self.node_count_ = len(self._nodes)
        return self

    def _validate_hyperparameters(self):
        if self.max_depth is not None and self.max_depth < 0:
            raise ValueError("max_depth must be non-negative or None.")
        if not (0.0 <= self.min_weight_fraction_leaf <= 0.5):
            raise ValueError("min_weight_fraction_leaf must be in [0.0, 0.5].")
        if self.min_impurity_decrease < 0:
            raise ValueError("min_impurity_decrease must be non-negative.")
        if self.n_center_candidates is not None and self.n_center_candidates < 0:
            raise ValueError("n_center_candidates must be non-negative or None.")
        if self.radius_candidates is not None and self.radius_candidates < 1:
            raise ValueError("radius_candidates must be a positive integer or None.")

    def _build_node(self, X, y, sample_weight, sample_indices, depth):
        y_node = y[sample_indices]
        weight_node = sample_weight[sample_indices]
        node_id = len(self._nodes)
        impurity = self._impurity(y_node, weight_node)
        prediction, value = self._leaf_value(y_node, weight_node)
        node = _SphereNode(
            node_id=node_id,
            depth=depth,
            n_node_samples=sample_indices.shape[0],
            weighted_n_node_samples=float(np.sum(weight_node)),
            impurity=float(impurity),
            prediction=prediction,
            value=value,
        )
        self._nodes.append(node)

        if self._should_stop(node, y_node):
            return node_id

        split = self._best_split(X, y, sample_weight, sample_indices, impurity)
        if split is None or split.weighted_impurity_decrease < self.min_impurity_decrease:
            return node_id

        distances = self._squared_distances(X[sample_indices], split.features, split.center)
        left_mask = distances <= split.radius_squared
        if (
            np.count_nonzero(left_mask) < self.min_samples_leaf_
            or np.count_nonzero(~left_mask) < self.min_samples_leaf_
        ):
            return node_id

        left_indices = sample_indices[left_mask]
        right_indices = sample_indices[~left_mask]

        node.is_leaf = False
        node.features = split.features
        node.center = split.center
        node.radius_squared = float(split.radius_squared)
        node.impurity_decrease = float(split.weighted_impurity_decrease)

        importance_share = split.weighted_impurity_decrease / split.features.shape[0]
        self._feature_importances[split.features] += importance_share

        node.left_child = self._build_node(X, y, sample_weight, left_indices, depth + 1)
        node.right_child = self._build_node(X, y, sample_weight, right_indices, depth + 1)
        return node_id

    def _should_stop(self, node, y_node):
        if self.max_depth is not None and node.depth >= self.max_depth:
            return True
        if node.n_node_samples < self.min_samples_split_:
            return True
        if node.n_node_samples < 2 * self.min_samples_leaf_:
            return True
        if node.weighted_n_node_samples < 2 * self.min_weight_leaf_:
            return True
        if node.impurity <= 1e-12:
            return True
        return False

    def _best_split(self, X, y, sample_weight, sample_indices, parent_impurity):
        features = self.random_state_.choice(
            self.n_features_in_, size=self.max_features_, replace=False
        )
        X_node = X[sample_indices]
        y_node = y[sample_indices]
        weight_node = sample_weight[sample_indices]
        centers = self._candidate_centers(X_node, y_node, weight_node, features)

        best = None
        for center in centers:
            distances = self._squared_distances(X_node, features, center)
            split = self._best_radius(
                distances,
                y_node,
                weight_node,
                parent_impurity,
                features,
                center,
            )
            if split is not None and (
                best is None or split.weighted_impurity_decrease > best.weighted_impurity_decrease
            ):
                best = split
        return best

    def _candidate_centers(self, X_node, y_node, weight_node, features):
        X_features = X_node[:, features]
        centers = [np.average(X_features, axis=0, weights=weight_node)]

        if self._task == "classification":
            for encoded_class in np.unique(y_node):
                mask = y_node == encoded_class
                if np.any(mask):
                    centers.append(np.average(X_features[mask], axis=0, weights=weight_node[mask]))

        n_random = X_node.shape[0] if self.n_center_candidates is None else self.n_center_candidates
        n_random = min(n_random, X_node.shape[0])
        if n_random > 0:
            random_indices = self.random_state_.choice(
                X_node.shape[0], size=n_random, replace=False
            )
            centers.extend(X_features[random_indices])

        unique_centers = []
        seen = set()
        for center in centers:
            center = np.asarray(center, dtype=np.float64)
            key = center.tobytes()
            if key not in seen:
                seen.add(key)
                unique_centers.append(center)
        return unique_centers

    def _best_radius(self, distances, y_node, weight_node, parent_impurity, features, center):
        order = np.argsort(distances, kind="mergesort")
        sorted_distances = distances[order]
        candidate_positions = np.flatnonzero(np.diff(sorted_distances) > 0.0) + 1
        if candidate_positions.size == 0:
            return None

        valid = (
            (candidate_positions >= self.min_samples_leaf_)
            & ((distances.shape[0] - candidate_positions) >= self.min_samples_leaf_)
        )
        candidate_positions = candidate_positions[valid]
        if candidate_positions.size == 0:
            return None

        if self.radius_candidates is not None and candidate_positions.size > self.radius_candidates:
            chosen = np.linspace(0, candidate_positions.size - 1, self.radius_candidates)
            candidate_positions = candidate_positions[np.unique(np.round(chosen).astype(int))]

        return self._best_radius_for_task(
            sorted_distances,
            y_node[order],
            weight_node[order],
            candidate_positions,
            parent_impurity,
            features,
            center,
        )

    def _make_split(
        self,
        sorted_distances,
        position,
        weighted_left,
        weighted_right,
        left_impurity,
        right_impurity,
        parent_impurity,
        features,
        center,
    ):
        weighted_total = weighted_left + weighted_right
        if weighted_left < self.min_weight_leaf_ or weighted_right < self.min_weight_leaf_:
            return None

        child_impurity = (
            weighted_left * left_impurity + weighted_right * right_impurity
        ) / weighted_total
        improvement = parent_impurity - child_impurity
        weighted_impurity_decrease = weighted_total / self._total_weight * improvement
        if weighted_impurity_decrease <= 0:
            return None

        radius_squared = (sorted_distances[position - 1] + sorted_distances[position]) / 2.0
        return _SphereSplit(
            features=features.copy(),
            center=center.copy(),
            radius_squared=float(radius_squared),
            improvement=float(improvement),
            weighted_impurity_decrease=float(weighted_impurity_decrease),
            left_impurity=float(left_impurity),
            right_impurity=float(right_impurity),
        )

    def _predict_node_id(self, x):
        node_id = self.root_
        while not self._nodes[node_id].is_leaf:
            node = self._nodes[node_id]
            distance = float(np.sum((x[node.features] - node.center) ** 2))
            node_id = node.left_child if distance <= node.radius_squared else node.right_child
        return node_id

    def apply(self, X):
        check_is_fitted(self, "root_")
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but this estimator was fitted with "
                f"{self.n_features_in_} features."
            )
        return np.array([self._predict_node_id(x) for x in X], dtype=np.intp)

    @property
    def feature_importances_(self):
        check_is_fitted(self, "_feature_importances")
        total = np.sum(self._feature_importances)
        if total <= 0:
            return self._feature_importances.copy()
        return self._feature_importances / total

    @staticmethod
    def _squared_distances(X, features, center):
        diff = X[:, features] - center
        return np.einsum("ij,ij->i", diff, diff)

    @staticmethod
    def _resolve_min_samples(value, n_samples, name):
        if isinstance(value, Integral):
            if value < 1:
                raise ValueError(f"{name} must be at least 1.")
            return int(value)
        if isinstance(value, Real):
            if not (0.0 < value <= 1.0):
                raise ValueError(f"{name} as a float must be in (0.0, 1.0].")
            return int(np.ceil(value * n_samples))
        raise ValueError(f"{name} must be an int or a float.")

    @staticmethod
    def _resolve_max_features(value, n_features):
        if value is None:
            return n_features
        if value in {"sqrt", "auto"}:
            return max(1, int(np.sqrt(n_features)))
        if value == "log2":
            return max(1, int(np.log2(n_features)))
        if isinstance(value, Integral):
            if value < 1 or value > n_features:
                raise ValueError("max_features must be in [1, n_features].")
            return int(value)
        if isinstance(value, Real):
            if not (0.0 < value <= 1.0):
                raise ValueError("max_features as a float must be in (0.0, 1.0].")
            return max(1, int(value * n_features))
        raise ValueError('max_features must be int, float, "sqrt", "log2", "auto", or None.')


class SphericalDecisionTreeClassifier(ClassifierMixin, _BaseSphericalDecisionTree):
    """Decision tree classifier using inside/outside hypersphere split rules.

    Each internal node chooses a feature subspace, candidate centers in that
    subspace, and the radius that maximizes impurity decrease. The left child
    receives samples satisfying ``||x_features - center||^2 <= radius^2``.
    """

    _task = "classification"

    def __init__(
        self,
        *,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        min_impurity_decrease=0.0,
        n_center_candidates=16,
        radius_candidates=None,
        random_state=None,
        class_weight=None,
    ):
        super().__init__(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            n_center_candidates=n_center_candidates,
            radius_candidates=radius_candidates,
            random_state=random_state,
        )
        self.class_weight = class_weight

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y, dtype=np.float64)
        check_classification_targets(y)
        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        self.n_classes_ = self.classes_.shape[0]

        if self.class_weight is not None:
            class_sample_weight = compute_sample_weight(self.class_weight, y)
            sample_weight = (
                class_sample_weight
                if sample_weight is None
                else np.asarray(sample_weight) * class_sample_weight
            )

        return self._fit_tree(X, y_encoded.astype(np.intp), sample_weight)

    def predict(self, X):
        check_is_fitted(self, "classes_")
        X = check_array(X, dtype=np.float64)
        node_ids = self.apply(X)
        encoded = np.array([self._nodes[node_id].prediction for node_id in node_ids], dtype=np.intp)
        return self.classes_[encoded]

    def predict_proba(self, X):
        check_is_fitted(self, "classes_")
        X = check_array(X, dtype=np.float64)
        node_ids = self.apply(X)
        return np.vstack([self._nodes[node_id].value for node_id in node_ids])

    def _leaf_value(self, y_node, weight_node):
        counts = np.bincount(y_node, weights=weight_node, minlength=self.n_classes_)
        total = np.sum(counts)
        proba = counts / total if total > 0 else np.full(self.n_classes_, 1.0 / self.n_classes_)
        return int(np.argmax(counts)), proba

    def _impurity(self, y_node, weight_node):
        counts = np.bincount(y_node, weights=weight_node, minlength=self.n_classes_)
        total = np.sum(counts)
        if total <= 0:
            return 0.0
        proba = counts / total
        if self.criterion == "gini":
            return 1.0 - np.dot(proba, proba)
        if self.criterion in {"entropy", "log_loss"}:
            positive = proba > 0
            return -float(np.sum(proba[positive] * np.log(proba[positive])))
        raise ValueError('criterion must be "gini", "entropy", or "log_loss".')

    def _best_radius_for_task(
        self,
        sorted_distances,
        sorted_y,
        sorted_weight,
        candidate_positions,
        parent_impurity,
        features,
        center,
    ):
        right_counts = np.bincount(
            sorted_y, weights=sorted_weight, minlength=self.n_classes_
        ).astype(np.float64)
        left_counts = np.zeros(self.n_classes_, dtype=np.float64)
        best = None
        previous = 0

        for position in candidate_positions:
            y_slice = sorted_y[previous:position]
            weight_slice = sorted_weight[previous:position]
            left_counts += np.bincount(y_slice, weights=weight_slice, minlength=self.n_classes_)
            right_counts -= np.bincount(y_slice, weights=weight_slice, minlength=self.n_classes_)
            previous = position

            weighted_left = float(np.sum(left_counts))
            weighted_right = float(np.sum(right_counts))
            left_impurity = self._counts_impurity(left_counts, weighted_left)
            right_impurity = self._counts_impurity(right_counts, weighted_right)
            split = self._make_split(
                sorted_distances,
                position,
                weighted_left,
                weighted_right,
                left_impurity,
                right_impurity,
                parent_impurity,
                features,
                center,
            )
            if split is not None and (
                best is None or split.weighted_impurity_decrease > best.weighted_impurity_decrease
            ):
                best = split
        return best

    def _counts_impurity(self, counts, total):
        if total <= 0:
            return 0.0
        proba = counts / total
        if self.criterion == "gini":
            return 1.0 - float(np.dot(proba, proba))
        positive = proba > 0
        return -float(np.sum(proba[positive] * np.log(proba[positive])))


class SphericalDecisionTreeRegressor(RegressorMixin, _BaseSphericalDecisionTree):
    """Decision tree regressor using inside/outside hypersphere split rules."""

    _task = "regression"

    def __init__(
        self,
        *,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        min_impurity_decrease=0.0,
        n_center_candidates=16,
        radius_candidates=None,
        random_state=None,
    ):
        super().__init__(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            n_center_candidates=n_center_candidates,
            radius_candidates=radius_candidates,
            random_state=random_state,
        )

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y, dtype=np.float64, y_numeric=True)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 1:
            raise ValueError("SphericalDecisionTreeRegressor currently supports single-output y.")
        if self.criterion not in {"squared_error", "friedman_mse"}:
            raise ValueError('criterion must be "squared_error" or "friedman_mse".')
        return self._fit_tree(X, y, sample_weight)

    def predict(self, X):
        check_is_fitted(self, "root_")
        X = check_array(X, dtype=np.float64)
        node_ids = self.apply(X)
        return np.array([self._nodes[node_id].prediction for node_id in node_ids], dtype=np.float64)

    def _leaf_value(self, y_node, weight_node):
        prediction = float(np.average(y_node, weights=weight_node))
        return prediction, prediction

    def _impurity(self, y_node, weight_node):
        weighted_total = float(np.sum(weight_node))
        if weighted_total <= 0:
            return 0.0
        mean = float(np.sum(weight_node * y_node) / weighted_total)
        return max(0.0, float(np.sum(weight_node * (y_node - mean) ** 2) / weighted_total))

    def _best_radius_for_task(
        self,
        sorted_distances,
        sorted_y,
        sorted_weight,
        candidate_positions,
        parent_impurity,
        features,
        center,
    ):
        left_w = 0.0
        left_y = 0.0
        left_y2 = 0.0
        right_w = float(np.sum(sorted_weight))
        right_y = float(np.sum(sorted_weight * sorted_y))
        right_y2 = float(np.sum(sorted_weight * sorted_y**2))
        best = None
        previous = 0

        for position in candidate_positions:
            weight_slice = sorted_weight[previous:position]
            y_slice = sorted_y[previous:position]
            moved_w = float(np.sum(weight_slice))
            moved_y = float(np.sum(weight_slice * y_slice))
            moved_y2 = float(np.sum(weight_slice * y_slice**2))

            left_w += moved_w
            left_y += moved_y
            left_y2 += moved_y2
            right_w -= moved_w
            right_y -= moved_y
            right_y2 -= moved_y2
            previous = position

            left_impurity = self._variance_from_sums(left_w, left_y, left_y2)
            right_impurity = self._variance_from_sums(right_w, right_y, right_y2)
            split = self._make_split(
                sorted_distances,
                position,
                left_w,
                right_w,
                left_impurity,
                right_impurity,
                parent_impurity,
                features,
                center,
            )
            if split is not None and (
                best is None or split.weighted_impurity_decrease > best.weighted_impurity_decrease
            ):
                best = split
        return best

    @staticmethod
    def _variance_from_sums(weight, weighted_y, weighted_y2):
        if weight <= 0:
            return 0.0
        return max(0.0, weighted_y2 / weight - (weighted_y / weight) ** 2)


# Cython-backed public estimators. The pure-Python prototype above is kept in
# this module as readable reference code while the exported classes below use the
# same tree-builder architecture as treeple's oblique trees.
import copy as _copy

from scipy.sparse import issparse as _issparse
from sklearn.utils._param_validation import Hidden as _Hidden
from sklearn.utils._param_validation import Interval as _Interval
from sklearn.utils._param_validation import StrOptions as _StrOptions

from .._lib.sklearn.tree import DecisionTreeClassifier as _SkDecisionTreeClassifier
from .._lib.sklearn.tree import DecisionTreeRegressor as _SkDecisionTreeRegressor
from .._lib.sklearn.tree import _criterion as _sk_criterion
from .._lib.sklearn.tree._criterion import BaseCriterion as _BaseCriterion
from .._lib.sklearn.tree._tree import BestFirstTreeBuilder as _BestFirstTreeBuilder
from .._lib.sklearn.tree._tree import DepthFirstTreeBuilder as _DepthFirstTreeBuilder
from ._spherical_splitter import BestSphericalSplitter as _BestSphericalSplitter
from ._spherical_tree_backend import SphericalTree as _CythonSphericalTree


_CRITERIA_CLF = {
    "gini": _sk_criterion.Gini,
    "log_loss": _sk_criterion.Entropy,
    "entropy": _sk_criterion.Entropy,
}
_CRITERIA_REG = {
    "squared_error": _sk_criterion.MSE,
    "friedman_mse": _sk_criterion.FriedmanMSE,
    "absolute_error": _sk_criterion.MAE,
    "poisson": _sk_criterion.Poisson,
}


class _CythonSphericalTreeMixin:
    _parameter_constraints_extra = {
        "n_center_candidates": [_Interval(Integral, 1, None, closed="left")],
        "radius_candidates": [_Interval(Integral, 1, None, closed="left"), None],
    }

    def _make_criterion(self, y):
        if isinstance(self.criterion, _BaseCriterion):
            return _copy.deepcopy(self.criterion)
        if self._spherical_task == "classification":
            return _CRITERIA_CLF[self.criterion](self.n_outputs_, self.n_classes_)
        return _CRITERIA_REG[self.criterion](self.n_outputs_, y.shape[0])

    @property
    def node_count_(self):
        check_is_fitted(self, "tree_")
        return self.tree_.node_count

    def _build_tree(
        self,
        X,
        y,
        sample_weight,
        missing_values_in_feature_mask,
        min_samples_leaf,
        min_weight_leaf,
        max_leaf_nodes,
        min_samples_split,
        max_depth,
        random_state,
    ):
        if _issparse(X):
            raise ValueError(
                "Sparse input is not supported for spherical trees. "
                "Please convert your data to a dense array."
            )

        criterion = self._make_criterion(y)
        radius_candidates = 0 if self.radius_candidates is None else self.radius_candidates
        splitter = _BestSphericalSplitter(
            criterion,
            self.max_features_,
            min_samples_leaf,
            min_weight_leaf,
            random_state,
            None,
            self.n_center_candidates,
            radius_candidates,
        )

        if self._spherical_task == "classification":
            self.tree_ = _CythonSphericalTree(
                self.n_features_in_,
                self.n_classes_,
                self.n_outputs_,
            )
        else:
            self.tree_ = _CythonSphericalTree(
                self.n_features_in_,
                np.array([1] * self.n_outputs_, dtype=np.intp),
                self.n_outputs_,
            )

        if max_leaf_nodes < 0:
            builder = _DepthFirstTreeBuilder(
                splitter,
                min_samples_split,
                min_samples_leaf,
                min_weight_leaf,
                max_depth,
                self.min_impurity_decrease,
                self.store_leaf_values,
            )
        else:
            builder = _BestFirstTreeBuilder(
                splitter,
                min_samples_split,
                min_samples_leaf,
                min_weight_leaf,
                max_depth,
                max_leaf_nodes,
                self.min_impurity_decrease,
                self.store_leaf_values,
            )

        builder.build(self.tree_, X, y, sample_weight, None)

        if self.n_outputs_ == 1 and self._spherical_task == "classification":
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        self._prune_tree()
        return self


class SphericalDecisionTreeClassifier(_CythonSphericalTreeMixin, _SkDecisionTreeClassifier):
    """Decision tree classifier using Cython-optimized hypersphere split rules."""

    _spherical_task = "classification"
    _parameter_constraints = {
        **_SkDecisionTreeClassifier._parameter_constraints,
        **_CythonSphericalTreeMixin._parameter_constraints_extra,
        "criterion": [_StrOptions({"gini", "entropy", "log_loss"}), _Hidden(_BaseCriterion)],
    }

    def __init__(
        self,
        *,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        min_impurity_decrease=0.0,
        n_center_candidates=16,
        radius_candidates=None,
        random_state=None,
        class_weight=None,
        max_leaf_nodes=None,
        ccp_alpha=0.0,
        store_leaf_values=False,
    ):
        super().__init__(
            criterion=criterion,
            splitter="best",
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            random_state=random_state,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            store_leaf_values=store_leaf_values,
        )
        self.n_center_candidates = n_center_candidates
        self.radius_candidates = radius_candidates


class SphericalDecisionTreeRegressor(_CythonSphericalTreeMixin, _SkDecisionTreeRegressor):
    """Decision tree regressor using Cython-optimized hypersphere split rules."""

    _spherical_task = "regression"
    _parameter_constraints = {
        **_SkDecisionTreeRegressor._parameter_constraints,
        **_CythonSphericalTreeMixin._parameter_constraints_extra,
        "criterion": [
            _StrOptions({"squared_error", "friedman_mse", "absolute_error", "poisson"}),
            _Hidden(_BaseCriterion),
        ],
    }

    def __init__(
        self,
        *,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        min_impurity_decrease=0.0,
        n_center_candidates=16,
        radius_candidates=None,
        random_state=None,
        max_leaf_nodes=None,
        ccp_alpha=0.0,
        store_leaf_values=False,
    ):
        super().__init__(
            criterion=criterion,
            splitter="best",
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            random_state=random_state,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
            store_leaf_values=store_leaf_values,
        )
        self.n_center_candidates = n_center_candidates
        self.radius_candidates = radius_candidates
