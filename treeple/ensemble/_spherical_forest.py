from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from ..tree import SphericalDecisionTreeClassifier, SphericalDecisionTreeRegressor


def _generate_sample_indices(random_state, n_samples, bootstrap, max_samples):
    if max_samples is None:
        sample_size = n_samples
    elif isinstance(max_samples, float):
        if not (0.0 < max_samples <= 1.0):
            raise ValueError("max_samples as a float must be in (0.0, 1.0].")
        sample_size = max(1, int(round(max_samples * n_samples)))
    else:
        if max_samples < 1 or max_samples > n_samples:
            raise ValueError("max_samples as an int must be in [1, n_samples].")
        sample_size = int(max_samples)

    if bootstrap:
        return random_state.randint(0, n_samples, sample_size)
    return random_state.choice(n_samples, size=sample_size, replace=False)


def _fit_estimator(estimator, X, y, sample_weight, indices):
    if sample_weight is None:
        estimator.fit(X[indices], y[indices])
    else:
        estimator.fit(X[indices], y[indices], sample_weight=sample_weight[indices])
    return estimator


class _BaseSphericalForest(BaseEstimator):
    _tree_class = None

    def __init__(
        self,
        *,
        n_estimators=100,
        criterion,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        min_impurity_decrease=0.0,
        n_center_candidates=16,
        radius_candidates=None,
        bootstrap=True,
        max_samples=None,
        n_jobs=None,
        random_state=None,
        verbose=0,
    ):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.n_center_candidates = n_center_candidates
        self.radius_candidates = radius_candidates
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def _validate_forest_parameters(self):
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be at least 1.")

    def _make_base_estimator(self, random_state):
        return self._tree_class(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            min_impurity_decrease=self.min_impurity_decrease,
            n_center_candidates=self.n_center_candidates,
            radius_candidates=self.radius_candidates,
            random_state=random_state,
        )

    @property
    def feature_importances_(self):
        check_is_fitted(self, "estimators_")
        all_importances = np.asarray(
            [tree.feature_importances_ for tree in self.estimators_], dtype=np.float64
        )
        return np.mean(all_importances, axis=0)


class SphericalRandomForestClassifier(ClassifierMixin, _BaseSphericalForest):
    """Random forest classifier whose base trees use hypersphere split rules."""

    _tree_class = SphericalDecisionTreeClassifier

    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        min_impurity_decrease=0.0,
        n_center_candidates=16,
        radius_candidates=None,
        bootstrap=True,
        max_samples=None,
        n_jobs=None,
        random_state=None,
        verbose=0,
        class_weight=None,
    ):
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            n_center_candidates=n_center_candidates,
            radius_candidates=radius_candidates,
            bootstrap=bootstrap,
            max_samples=max_samples,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )
        self.class_weight = class_weight

    def _make_base_estimator(self, random_state):
        estimator = super()._make_base_estimator(random_state)
        estimator.class_weight = self.class_weight
        return estimator

    def fit(self, X, y, sample_weight=None):
        self._validate_forest_parameters()
        X, y = check_X_y(X, y, dtype=np.float64)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self.n_features_in_ = X.shape[1]

        random_state = check_random_state(self.random_state)
        seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_estimators)
        sample_indices = [
            _generate_sample_indices(
                check_random_state(seed), X.shape[0], self.bootstrap, self.max_samples
            )
            for seed in seeds
        ]
        estimators = [self._make_base_estimator(seed) for seed in seeds]
        self.estimators_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_fit_estimator)(estimator, X, y, sample_weight, indices)
            for estimator, indices in zip(estimators, sample_indices)
        )
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "estimators_")
        X = check_array(X, dtype=np.float64)
        proba = np.zeros((X.shape[0], self.n_classes_), dtype=np.float64)
        class_to_index = {klass: idx for idx, klass in enumerate(self.classes_)}

        for estimator in self.estimators_:
            estimator_proba = estimator.predict_proba(X)
            for local_idx, klass in enumerate(estimator.classes_):
                proba[:, class_to_index[klass]] += estimator_proba[:, local_idx]
        proba /= len(self.estimators_)
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class SphericalRandomForestRegressor(RegressorMixin, _BaseSphericalForest):
    """Random forest regressor whose base trees use hypersphere split rules."""

    _tree_class = SphericalDecisionTreeRegressor

    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        min_impurity_decrease=0.0,
        n_center_candidates=16,
        radius_candidates=None,
        bootstrap=True,
        max_samples=None,
        n_jobs=None,
        random_state=None,
        verbose=0,
    ):
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            n_center_candidates=n_center_candidates,
            radius_candidates=radius_candidates,
            bootstrap=bootstrap,
            max_samples=max_samples,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

    def fit(self, X, y, sample_weight=None):
        self._validate_forest_parameters()
        X, y = check_X_y(X, y, dtype=np.float64, y_numeric=True)
        self.n_features_in_ = X.shape[1]

        random_state = check_random_state(self.random_state)
        seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_estimators)
        sample_indices = [
            _generate_sample_indices(
                check_random_state(seed), X.shape[0], self.bootstrap, self.max_samples
            )
            for seed in seeds
        ]
        estimators = [self._make_base_estimator(seed) for seed in seeds]
        self.estimators_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_fit_estimator)(estimator, X, y, sample_weight, indices)
            for estimator, indices in zip(estimators, sample_indices)
        )
        return self

    def predict(self, X):
        check_is_fitted(self, "estimators_")
        X = check_array(X, dtype=np.float64)
        predictions = np.asarray([estimator.predict(X) for estimator in self.estimators_])
        return np.mean(predictions, axis=0)
