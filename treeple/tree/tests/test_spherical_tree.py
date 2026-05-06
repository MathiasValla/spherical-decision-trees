import numpy as np
from numpy.testing import assert_allclose
from sklearn.base import clone
from sklearn.datasets import make_regression

from treeple.ensemble import SphericalRandomForestClassifier, SphericalRandomForestRegressor
from treeple.tree import SphericalDecisionTreeClassifier, SphericalDecisionTreeRegressor


def test_spherical_classifier_fits_disk_with_stump():
    rng = np.random.RandomState(0)
    X = rng.uniform(-2.0, 2.0, size=(300, 2))
    y = (np.sum(X**2, axis=1) <= 1.0).astype(int)

    clf = SphericalDecisionTreeClassifier(
        max_depth=1,
        max_features=None,
        n_center_candidates=8,
        random_state=0,
    )
    clf.fit(X, y)

    assert clf.score(X, y) > 0.97
    assert clf.node_count_ == 3
    assert_allclose(clf.predict_proba(X).sum(axis=1), 1.0)


def test_spherical_classifier_cloneable():
    clf = SphericalDecisionTreeClassifier(max_depth=2, random_state=0)
    cloned = clone(clf)
    assert cloned.get_params()["max_depth"] == 2


def test_spherical_classifier_allows_single_class_bootstrap_sample():
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = np.zeros(6, dtype=int)
    clf = SphericalDecisionTreeClassifier(random_state=0)
    clf.fit(X, y)

    assert np.all(clf.predict(X) == 0)
    assert_allclose(clf.predict_proba(X), np.ones((6, 1)))


def test_spherical_center_strategy_options_smoke():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(80, 3))
    y_clf = (X[:, 0] ** 2 + X[:, 1] ** 2 <= 1.0).astype(int)
    y_reg = X[:, 0] ** 2 - X[:, 1]

    for strategy in ("random", "target", "hybrid", "radial", "target_radial"):
        clf = SphericalDecisionTreeClassifier(
            max_depth=2,
            n_center_candidates=16,
            radius_candidates=None,
            center_strategy=strategy,
            random_state=0,
        ).fit(X, y_clf)
        reg = SphericalDecisionTreeRegressor(
            max_depth=2,
            n_center_candidates=16,
            radius_candidates=None,
            center_strategy=strategy,
            random_state=0,
        ).fit(X, y_reg)

        assert clf.predict(X[:5]).shape == (5,)
        assert reg.predict(X[:5]).shape == (5,)


def test_spherical_target_radial_can_sample_far_centers():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(120, 2))
    y = (X[:, 0] + X[:, 1] > 0.0).astype(int)

    clf = SphericalDecisionTreeClassifier(
        max_depth=1,
        max_features=None,
        n_center_candidates=80,
        radius_candidates=None,
        center_strategy="target_radial",
        random_state=0,
    ).fit(X, y)

    centers = clf.tree_.get_projection_matrix()
    data_min = X.min(axis=0)
    data_max = X.max(axis=0)
    split_centers = centers[clf.tree_.children_left != -1]

    assert np.any((split_centers < data_min) | (split_centers > data_max))


def test_spherical_classifier_ccp_pruning_preserves_spherical_backend():
    rng = np.random.RandomState(0)
    X = rng.uniform(-2.0, 2.0, size=(240, 2))
    y = (np.sum((X - np.array([0.2, -0.1])) ** 2, axis=1) <= 1.0).astype(int)

    base = SphericalDecisionTreeClassifier(
        max_features=None,
        n_center_candidates=48,
        center_strategy="target_radial",
        random_state=0,
    ).fit(X, y)
    path = base.cost_complexity_pruning_path(X, y).ccp_alphas

    pruned = None
    for alpha in path[1:-1]:
        candidate = SphericalDecisionTreeClassifier(
            max_features=None,
            n_center_candidates=48,
            center_strategy="target_radial",
            ccp_alpha=float(alpha),
            random_state=0,
        ).fit(X, y)
        if 1 < candidate.tree_.node_count < base.tree_.node_count:
            pruned = candidate
            break

    assert pruned is not None
    assert type(pruned.tree_).__name__ == "SphericalTree"
    assert pruned.tree_.get_projection_matrix().shape == (
        pruned.tree_.node_count,
        X.shape[1],
    )
    assert_allclose(
        pruned.tree_.get_projection_matrix()[0],
        base.tree_.get_projection_matrix()[0],
    )
    assert pruned.predict(X[:10]).shape == (10,)


def test_spherical_regressor_ccp_pruning_preserves_spherical_backend():
    rng = np.random.RandomState(1)
    X = rng.normal(size=(180, 3))
    y = X[:, 0] ** 2 - X[:, 1] + 0.1 * rng.normal(size=X.shape[0])

    reg = SphericalDecisionTreeRegressor(
        n_center_candidates=32,
        center_strategy="target_radial",
        ccp_alpha=0.01,
        random_state=0,
    ).fit(X, y)

    assert type(reg.tree_).__name__ == "SphericalTree"
    assert reg.tree_.get_projection_matrix().shape == (reg.tree_.node_count, X.shape[1])
    assert reg.predict(X[:7]).shape == (7,)


def test_spherical_regressor_smoke():
    X, y = make_regression(
        n_samples=80,
        n_features=3,
        n_informative=2,
        noise=0.1,
        random_state=0,
    )
    reg = SphericalDecisionTreeRegressor(max_depth=3, random_state=0)
    reg.fit(X, y)
    pred = reg.predict(X[:5])

    assert pred.shape == (5,)
    assert reg.apply(X[:5]).shape == (5,)
    assert reg.feature_importances_.shape == (3,)


def test_spherical_random_forest_classifier_smoke():
    rng = np.random.RandomState(1)
    X = rng.uniform(-2.0, 2.0, size=(160, 2))
    y = (np.sum((X - np.array([0.25, -0.25])) ** 2, axis=1) <= 0.9).astype(int)

    forest = SphericalRandomForestClassifier(
        n_estimators=5,
        max_depth=2,
        max_features=None,
        random_state=0,
        n_center_candidates=4,
    )
    forest.fit(X, y)

    assert forest.predict(X[:10]).shape == (10,)
    assert_allclose(forest.predict_proba(X[:10]).sum(axis=1), 1.0)


def test_spherical_random_forest_regressor_smoke():
    X, y = make_regression(n_samples=100, n_features=4, random_state=0)
    forest = SphericalRandomForestRegressor(
        n_estimators=3,
        max_depth=2,
        max_features=2,
        random_state=0,
    )
    forest.fit(X, y)

    assert forest.predict(X[:7]).shape == (7,)
    assert forest.feature_importances_.shape == (4,)
