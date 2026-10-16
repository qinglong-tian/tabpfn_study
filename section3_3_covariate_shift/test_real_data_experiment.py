"""Regression tests for the IW training, selection, and target-sampling bugs."""
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

import real_data_experiment as experiment


class ImportanceWeightingTests(unittest.TestCase):
    def test_ratios_preserved_and_sampling_reproducible(self):
        X = np.array([[-1., 0.], [0., 1.], [1., 2.]])
        beta = np.array([1., -0.5])
        w = experiment.training_weights(X, beta)
        assert_allclose(w.mean(), 1)
        assert_allclose(w / w[0], np.exp((X - X[0]) @ beta))
        first = experiment.resample_target(X, np.arange(3), beta, 42)
        second = experiment.resample_target(X, np.arange(3), beta, 42)
        for a, b in zip(first, second):
            assert_array_equal(a, b)
        assert_array_equal(first[0], X[first[2]])
        assert_array_equal(first[1], np.arange(3)[first[2]])

    def test_weighted_cv_weights_both_fit_and_validation(self):
        # One costly validation case changes the selected candidate only when
        # density ratios enter scoring. The spy checks every training call too.
        X = np.arange(10.).reshape(-1, 1)
        y = np.zeros(10)
        y[0] = 10
        weights = np.ones(10)
        weights[0] = 100
        calls = []

        class ConstantModel:
            def __init__(self, value):
                self.value = value

            def fit(self, X, y, **kwargs):
                calls.append((X[:, 0].astype(int), kwargs.get("sample_weight")))
                return self

            def predict(self, X):
                return np.full(len(X), self.value)

        with patch.object(experiment, "xgb_model", side_effect=lambda params, seed: ConstantModel(params["constant"])):
            _, ordinary, _ = experiment.fit_xgboost_cv(X, y, 42, cv_folds=2, workers=1, grid={"constant": [0, 10]})
            calls.clear()
            _, weighted, _ = experiment.fit_xgboost_cv(X, y, 42, cv_folds=2, weights=weights, workers=1, grid={"constant": [0, 10]})
        self.assertEqual(ordinary["params"]["constant"], 0)
        self.assertEqual(weighted["params"]["constant"], 10)
        assert_allclose(weighted["validation_mse"], np.average((y - 10) ** 2, weights=weights))
        self.assertEqual(len(calls), 5)  # 2 candidates x 2 folds + full refit
        for indices, passed_weights in calls:
            assert_allclose(passed_weights, weights[indices] / weights[indices].mean())

    def test_no_shift_reduces_iw_to_naive_xgboost(self):
        rng = np.random.default_rng(123)
        X = rng.normal(size=(60, 2))
        y = X[:, 0] + rng.normal(size=60)
        grid = {"n_estimators": [5, 10], "max_depth": [2]}
        w = experiment.training_weights(X, np.zeros(2))
        plain, plain_best, _ = experiment.fit_xgboost_cv(X, y, 42, cv_folds=2, workers=1, grid=grid)
        iw, iw_best, _ = experiment.fit_xgboost_cv(X, y, 42, cv_folds=2, weights=w, workers=1, grid=grid)
        self.assertEqual(plain_best, iw_best)
        assert_allclose(plain.predict(X), iw.predict(X))


if __name__ == "__main__":
    unittest.main()
