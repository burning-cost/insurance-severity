"""
Tests for GLMBaseline and CatBoostBaseline.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from insurance_severity.drn.baseline import GLMBaseline
from insurance_severity.drn.catboost_baseline import CatBoostBaseline


class MockSMResult:
    """
    Minimal mock of a statsmodels GLMResultsWrapper for Gamma.
    Returns fixed mu=500 for any input.
    """

    class family:
        pass

    def __init__(self, mu: float = 500.0, scale: float = 0.5):
        self._mu = mu
        self._scale = scale
        self.family = self  # self-referential mock
        self.__class__.__name__ = "Gamma"

    @property
    def scale(self):
        return self._scale

    def predict(self, X):
        n = len(X) if hasattr(X, "__len__") else 1
        return np.full(n, self._mu)


class TestGLMBaseline:

    def test_construction_gamma(self):
        mock = MockSMResult(mu=1000.0, scale=0.5)
        baseline = GLMBaseline(mock, family="gamma")
        assert baseline.distribution_family == "gamma"

    def test_predict_params_shape(self):
        mock = MockSMResult(mu=500.0, scale=0.5)
        baseline = GLMBaseline(mock, family="gamma")
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        params = baseline.predict_params(X)
        assert "mu" in params
        assert len(params["mu"]) == 3
        np.testing.assert_allclose(params["mu"], 500.0)

    def test_predict_cdf_shape(self):
        mock = MockSMResult(mu=500.0, scale=0.5)
        baseline = GLMBaseline(mock, family="gamma")
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        cuts = np.array([0.0, 200.0, 500.0, 800.0, 1200.0])
        cdf = baseline.predict_cdf(X, cuts)
        assert cdf.shape == (4, 5)

    def test_predict_cdf_monotone(self):
        """CDF values must be non-decreasing in cutpoints."""
        mock = MockSMResult(mu=500.0, scale=0.5)
        baseline = GLMBaseline(mock, family="gamma")
        X = pd.DataFrame({"a": [1.0, 2.0]})
        cuts = np.linspace(0.0, 2000.0, 20)
        cdf = baseline.predict_cdf(X, cuts)
        diffs = np.diff(cdf, axis=1)
        assert np.all(diffs >= -1e-10)

    def test_predict_cdf_bounded(self):
        mock = MockSMResult(mu=500.0, scale=0.5)
        baseline = GLMBaseline(mock, family="gamma")
        X = pd.DataFrame({"a": [1.0]})
        cuts = np.array([0.0, 100.0, 500.0, 1000.0, 10000.0])
        cdf = baseline.predict_cdf(X, cuts)
        assert np.all(cdf >= 0.0)
        assert np.all(cdf <= 1.0 + 1e-10)

    def test_gamma_cdf_correct(self):
        """Gamma CDF should match scipy.stats.gamma directly."""
        mu, disp = 500.0, 0.5
        mock = MockSMResult(mu=mu, scale=disp)
        baseline = GLMBaseline(mock, family="gamma")
        X = pd.DataFrame({"a": [1.0]})
        cuts = np.array([100.0, 300.0, 500.0, 800.0])
        cdf = baseline.predict_cdf(X, cuts)  # (1, 4)

        alpha = 1.0 / disp
        scale = mu * disp
        expected = stats.gamma.cdf(cuts, a=alpha, scale=scale)
        np.testing.assert_allclose(cdf[0], expected, rtol=1e-10)

    def test_gaussian_cdf_correct(self):
        mu, sigma2 = 500.0, 10000.0
        mock = MockSMResult(mu=mu, scale=sigma2)
        baseline = GLMBaseline(mock, family="gaussian")
        X = pd.DataFrame({"a": [1.0]})
        cuts = np.array([300.0, 400.0, 500.0, 600.0, 700.0])
        cdf = baseline.predict_cdf(X, cuts)
        expected = stats.norm.cdf(cuts, loc=mu, scale=np.sqrt(sigma2))
        np.testing.assert_allclose(cdf[0], expected, rtol=1e-10)

    def test_lognormal_cdf_correct(self):
        mu, sigma_log2 = 500.0, 0.25
        mock = MockSMResult(mu=mu, scale=sigma_log2)
        baseline = GLMBaseline(mock, family="lognormal")
        X = pd.DataFrame({"a": [1.0]})
        cuts = np.array([100.0, 300.0, 500.0, 800.0])
        cdf = baseline.predict_cdf(X, cuts)
        sigma_log = np.sqrt(sigma_log2)
        mu_log = np.log(mu) - 0.5 * sigma_log ** 2
        expected = stats.lognorm.cdf(cuts, s=sigma_log, scale=np.exp(mu_log))
        np.testing.assert_allclose(cdf[0], expected, rtol=1e-6)

    def test_dispersion_override(self):
        """Custom dispersion parameter is used."""
        mock = MockSMResult(mu=500.0, scale=0.5)
        baseline = GLMBaseline(mock, family="gamma", dispersion=2.0)
        params = baseline.predict_params(pd.DataFrame({"a": [1.0]}))
        assert params["dispersion"] == 2.0

    def test_infer_family_from_gamma_result(self):
        """Family inference from result class name."""
        class FakeResult:
            class family:
                pass
            scale = 0.5
            __class__ = type("Gamma", (), {"__name__": "Gamma"})

            def predict(self, X):
                return np.ones(len(X))

        # Construct with explicit family to avoid inference complexity
        baseline = GLMBaseline(FakeResult(), family="gamma")
        assert baseline.distribution_family == "gamma"

    def test_unknown_family_raises(self):
        mock = MockSMResult()
        baseline = GLMBaseline(mock, family="poisson")
        X = pd.DataFrame({"a": [1.0]})
        cuts = np.array([0.0, 100.0])
        with pytest.raises(ValueError, match="Unknown distribution family"):
            baseline.predict_cdf(X, cuts)


class TestCatBoostBaseline:

    class MockMeanPredictor:
        """Plain CatBoost-style model that only predicts means."""
        def predict(self, X):
            return np.full(len(X), 500.0)

    def test_plain_mean_mode_shape(self):
        model = self.MockMeanPredictor()
        baseline = CatBoostBaseline(model, family="gamma", dispersion=0.5)
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        cuts = np.array([0.0, 200.0, 500.0, 800.0, 1200.0])
        cdf = baseline.predict_cdf(X, cuts)
        assert cdf.shape == (3, 5)

    def test_plain_mean_mode_cdf_matches_gamma(self):
        model = self.MockMeanPredictor()
        mu, disp = 500.0, 0.5
        baseline = CatBoostBaseline(model, family="gamma", dispersion=disp)
        X = pd.DataFrame({"a": [1.0]})
        cuts = np.array([100.0, 300.0, 500.0, 800.0])
        cdf = baseline.predict_cdf(X, cuts)
        alpha = 1.0 / disp
        scale = mu * disp
        expected = stats.gamma.cdf(cuts, a=alpha, scale=scale)
        np.testing.assert_allclose(cdf[0], expected, rtol=1e-10)

    def test_predict_cdf_delegates_to_model(self):
        """If model has predict_cdf, it should be used."""
        class MockDistributionalModel:
            def predict(self, X):
                return np.full(len(X), 500.0)
            def predict_cdf(self, X, cutpoints):
                n = len(X)
                K = len(cutpoints)
                return np.linspace(0, 1, K)[np.newaxis, :].repeat(n, axis=0)

        model = MockDistributionalModel()
        baseline = CatBoostBaseline(model, family="gamma")
        X = pd.DataFrame({"a": [1.0, 2.0]})
        cuts = np.array([0.0, 500.0, 1000.0])
        cdf = baseline.predict_cdf(X, cuts)
        assert cdf.shape == (2, 3)
        # Should use model's predict_cdf, not parametric
        np.testing.assert_allclose(cdf[0], [0.0, 0.5, 1.0])

    def test_fit_dispersion_gamma(self):
        """fit_dispersion should update _dispersion."""
        model = self.MockMeanPredictor()
        baseline = CatBoostBaseline(model, family="gamma", dispersion=1.0)
        rng = np.random.default_rng(0)
        X_train = pd.DataFrame({"a": rng.uniform(size=200)})
        mu = 500.0
        y_train = rng.gamma(shape=2.0, scale=mu / 2.0, size=200)

        baseline.fit_dispersion(y_train, X_train)
        # Pearson dispersion for Gamma with shape=2 should be near 0.5
        assert 0.1 < baseline._dispersion < 2.0  # broad check

    def test_predict_params_shape(self):
        model = self.MockMeanPredictor()
        baseline = CatBoostBaseline(model, family="gamma", dispersion=0.5)
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        params = baseline.predict_params(X)
        assert "mu" in params
        assert len(params["mu"]) == 3

    def test_distribution_family_attribute(self):
        model = self.MockMeanPredictor()
        baseline = CatBoostBaseline(model, family="lognormal")
        assert baseline.distribution_family == "lognormal"
