"""Tests for regression with one predictor."""

import numpy as np
import pytest
from scipy import integrate, stats

import bayesplain as bp
from bayesplain.core import regression as core_reg


@pytest.fixture(scope="module")
def line():
    rng = np.random.default_rng(4)
    x = rng.uniform(0, 10, 40)
    y = 3.0 + 0.8 * x + rng.normal(0, 2.0, 40)
    return x, y


class TestCore:
    def test_fit_matches_linregress(self, line):
        x, y = line
        fit = core_reg.fit_line(x, y)
        ref = stats.linregress(x, y)
        assert fit.slope == pytest.approx(ref.slope)
        assert fit.slope_se == pytest.approx(ref.stderr)
        assert fit.r_squared == pytest.approx(ref.rvalue**2)

    @pytest.mark.parametrize("n, beta", [(12, 1.0), (20, 0.3), (50, 0.1)])
    def test_bayes_factor_matches_the_zellner_siow_r_squared_form(self, n, beta):
        # Liang et al. (2008): BF(g) = (1+g)^((n-2)/2) (1+g(1-R2))^(-(n-1)/2),
        # with g ~ inverse-gamma(1/2, n r^2 / 2).
        rng = np.random.default_rng(n)
        x = rng.normal(size=n)
        y = beta * x + rng.normal(size=n)
        fit = core_reg.fit_line(x, y)
        r2, r = fit.r_squared, 0.707
        prior = stats.invgamma(0.5, scale=n * r * r / 2)

        def integrand(g):
            return np.exp(
                (n - 2) / 2 * np.log1p(g)
                - (n - 1) / 2 * np.log1p(g * (1 - r2))
                + prior.logpdf(g)
            )

        want = np.log(integrate.quad(integrand, 0, np.inf, limit=500)[0])
        assert core_reg.log_bayes_factor_slope(fit, scale=r) == pytest.approx(
            want, abs=1e-6
        )

    @pytest.mark.parametrize(
        "x, y, match",
        [
            ([1, 2], [1, 2], "at least 3"),
            ([1, 1, 1, 1], [1, 2, 3, 4], "identical"),
            ([1, 2, 3, 4], [2, 4, 6, 8], "exactly on a line"),
            ([1, 2, 3], [1, 2], "line up"),
        ],
    )
    def test_bad_inputs_are_explained(self, x, y, match):
        with pytest.raises(ValueError, match=match):
            core_reg.fit_line(x, y)

    def test_a_second_predictor_points_to_bambi(self):
        with pytest.raises(ValueError, match="Bambi"):
            core_reg.fit_line(np.ones((6, 2)), np.arange(6))


class TestRegression:
    def test_credible_interval_equals_the_confidence_interval(self, line):
        res = bp.regression(*line)
        lo, hi = res.interval(kind="eti")
        flo, fhi = res.frequentist.interval
        assert lo == pytest.approx(flo, abs=1e-9)
        assert hi == pytest.approx(fhi, abs=1e-9)

    def test_prior_moves_the_bayes_factor_not_the_interval(self, line):
        narrow = bp.regression(*line, prior="modest")
        wide = bp.regression(*line, prior="generous")
        assert narrow.interval() == wide.interval()
        assert narrow.bayes_factor().log_bf10 != wide.bayes_factor().log_bf10

    def test_forecast_equals_the_textbook_prediction_interval(self, line):
        forecast = bp.regression(*line).predict(x=5.0)
        lo, hi = forecast.interval()
        flo, fhi = forecast.frequentist_interval
        assert lo == pytest.approx(flo)
        assert hi == pytest.approx(fhi)

    def test_extrapolation_is_flagged(self, line):
        assert "extrapolation" in bp.regression(*line).predict(x=100).what

    def test_nonzero_reference_retargets_both_halves(self, line):
        fit = core_reg.fit_line(*line)
        res = bp.regression(*line, reference=0.8)
        t = (fit.slope - 0.8) / fit.slope_se
        assert res.frequentist.pvalue == pytest.approx(2 * stats.t.sf(abs(t), 38))
        # A reference sitting exactly on the fitted slope is what the data
        # look like under that null, so the evidence must favour it.
        on_the_line = bp.regression(*line, reference=fit.slope)
        assert on_the_line.frequentist.pvalue == pytest.approx(1.0)
        assert on_the_line.bayes_factor().bf10 < 1

    def test_missing_rows_are_dropped(self, line):
        x, y = line
        x = np.append(x, np.nan)
        y = np.append(y, 1.0)
        assert bp.regression(x, y).point() == pytest.approx(
            bp.regression(*line).point()
        )

    def test_line_plot(self, line):
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        assert bp.regression(*line).plot(kind="line") is not None
