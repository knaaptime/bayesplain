"""Tests for the correlation analysis.

The exact sampling density is self-validating: it must integrate to 1 over
(-1, 1) for any rho and n, which pins the transcription without reference to
any other implementation. The Bayes factor is additionally cross-checked
against ``pingouin``'s implementation of Ly et al. (2016).
"""

import numpy as np
import pytest
from scipy import integrate

import bayesplain as bp
from bayesplain.core import correlation as co


@pytest.fixture(scope="module")
def related():
    rng = np.random.default_rng(3)
    x = rng.normal(size=80)
    return x, 0.42 * x + rng.normal(size=80)


class TestSamplingDensity:
    @pytest.mark.parametrize(
        ("rho", "n"),
        [(0.0, 10), (0.5, 20), (-0.8, 8), (0.3, 200), (0.95, 50), (-0.99, 6)],
    )
    def test_integrates_to_one(self, rho, n):
        value, _ = integrate.quad(
            lambda r: np.exp(co.log_sampling_density(r, rho, n)),
            -1.0,
            1.0,
            limit=200,
        )
        assert value == pytest.approx(1.0, abs=1e-6)

    def test_symmetric_under_joint_sign_flip(self):
        a = co.log_sampling_density(0.4, 0.6, 30)
        b = co.log_sampling_density(-0.4, -0.6, 30)
        assert a == pytest.approx(b)

    def test_peaks_near_the_true_value(self):
        grid = np.linspace(-0.99, 0.99, 2001)
        values = co.log_sampling_density(0.7, grid, 500)
        assert grid[np.argmax(values)] == pytest.approx(0.7, abs=0.02)

    @pytest.mark.parametrize("r", [-1.0, 1.0, 1.5])
    def test_boundary_observed_correlation_rejected(self, r):
        with pytest.raises(ValueError, match="strictly inside"):
            co.log_sampling_density(r, 0.0, 20)

    def test_too_few_observations_rejected(self):
        with pytest.raises(ValueError, match="at least 4"):
            co.log_sampling_density(0.5, 0.0, 3)


class TestPrior:
    @pytest.mark.parametrize("kappa", [0.3, 0.5, 1.0, 1.5, 2.0])
    def test_integrates_to_one(self, kappa):
        value, _ = integrate.quad(
            lambda r: np.exp(co.log_prior_density(r, kappa)), -1.0, 1.0
        )
        assert value == pytest.approx(1.0, abs=1e-6)

    def test_flat_at_kappa_one(self):
        grid = np.linspace(-0.9, 0.9, 50)
        density = np.exp(co.log_prior_density(grid, 1.0))
        assert np.allclose(density, 0.5)

    def test_small_kappa_concentrates_near_zero(self):
        assert co.log_prior_density(0.0, 0.3) > co.log_prior_density(0.9, 0.3)

    def test_nonpositive_kappa_rejected(self):
        with pytest.raises(ValueError, match="kappa must be positive"):
            co.log_prior_density(0.0, 0.0)


class TestBayesFactor:
    @pytest.mark.validation
    @pytest.mark.parametrize(
        ("r", "n"),
        [(0.3, 30), (0.6, 20), (0.1, 100), (-0.45, 50), (0.8, 12), (0.05, 500)],
    )
    def test_matches_pingouin(self, r, n):
        pg = pytest.importorskip("pingouin")
        mine = np.exp(co.log_bayes_factor(r, n))
        theirs = float(pg.bayesfactor_pearson(r, n, method="ly", kappa=1.0))
        assert mine == pytest.approx(theirs, rel=1e-5)

    @pytest.mark.validation
    @pytest.mark.parametrize("kappa", [0.5, 1.0, 1.5])
    def test_matches_pingouin_across_prior_widths(self, kappa):
        pg = pytest.importorskip("pingouin")
        mine = np.exp(co.log_bayes_factor(0.4, 40, kappa=kappa))
        theirs = float(pg.bayesfactor_pearson(0.4, 40, method="ly", kappa=kappa))
        assert mine == pytest.approx(theirs, rel=1e-5)

    def test_survives_sample_sizes_that_overflow_a_naive_implementation(self):
        # exp(log_joint - log_null) overflows float64 well before this; the
        # integral is evaluated in log space so that it does not.
        for r, n in [(0.78, 2412), (0.95, 5000), (0.6, 20000)]:
            value = co.log_bayes_factor(r, n)
            assert np.isfinite(value)
            assert value > 100

    def test_sign_of_r_does_not_matter(self):
        assert co.log_bayes_factor(0.5, 40) == pytest.approx(
            co.log_bayes_factor(-0.5, 40)
        )

    def test_zero_correlation_favours_the_null(self):
        assert co.log_bayes_factor(0.0, 200) < 0

    def test_more_data_strengthens_a_real_correlation(self):
        assert co.log_bayes_factor(0.5, 200) > co.log_bayes_factor(0.5, 20)


class TestPosterior:
    def test_normalised(self):
        grid, density = co.posterior_on_grid(0.6, 50)
        assert np.trapezoid(density, grid) == pytest.approx(1.0, abs=1e-6)

    def test_concentrates_around_the_observed_correlation(self):
        grid, density = co.posterior_on_grid(0.6, 2000)
        mean = np.trapezoid(grid * density, grid)
        assert mean == pytest.approx(0.6, abs=0.02)

    def test_grid_stays_inside_the_unit_interval(self):
        grid = co.rho_grid(0.99, 5000)
        assert grid.min() > -1.0 and grid.max() < 1.0
        assert np.all(np.diff(grid) > 0)


class TestPublicAnalysis:
    def test_credible_interval_close_to_fisher_z(self, related):
        res = bp.correlation(*related)
        lo, hi = res.interval(kind="eti")
        flo, fhi = res.frequentist.interval
        assert lo == pytest.approx(flo, abs=0.02)
        assert hi == pytest.approx(fhi, abs=0.02)

    def test_point_estimate_near_pearson_r(self, related):
        res = bp.correlation(*related)
        assert res.point() == pytest.approx(res.frequentist.statistic, abs=0.02)

    def test_causation_note_is_always_present(self, related):
        res = bp.correlation(*related)
        assert any("not a cause" in note for note in res.notes)

    def test_aggregated_adds_the_ecological_warning(self, related):
        res = bp.correlation(*related, aggregated=True)
        assert any("areas, not people" in note for note in res.notes)
        assert any("Robinson" in note for note in res.notes)

    def test_not_aggregated_omits_it(self, related):
        res = bp.correlation(*related)
        assert not any("areas, not people" in note for note in res.notes)

    def test_scatter_plot_is_offered(self, related):
        res = bp.correlation(*related)
        assert "scatter" in res.plot_kinds()

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            bp.correlation([1, 2, 3], [1, 2])

    def test_too_few_pairs_rejected(self):
        with pytest.raises(ValueError, match="at least 4 complete pairs"):
            bp.correlation([1.0, 2.0], [2.0, 4.0])

    def test_constant_variable_rejected(self):
        with pytest.raises(ValueError, match="identical"):
            bp.correlation([1.0] * 10, list(range(10)))

    def test_missing_pairs_are_dropped(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=40)
        y = 0.5 * x + rng.normal(size=40)
        xn = x.copy()
        xn[3] = np.nan
        assert bp.correlation(xn, y).point() == pytest.approx(
            bp.correlation(np.delete(x, 3), np.delete(y, 3)).point(), abs=1e-6
        )

    def test_sensitivity_walks_the_correlation_ladder(self, related):
        text = " ".join(str(bp.correlation(*related).sensitivity()).split())
        for name in bp.priors.CORRELATION_SENSITIVITY_LADDER:
            assert name in text


@pytest.mark.pedagogy
class TestModifiableArealUnit:
    def test_same_relationship_differs_between_geographies(self):
        # Week 8's point, on real data: identical variables, identical county,
        # two aggregations, and the intervals do not overlap. This is the
        # geography talking, not sampling noise.
        pytest.importorskip("pandas")
        tracts = bp.datasets.load_tracts().dropna(
            subset=["median_income", "median_rent"]
        )
        blocks = bp.datasets.load_blockgroups().dropna(
            subset=["median_income", "median_rent"]
        )
        coarse = bp.correlation(
            tracts.median_income, tracts.median_rent, aggregated=True
        )
        fine = bp.correlation(blocks.median_income, blocks.median_rent, aggregated=True)
        assert coarse.point() > fine.point()
        assert coarse.interval()[0] > fine.interval()[1]
