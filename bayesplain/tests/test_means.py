"""Tests for means: the Student-t posteriors and the JZS Bayes factor.

The Bayes factor is cross-validated against ``pingouin``, which implements the
same Rouder et al. (2009) integral independently. Those tests are skipped when
pingouin is not installed, since it is a validation-only dependency.
"""

import numpy as np
import pytest
from scipy import stats

import bayesplain as bp
from bayesplain.core import normal_t as nt


@pytest.fixture(scope="module")
def commutes():
    rng = np.random.default_rng(7)
    return rng.normal(31.0, 9.0, 60), rng.normal(35.0, 12.0, 55)


class TestPosterior:
    def test_matches_the_frequentist_t_interval_exactly(self):
        # The arithmetic is identical; only the licensed sentence differs.
        rng = np.random.default_rng(0)
        x = rng.normal(50, 12, 40)
        res = bp.mean(x)
        lo, hi = res.interval(kind="eti")
        flo, fhi = res.frequentist.interval
        assert lo == pytest.approx(flo, abs=1e-9)
        assert hi == pytest.approx(fhi, abs=1e-9)

    def test_posterior_is_exact(self):
        rng = np.random.default_rng(1)
        res = bp.mean(rng.normal(0, 1, 30))
        assert res.exact
        assert res.monte_carlo_error() == 0.0

    def test_posterior_centres_on_the_sample_mean(self):
        rng = np.random.default_rng(2)
        x = rng.normal(10, 2, 50)
        assert bp.mean(x).point("mean") == pytest.approx(x.mean(), abs=1e-9)

    def test_degrees_of_freedom_are_n_minus_one(self):
        post = nt.mean_posterior(25, 10.0, 3.0)
        assert post.kwds["df"] == 24

    def test_interval_narrows_with_more_data(self):
        rng = np.random.default_rng(3)
        small = np.diff(bp.mean(rng.normal(0, 1, 25)).interval())[0]
        large = np.diff(bp.mean(rng.normal(0, 1, 2500)).interval())[0]
        assert large < small / 5


class TestCompareMeans:
    def test_welch_is_the_default(self, commutes):
        res = bp.compare_means(*commutes)
        assert "Welch" in res.frequentist.test
        assert not res.exact  # Behrens-Fisher: sampled, and says so

    def test_equal_var_gives_a_closed_form(self, commutes):
        res = bp.compare_means(*commutes, equal_var=True)
        assert res.exact
        assert res.monte_carlo_error() == 0.0

    def test_difference_matches_observed_gap(self, commutes):
        a, b = commutes
        res = bp.compare_means(a, b)
        assert res.point("mean") == pytest.approx(b.mean() - a.mean(), abs=0.05)

    def test_credible_interval_close_to_welch_interval(self, commutes):
        res = bp.compare_means(*commutes)
        lo, hi = res.interval(kind="eti")
        flo, fhi = res.frequentist.interval
        assert lo == pytest.approx(flo, abs=0.05)
        assert hi == pytest.approx(fhi, abs=0.05)

    def test_group_order_flips_the_sign(self, commutes):
        a, b = commutes
        forward = bp.compare_means(a, b, seed=5)
        reverse = bp.compare_means(b, a, seed=5)
        assert forward.point("mean") == pytest.approx(-reverse.point("mean"), abs=0.05)

    def test_identical_samples_centre_on_zero(self):
        rng = np.random.default_rng(4)
        x = rng.normal(0, 1, 100)
        res = bp.compare_means(x, x.copy())
        assert res.point("mean") == pytest.approx(0.0, abs=0.05)

    def test_mismatched_spreads_flag_equal_var(self):
        rng = np.random.default_rng(5)
        a, b = rng.normal(0, 1, 40), rng.normal(0, 9, 40)
        res = bp.compare_means(a, b, equal_var=True)
        assert any("spreads differ" in note for note in res.notes)

    def test_small_groups_get_a_normality_note(self):
        rng = np.random.default_rng(6)
        res = bp.compare_means(rng.normal(size=8), rng.normal(size=9))
        assert any("roughly normal" in note for note in res.notes)

    def test_wrong_label_count_rejected(self, commutes):
        with pytest.raises(ValueError, match="exactly two names"):
            bp.compare_means(*commutes, labels=["only one"])


class TestValidation:
    @pytest.mark.parametrize("bad", [[], [1.0], [np.nan, np.inf]])
    def test_too_few_observations_rejected(self, bad):
        with pytest.raises(ValueError, match="at least 2 finite"):
            bp.mean(bad)

    def test_constant_sample_rejected(self):
        with pytest.raises(ValueError, match="identical"):
            bp.mean([5.0] * 20)

    def test_missing_values_are_dropped(self):
        clean = bp.mean([1.0, 2.0, 3.0, 4.0])
        with_nan = bp.mean([1.0, 2.0, np.nan, 3.0, 4.0])
        assert clean.point() == pytest.approx(with_nan.point())


class TestBayesFactor:
    def test_prior_does_not_move_the_estimate(self):
        # The clean demonstration: the reference-prior posterior is fixed by
        # the data, so only the Bayes factor responds to prior= at all.
        rng = np.random.default_rng(8)
        x, y = rng.normal(0, 1, 50), rng.normal(0.4, 1, 50)
        intervals, factors = [], []
        for name in bp.priors.EFFECT_SENSITIVITY_LADDER:
            res = bp.compare_means(x, y, prior=name, seed=1)
            intervals.append(res.interval())
            factors.append(res.log_bf10)
        assert all(iv == pytest.approx(intervals[0]) for iv in intervals)
        assert max(factors) - min(factors) > 0.3

    def test_wider_prior_weakens_a_small_effect(self):
        narrow = nt.log_bayes_factor_ttest(1.2, 40, 39, scale=0.35)
        wide = nt.log_bayes_factor_ttest(1.2, 40, 39, scale=1.5)
        assert narrow > wide

    def test_larger_t_gives_stronger_evidence(self):
        weak = nt.log_bayes_factor_ttest(0.5, 30, 29)
        strong = nt.log_bayes_factor_ttest(3.5, 30, 29)
        assert strong > weak

    def test_sign_of_t_does_not_matter(self):
        assert nt.log_bayes_factor_ttest(2.0, 30, 29) == pytest.approx(
            nt.log_bayes_factor_ttest(-2.0, 30, 29)
        )

    @pytest.mark.parametrize(
        ("t", "n_eff", "df"), [(0.0, 20, 19), (10.0, 20, 19), (2.0, 1000, 999)]
    )
    def test_extremes_stay_finite(self, t, n_eff, df):
        assert np.isfinite(nt.log_bayes_factor_ttest(t, n_eff, df))

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"t": 2.0, "n_effective": 20, "df": 0}, "df must be positive"),
            ({"t": 2.0, "n_effective": 0, "df": 19}, "n_effective must be"),
            ({"t": 2.0, "n_effective": 20, "df": 19, "scale": 0}, "scale must"),
        ],
    )
    def test_invalid_arguments_rejected(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            nt.log_bayes_factor_ttest(**kwargs)


@pytest.mark.validation
class TestAgainstPingouin:
    """Cross-check the JZS integral against an independent implementation."""

    @pytest.mark.parametrize(
        ("t", "n"), [(2.0, 20), (0.5, 30), (3.5, 15), (1.0, 100), (2.5, 200)]
    )
    def test_one_sample(self, t, n):
        pg = pytest.importorskip("pingouin")
        mine = np.exp(nt.log_bayes_factor_ttest(t, n_effective=n, df=n - 1))
        assert mine == pytest.approx(
            float(pg.bayesfactor_ttest(t, n, paired=True)), rel=1e-6
        )

    @pytest.mark.parametrize(
        ("t", "n1", "n2"), [(2.0, 20, 25), (0.8, 50, 45), (3.0, 12, 15)]
    )
    def test_two_sample(self, t, n1, n2):
        pg = pytest.importorskip("pingouin")
        mine = np.exp(
            nt.log_bayes_factor_ttest(
                t, n_effective=n1 * n2 / (n1 + n2), df=n1 + n2 - 2
            )
        )
        assert mine == pytest.approx(float(pg.bayesfactor_ttest(t, n1, n2)), rel=1e-6)

    @pytest.mark.parametrize("scale", [0.5, 0.707, 1.0, 1.4])
    def test_prior_scale(self, scale):
        pg = pytest.importorskip("pingouin")
        mine = np.exp(nt.log_bayes_factor_ttest(2.2, 40, 39, scale=scale))
        assert mine == pytest.approx(
            float(pg.bayesfactor_ttest(2.2, 40, paired=True, r=scale)), rel=1e-6
        )


class TestFrequentistTwins:
    def test_one_sample_matches_scipy(self):
        rng = np.random.default_rng(9)
        x = rng.normal(3, 2, 40)
        twin = bp.frequentist.one_mean(x, mu0=2.5)
        expected = stats.ttest_1samp(x, 2.5)
        assert twin.statistic == pytest.approx(expected.statistic)
        assert twin.pvalue == pytest.approx(expected.pvalue)

    def test_welch_matches_scipy(self):
        rng = np.random.default_rng(10)
        a, b = rng.normal(0, 1, 30), rng.normal(0.5, 3, 40)
        twin = bp.frequentist.two_means(a, b, equal_var=False)
        expected = stats.ttest_ind(b, a, equal_var=False)
        assert twin.statistic == pytest.approx(expected.statistic)
        assert twin.pvalue == pytest.approx(expected.pvalue)

    def test_pooled_matches_scipy(self):
        rng = np.random.default_rng(11)
        a, b = rng.normal(0, 1, 30), rng.normal(0.5, 1, 40)
        twin = bp.frequentist.two_means(a, b, equal_var=True)
        expected = stats.ttest_ind(b, a, equal_var=True)
        assert twin.statistic == pytest.approx(expected.statistic)
        assert twin.pvalue == pytest.approx(expected.pvalue)
