"""Tests for rates: Gamma-Poisson posteriors, the rate ratio, and their twins."""

import numpy as np
import pytest
from scipy import integrate, stats

import bayesplain as bp
from bayesplain.core import beta_binomial as bb
from bayesplain.core import gamma_poisson as gp


class TestCore:
    def test_posterior_is_the_conjugate_update(self):
        post = gp.posterior(12, 4.0, a=0.5, b=0.0)
        assert post.mean() == pytest.approx(12.5 / 4.0)

    def test_ratio_posterior_matches_sampling_two_gammas(self):
        rng = np.random.default_rng(1)
        exact = gp.ratio_posterior(7, 2.0, 30, 5.0, a=0.5)
        g1 = stats.gamma(7.5, scale=1 / 2.0).rvs(400_000, random_state=rng)
        g2 = stats.gamma(30.5, scale=1 / 5.0).rvs(400_000, random_state=rng)
        assert exact.median() == pytest.approx(np.median(g2 / g1), rel=0.01)
        assert exact.ppf(0.975) == pytest.approx(np.quantile(g2 / g1, 0.975), rel=0.02)

    def test_ratio_posterior_with_a_proper_prior(self):
        rng = np.random.default_rng(2)
        exact = gp.ratio_posterior(7, 2.0, 30, 5.0, a=2.0, b=1.0)
        g1 = stats.gamma(9.0, scale=1 / 3.0).rvs(400_000, random_state=rng)
        g2 = stats.gamma(32.0, scale=1 / 6.0).rvs(400_000, random_state=rng)
        assert exact.median() == pytest.approx(np.median(g2 / g1), rel=0.01)

    @pytest.mark.parametrize("a", [0.5, 1.0, 2.0])
    def test_equal_exposure_reduces_to_beta_binomial(self, a):
        # Conditional on the total, the second group's share is binomial with
        # a Beta(a, a) prior when exposures are equal.
        got = gp.log_bayes_factor_equal_rates(18, 3.0, 35, 3.0, a=a)
        want = bb.log_bayes_factor_point_null(35, 53, 0.5, a, a)
        assert got == pytest.approx(want, abs=1e-8)

    def test_equal_rates_bf_matches_direct_integration(self):
        y1, t1, y2, t2, a = 7, 2.0, 30, 5.0, 1.0
        c = t2 / t1
        prior = stats.betaprime(a, a)

        def integrand(ratio):
            share = c * ratio / (1 + c * ratio)
            return stats.binom.pmf(y2, y1 + y2, share) * prior.pdf(ratio)

        num, _ = integrate.quad(integrand, 0, np.inf, limit=500)
        want = np.log(num / stats.binom.pmf(y2, y1 + y2, c / (1 + c)))
        assert gp.log_bayes_factor_equal_rates(y1, t1, y2, t2, a) == pytest.approx(
            want, abs=1e-6
        )

    def test_point_null_bf_matches_direct_integration(self):
        y, t, rate0, a = 12, 4.0, 2.0, 1.0
        prior = stats.betaprime(a, a)
        num, _ = integrate.quad(
            lambda r: stats.poisson.pmf(y, rate0 * r * t) * prior.pdf(r),
            0,
            np.inf,
            limit=500,
        )
        want = np.log(num / stats.poisson.pmf(y, rate0 * t))
        assert gp.log_bayes_factor_point_null(y, t, rate0, a) == pytest.approx(
            want, abs=1e-6
        )

    def test_proper_prior_bf_is_the_negative_binomial_ratio(self):
        y, t, rate0 = 12, 4.0, 2.0
        num, _ = integrate.quad(
            lambda lam: (
                stats.poisson.pmf(y, lam * t) * stats.gamma(2, scale=1 / 3).pdf(lam)
            ),
            0,
            np.inf,
        )
        want = np.log(num / stats.poisson.pmf(y, rate0 * t))
        got = gp.log_bayes_factor_point_null(y, t, rate0, a=2.0, b=3.0)
        assert got == pytest.approx(want, abs=1e-8)

    def test_bayes_factors_stay_finite_for_large_counts(self):
        assert np.isfinite(gp.log_bayes_factor_equal_rates(5000, 100, 5600, 100))
        assert np.isfinite(gp.log_bayes_factor_point_null(20_000, 10, 1900))

    def test_predictive_is_a_gamma_mixture_of_poissons(self):
        rng = np.random.default_rng(3)
        lam = stats.gamma(12.5, scale=1 / 4.0).rvs(400_000, random_state=rng)
        sims = rng.poisson(lam * 2.0)
        forecast = gp.predictive(12, 4.0, new_exposure=2.0)
        assert forecast.mean() == pytest.approx(sims.mean(), rel=0.01)
        assert forecast.var() == pytest.approx(sims.var(), rel=0.02)

    @pytest.mark.parametrize(
        "events, exposure, match",
        [
            (-1, 1.0, "non-negative"),
            (2.5, 1.0, "whole number"),
            (3, 0.0, "exposure must be positive"),
        ],
    )
    def test_bad_inputs_are_explained(self, events, exposure, match):
        with pytest.raises(ValueError, match=match):
            gp.validate_events(events, exposure)


class TestRate:
    @pytest.fixture
    def crossing(self):
        return bp.rate(14, exposure=4, reference=2.0, unit="per year")

    def test_posterior_is_exact(self, crossing):
        assert crossing.exact
        assert crossing.monte_carlo_error() == 0.0

    def test_jeffreys_interval_sits_inside_the_exact_interval(self, crossing):
        lo, hi = crossing.interval(kind="eti")
        flo, fhi = crossing.frequentist.interval
        assert flo <= lo and hi <= fhi

    def test_exact_poisson_p_value(self, crossing):
        null = stats.poisson(8.0)
        want = 2 * null.sf(13)
        assert crossing.frequentist.pvalue == pytest.approx(want)

    def test_reference_is_required(self):
        with pytest.raises(ValueError, match="needs a reference rate"):
            bp.rate(14, exposure=4)

    def test_per_period_counts_are_summed(self):
        res = bp.rate([3, 4, 5, 2], exposure=1.0, reference=3.0)
        assert res.posterior.mean() == pytest.approx((14 + 0.5) / 4)

    def test_overdispersed_periods_get_a_note(self):
        res = bp.rate([3, 9, 1, 12, 2], 1.0, reference=5.0)
        assert any("vary between periods" in note for note in res.notes)

    def test_steady_periods_get_no_dispersion_note(self):
        res = bp.rate([5, 6, 5, 4, 6], 1.0, reference=5.0)
        assert not any("vary between periods" in note for note in res.notes)

    def test_rare_events_get_a_note(self):
        res = bp.rate(3, exposure=10, reference=0.5)
        assert any("only 3 events" in note for note in res.notes)

    def test_prior_from_a_previous_period(self):
        prior = bp.priors.from_previous_period(events=30, exposure=5.0)
        res = bp.rate(14, exposure=4, reference=2.0, prior=prior)
        assert res.posterior.mean() == pytest.approx((30.5 + 14) / 9.0)
        assert "prior_posterior" in res.plot_kinds()

    def test_sensitivity_runs(self, crossing):
        report = str(crossing.sensitivity())
        assert "jeffreys" in report and "uninformed" in report

    def test_forecast_is_wider_than_plug_in_with_little_data(self):
        forecast = bp.rate(3, exposure=1, reference=2.0).predict(exposure=5)
        lo, hi = forecast.interval()
        plo, phi = forecast._interval_of(forecast.plug_in, 0.95)
        assert hi - lo > phi - plo


class TestCompareRates:
    @pytest.fixture
    def road_diet(self):
        return bp.compare_rates(
            events=[31, 9], exposure=[4, 2], labels=["before", "after"]
        )

    def test_ratio_is_exact(self, road_diet):
        assert road_diet.exact
        assert road_diet.direction_reference == 1.0

    def test_ratio_centres_on_the_observed_ratio(self, road_diet):
        assert road_diet.point() == pytest.approx((9 / 2) / (31 / 4), rel=0.05)

    def test_twin_is_the_exact_conditional_test(self, road_diet):
        want = stats.binomtest(9, 40, 2 / 6).pvalue
        assert road_diet.frequentist.pvalue == pytest.approx(want)

    def test_credible_and_confidence_intervals_roughly_agree(self, road_diet):
        lo, hi = road_diet.interval(kind="eti")
        flo, fhi = road_diet.frequentist.interval
        assert lo == pytest.approx(flo, rel=0.15)
        assert hi == pytest.approx(fhi, rel=0.15)

    def test_sentence_calls_a_ratio_a_ratio(self, road_diet):
        assert "the ratio is most likely" in road_diet.sentence()

    def test_difference_estimand_is_sampled(self):
        res = bp.compare_rates([31, 9], [4, 2], estimand="difference")
        assert not res.exact
        assert res.point() == pytest.approx(9 / 2 - 31 / 4, abs=0.3)

    def test_no_events_anywhere_is_rejected(self):
        with pytest.raises(ValueError, match="no events in either group"):
            bp.compare_rates([0, 0], [1, 1])

    def test_zero_events_in_one_group_still_works(self):
        res = bp.compare_rates([0, 8], [2, 2])
        lo, hi = res.interval()
        assert 0 < lo < hi

    def test_predict_points_elsewhere(self, road_diet):
        with pytest.raises(NotImplementedError, match="rate\\(\\) on each group"):
            road_diet.predict()
