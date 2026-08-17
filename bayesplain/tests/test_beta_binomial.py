"""Correctness tests for the Beta-Binomial core."""

import numpy as np
import pytest
from scipy import integrate, stats

from bayesplain.core import beta_binomial as bb


class TestPosterior:
    def test_conjugate_update_adds_counts(self):
        post = bb.posterior(34, 220, a=1.0, b=1.0)
        assert post.args == (35.0, 187.0)

    def test_posterior_mean_lies_between_prior_mean_and_mle(self):
        # The defining property of a conjugate update: it interpolates.
        prior_mean = bb.prior_predictive_mean(8.0, 2.0)  # 0.8
        mle = 34 / 220  # ~0.155
        post_mean = bb.posterior(34, 220, a=8.0, b=2.0).mean()
        assert mle < post_mean < prior_mean

    @pytest.mark.parametrize("n", [50, 200, 800, 3200])
    def test_interval_width_shrinks_as_sqrt_n(self, n):
        # Width should fall roughly as 1/sqrt(n); check against the previous
        # power of four, which should halve it.
        def width(size):
            post = bb.posterior(int(0.3 * size), size)
            return post.ppf(0.975) - post.ppf(0.025)

        if n == 50:
            pytest.skip("no smaller size to compare against")
        ratio = width(n) / width(n // 4)
        assert 0.45 < ratio < 0.55

    def test_zero_successes_gives_usable_interval(self):
        # A Wald interval collapses to zero width here; the posterior does not.
        post = bb.posterior(0, 30)
        lo, hi = post.ppf(0.025), post.ppf(0.975)
        assert lo >= 0.0
        assert 0.05 < hi < 0.20

    def test_all_successes_is_symmetric_with_none(self):
        low = bb.posterior(0, 30)
        high = bb.posterior(30, 30)
        assert low.mean() == pytest.approx(1.0 - high.mean())


class TestValidation:
    @pytest.mark.parametrize(
        ("successes", "n"),
        [(5, 3), (-1, 10), (2.5, 10), (5, 0), (5, -2), (3, 2.5)],
    )
    def test_impossible_counts_rejected(self, successes, n):
        with pytest.raises(ValueError):
            bb.validate_counts(successes, n)

    def test_rate_instead_of_count_gets_a_pointed_message(self):
        with pytest.raises(ValueError, match="multiply it by"):
            bb.validate_counts(0.15, 220)

    @pytest.mark.parametrize(("a", "b"), [(0, 1), (1, 0), (-1, 2)])
    def test_nonpositive_prior_shapes_rejected(self, a, b):
        with pytest.raises(ValueError, match="positive"):
            bb.posterior(5, 10, a=a, b=b)


class TestMarginalLikelihood:
    def test_matches_numerical_integration(self):
        # The closed form should agree with brute-force integration of the
        # binomial likelihood against the prior.
        x, n, a, b = 34, 220, 2.0, 5.0
        analytic = np.exp(bb.log_marginal_likelihood(x, n, a, b))
        numeric, _ = integrate.quad(
            lambda p: stats.binom.pmf(x, n, p) * stats.beta.pdf(p, a, b),
            0.0,
            1.0,
        )
        assert analytic == pytest.approx(numeric, rel=1e-9)

    def test_sums_to_one_over_all_outcomes(self):
        # A marginal likelihood is a probability mass function over x, so it
        # must sum to 1 across every possible number of successes.
        n, a, b = 25, 1.5, 3.0
        total = sum(
            np.exp(bb.log_marginal_likelihood(x, n, a, b)) for x in range(n + 1)
        )
        assert total == pytest.approx(1.0, abs=1e-12)

    def test_flat_prior_gives_uniform_marginal(self):
        # Under Beta(1,1) every count 0..n is equally likely a priori.
        n = 10
        masses = [np.exp(bb.log_marginal_likelihood(x, n, 1.0, 1.0)) for x in range(11)]
        assert np.allclose(masses, 1.0 / (n + 1))

    def test_large_n_does_not_overflow(self):
        value = bb.log_marginal_likelihood(400_000, 1_000_000, 1.0, 1.0)
        assert np.isfinite(value)


class TestBayesFactor:
    def test_data_at_the_null_favours_the_null(self):
        # Exactly half successes is the null's best case.
        log_bf = bb.log_bayes_factor_point_null(50, 100, p0=0.5)
        assert log_bf < 0

    def test_data_far_from_null_favours_the_alternative(self):
        log_bf = bb.log_bayes_factor_point_null(90, 100, p0=0.5)
        assert log_bf > np.log(100)

    def test_equals_ratio_of_marginal_likelihoods(self):
        x, n, p0, a, b = 34, 220, 0.10, 2.0, 2.0
        expected = bb.log_marginal_likelihood(x, n, a, b) - stats.binom.logpmf(x, n, p0)
        assert bb.log_bayes_factor_point_null(x, n, p0, a, b) == pytest.approx(expected)

    def test_wider_prior_weakens_evidence_for_a_small_effect(self):
        # Lindley's paradox in miniature: spreading prior mass over values the
        # data do not support costs the alternative.
        x, n, p0 = 55, 100, 0.5
        tight = bb.log_bayes_factor_point_null(x, n, p0, a=50.0, b=50.0)
        wide = bb.log_bayes_factor_point_null(x, n, p0, a=1.0, b=1.0)
        assert tight > wide

    @pytest.mark.parametrize("p0", [0.0, 1.0, -0.1, 1.5])
    def test_boundary_null_rejected(self, p0):
        with pytest.raises(ValueError, match="strictly between"):
            bb.log_bayes_factor_point_null(5, 10, p0=p0)


class TestDraws:
    def test_draws_match_the_analytic_posterior(self):
        rng = np.random.default_rng(0)
        samples = bb.draws(34, 220, size=200_000, rng=rng)
        post = bb.posterior(34, 220)
        assert samples.mean() == pytest.approx(post.mean(), abs=1e-3)
        assert samples.std() == pytest.approx(post.std(), abs=1e-3)

    def test_same_seed_reproduces(self):
        first = bb.draws(10, 40, size=1000, rng=np.random.default_rng(7))
        second = bb.draws(10, 40, size=1000, rng=np.random.default_rng(7))
        assert np.array_equal(first, second)
