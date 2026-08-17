"""Tests for the public proportion analyses."""

import numpy as np
import pytest

import bayesplain as bf


class TestProportion:
    def test_exact_posterior_is_available(self):
        res = bf.proportion(34, 220)
        assert res.exact
        assert res.monte_carlo_error() == 0.0

    def test_point_estimate_matches_conjugate_update(self):
        res = bf.proportion(34, 220)
        assert res.point("mean") == pytest.approx(35 / 222)

    def test_credible_interval_close_to_wilson_interval(self):
        # Worth pinning: the two frameworks usually produce nearly the same
        # numbers here while licensing very different sentences.
        res = bf.proportion(34, 220, reference=0.10)
        lo, hi = res.interval(kind="eti")
        wlo, whi = res.frequentist.interval
        assert lo == pytest.approx(wlo, abs=0.01)
        assert hi == pytest.approx(whi, abs=0.01)

    def test_hdi_is_never_wider_than_eti(self):
        res = bf.proportion(3, 40)
        hdi = res.interval(kind="hdi")
        eti = res.interval(kind="eti")
        assert (hdi[1] - hdi[0]) <= (eti[1] - eti[0]) + 1e-9

    def test_threshold_probability_is_analytic(self):
        res = bf.proportion(34, 220, reference=0.10)
        assert res.probability(">", 0.10) == pytest.approx(0.9955, abs=1e-3)

    def test_probability_directions_are_complementary(self):
        res = bf.proportion(34, 220)
        assert res.probability(">", 0.15) + res.probability("<=", 0.15) == (
            pytest.approx(1.0)
        )

    def test_between_and_outside_are_complementary(self):
        res = bf.proportion(34, 220)
        inside = res.probability("between", (0.12, 0.20))
        outside = res.probability("outside", (0.12, 0.20))
        assert inside + outside == pytest.approx(1.0)

    def test_small_sample_gets_a_note(self):
        res = bf.proportion(4, 12)
        assert any("prior is doing visible work" in note for note in res.notes)

    def test_degenerate_data_gets_a_note_and_still_works(self):
        res = bf.proportion(0, 25, reference=0.1)
        assert any("all 25 observations" in note for note in res.notes)
        lo, hi = res.interval()
        assert hi > lo >= 0.0

    def test_reference_outside_unit_interval_rejected(self):
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            bf.proportion(34, 220, reference=1.5)


class TestCompareProportions:
    @pytest.fixture
    def evictions(self):
        return bf.compare_proportions(
            successes=[34, 51], n=[220, 240], labels=["District A", "District B"]
        )

    def test_no_analytic_posterior_for_a_difference(self, evictions):
        # The difference of two Betas has no tidy closed form, so this one is
        # sampled -- and says so.
        assert not evictions.exact
        assert evictions.monte_carlo_error() > 0

    def test_difference_matches_the_observed_gap(self, evictions):
        observed = 51 / 240 - 34 / 220
        assert evictions.point("mean") == pytest.approx(observed, abs=0.005)

    def test_group_order_only_flips_the_sign(self):
        # Tolerances here are Monte Carlo noise, not slack: the two calls draw
        # different samples because each group consumes the generator in the
        # order it appears, so agreement is only to within ~2 MCSE.
        forward = bf.compare_proportions([34, 51], [220, 240], seed=1)
        reverse = bf.compare_proportions([51, 34], [240, 220], seed=1)
        assert forward.point("mean") == pytest.approx(-reverse.point("mean"), abs=5e-3)
        assert forward.probability(">", 0) == pytest.approx(
            1.0 - reverse.probability(">", 0), abs=5e-3
        )

    def test_bayes_factor_is_invariant_to_group_order(self):
        forward = bf.compare_proportions([34, 51], [220, 240])
        reverse = bf.compare_proportions([51, 34], [240, 220])
        assert forward.log_bf10 == pytest.approx(reverse.log_bf10)

    def test_identical_groups_centre_on_zero(self):
        res = bf.compare_proportions([50, 50], [200, 200])
        assert res.point("mean") == pytest.approx(0.0, abs=0.005)
        assert res.probability(">", 0) == pytest.approx(0.5, abs=0.01)

    def test_more_data_narrows_the_interval(self):
        small = bf.compare_proportions([17, 25], [110, 120])
        large = bf.compare_proportions([170, 250], [1100, 1200])
        small_width = np.diff(small.interval())[0]
        large_width = np.diff(large.interval())[0]
        assert large_width < small_width / 2

    @pytest.mark.parametrize("estimand", ["difference", "risk_ratio", "odds_ratio"])
    def test_every_estimand_agrees_on_direction(self, estimand):
        res = bf.compare_proportions([34, 51], [220, 240], estimand=estimand)
        # Whatever the scale, group 2 should be the higher one with the same
        # probability of direction.
        assert res.probability(">", res.direction_reference) > 0.9

    def test_ratio_estimands_reference_one_not_zero(self):
        res = bf.compare_proportions([34, 51], [220, 240], estimand="risk_ratio")
        assert res.direction_reference == 1.0
        assert res.point() == pytest.approx((51 / 240) / (34 / 220), abs=0.05)

    def test_wrong_number_of_groups_points_to_the_right_function(self):
        with pytest.raises(ValueError, match="compare_groups"):
            bf.compare_proportions([1, 2, 3], [10, 10, 10])

    def test_bad_estimand_rejected(self):
        with pytest.raises(ValueError, match="estimand must be"):
            bf.compare_proportions([1, 2], [10, 10], estimand="ratio")

    def test_wrong_number_of_labels_rejected(self):
        with pytest.raises(ValueError, match="exactly two names"):
            bf.compare_proportions([1, 2], [10, 10], labels=["only one"])

    def test_sparse_cells_get_a_note(self):
        res = bf.compare_proportions([1, 2], [40, 45])
        assert any("fewer than 5 cases" in note for note in res.notes)


class TestReproducibility:
    def test_default_seed_gives_identical_digits(self):
        first = bf.compare_proportions([34, 51], [220, 240])
        second = bf.compare_proportions([34, 51], [220, 240])
        assert np.array_equal(first.draws, second.draws)

    def test_seed_none_gives_different_draws(self):
        first = bf.compare_proportions([34, 51], [220, 240], seed=None)
        second = bf.compare_proportions([34, 51], [220, 240], seed=None)
        assert not np.array_equal(first.draws, second.draws)

    def test_package_seed_is_honoured(self):
        original = bf.get_seed()
        try:
            bf.set_seed(999)
            a = bf.compare_proportions([34, 51], [220, 240])
            bf.set_seed(999)
            b = bf.compare_proportions([34, 51], [220, 240])
            assert np.array_equal(a.draws, b.draws)
        finally:
            bf.set_seed(original)

    def test_too_few_draws_rejected(self):
        with pytest.raises(ValueError, match="at least 1000"):
            bf.set_draws(10)


class TestPriors:
    def test_preset_names_resolve(self):
        for name in bf.priors.available():
            assert bf.priors.resolve_proportion(name).name == name

    def test_tuple_resolves_to_custom(self):
        prior = bf.priors.resolve_proportion((3, 4))
        assert (prior.a, prior.b) == (3.0, 4.0)
        assert prior.name == "custom"

    def test_unknown_name_lists_the_options(self):
        with pytest.raises(ValueError, match="Available presets"):
            bf.priors.resolve_proportion("vague")

    def test_previous_study_prior_carries_the_right_weight(self):
        prior = bf.priors.from_previous_study(successes=12, n=80)
        assert prior.prior_weight == pytest.approx(82.0)
        assert prior.prior_mean == pytest.approx(13 / 82)

    def test_previous_study_prior_shifts_the_posterior_toward_it(self):
        flat = bf.proportion(34, 220, prior="uninformed")
        informed = bf.proportion(
            34, 220, prior=bf.priors.from_previous_study(successes=2, n=80)
        )
        assert informed.point() < flat.point()

    def test_stronger_prior_narrows_the_interval(self):
        weak = np.diff(bf.proportion(8, 20, prior="uninformed").interval())[0]
        strong = np.diff(bf.proportion(8, 20, prior="skeptical").interval())[0]
        assert strong < weak

    def test_describe_returns_text_for_every_preset(self):
        text = bf.priors.describe()
        for name in bf.priors.available():
            assert name in text
