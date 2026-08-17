"""Tests for the Result object -- the interface every analysis shares."""

import numpy as np
import pytest

import bayesplain as bf
from bayesplain.core import intervals


@pytest.fixture
def evictions():
    return bf.compare_proportions(
        successes=[34, 51], n=[220, 240], labels=["District A", "District B"]
    )


def flat(report) -> str:
    """Collapse whitespace, so prose assertions survive line wrapping.

    Report text is wrapped to terminal width, which can split a phrase across
    two lines. Tests care that the sentence is present, not where it breaks.
    """
    return " ".join(str(report).split())


class TestIntervals:
    def test_hdi_contains_the_requested_mass(self):
        rng = np.random.default_rng(0)
        draws = rng.normal(size=200_000)
        lo, hi = intervals.hdi_from_draws(draws, 0.9)
        assert np.mean((draws >= lo) & (draws <= hi)) == pytest.approx(0.9, abs=0.01)

    def test_hdi_of_a_symmetric_posterior_is_centred(self):
        rng = np.random.default_rng(1)
        draws = rng.normal(size=200_000)
        lo, hi = intervals.hdi_from_draws(draws, 0.95)
        assert lo == pytest.approx(-hi, abs=0.05)

    def test_hdi_is_shorter_than_eti_when_skewed(self):
        rng = np.random.default_rng(2)
        draws = rng.gamma(1.5, size=200_000)
        hdi = intervals.hdi_from_draws(draws, 0.95)
        eti = intervals.eti_from_draws(draws, 0.95)
        assert (hdi[1] - hdi[0]) < (eti[1] - eti[0])

    def test_analytic_eti_used_when_a_posterior_exists(self):
        res = bf.proportion(34, 220)
        lo, hi = res.interval(kind="eti")
        assert lo == pytest.approx(res.posterior.ppf(0.025), abs=1e-12)
        assert hi == pytest.approx(res.posterior.ppf(0.975), abs=1e-12)

    @pytest.mark.parametrize("level", [0, 1, -0.1, 95])
    def test_invalid_level_rejected(self, level):
        with pytest.raises(ValueError, match="between 0 and 1"):
            bf.proportion(34, 220).interval(level=level)

    def test_invalid_kind_rejected(self):
        with pytest.raises(ValueError, match="hdi"):
            bf.proportion(34, 220).interval(kind="quantile")

    def test_wider_level_gives_wider_interval(self):
        res = bf.proportion(34, 220)
        narrow = np.diff(res.interval(level=0.5))[0]
        wide = np.diff(res.interval(level=0.99))[0]
        assert wide > narrow


class TestProbability:
    def test_bad_operator_rejected(self, evictions):
        with pytest.raises(ValueError, match="op must be"):
            evictions.probability("≈", 0)

    def test_between_needs_a_pair(self, evictions):
        with pytest.raises(ValueError, match="needs a"):
            evictions.probability("between", 0.5)

    def test_reversed_bounds_are_tolerated(self, evictions):
        forward = evictions.probability("between", (0.0, 0.1))
        reversed_ = evictions.probability("between", (0.1, 0.0))
        assert forward == pytest.approx(reversed_)

    def test_probability_of_direction_is_at_least_half(self, evictions):
        assert 0.5 <= evictions.probability_of_direction() <= 1.0

    def test_default_value_uses_the_reference(self, evictions):
        assert evictions.probability(">") == evictions.probability(">", 0.0)


class TestDecide:
    def test_wide_rope_gives_practical_equivalence(self, evictions):
        decision = evictions.decide(rope=(-0.5, 0.5))
        assert decision.verdict == "practically equivalent"
        assert decision.probability_inside == pytest.approx(1.0, abs=1e-6)

    def test_narrow_rope_far_from_the_effect_gives_difference(self):
        res = bf.compare_proportions([10, 90], [100, 100])
        decision = res.decide(rope=(-0.01, 0.01))
        assert decision.verdict == "practically different"

    def test_straddling_rope_is_undecided(self, evictions):
        decision = evictions.decide(rope=(-0.02, 0.02))
        assert decision.verdict == "too uncertain to call"

    def test_rope_bounds_may_be_given_in_either_order(self, evictions):
        a = evictions.decide(rope=(-0.02, 0.02))
        b = evictions.decide(rope=(0.02, -0.02))
        assert a.verdict == b.verdict
        assert a.rope == b.rope


class TestBayesFactor:
    def test_reported_as_a_ratio_pair(self, evictions):
        result = evictions.bayes_factor()
        assert result.bf01 == pytest.approx(1.0 / result.bf10)
        assert result.log_bf10 == pytest.approx(np.log(result.bf10))

    def test_carries_a_caveat(self, evictions):
        assert "sensitivity" in flat(evictions.bayes_factor().caveat).lower()

    def test_not_printed_in_the_summary(self, evictions):
        # The design commitment: available, documented, never the headline.
        text = flat(evictions.summary())
        assert "BF10" not in text
        assert "Bayes factor" not in text

    def test_absent_bayes_factor_explains_itself(self, evictions):
        evictions.log_bf10 = None
        with pytest.raises(NotImplementedError, match="estimation question"):
            evictions.bayes_factor()


class TestSensitivity:
    def test_walks_the_whole_ladder(self, evictions):
        text = flat(evictions.sensitivity())
        for name in bf.priors.SENSITIVITY_LADDER:
            assert name in text

    def test_accepts_explicit_priors(self, evictions):
        text = flat(evictions.sensitivity(priors=[(1, 1), (5, 5)]))
        assert "custom" in text

    def test_reports_that_the_estimate_is_robust_here(self, evictions):
        # 460 observations: any reasonable prior gives the same answer.
        assert "estimate barely moves" in flat(evictions.sensitivity())

    def test_flags_that_the_bayes_factor_is_not(self, evictions):
        # ... while the Bayes factor over the same priors moves a lot. This
        # asymmetry is the reason the package leads with estimation.
        assert "Bayes factor is a different story" in flat(evictions.sensitivity())

    def test_small_samples_are_flagged_as_prior_dependent(self):
        res = bf.proportion(3, 6, reference=0.5)
        text = flat(res.sensitivity())
        assert "estimate barely moves" not in text

    def test_empty_prior_list_rejected(self, evictions):
        with pytest.raises(ValueError, match="no priors"):
            evictions.sensitivity(priors=[])


class TestLanguage:
    def test_sentence_names_both_groups(self, evictions):
        sentence = evictions.sentence()
        assert "District A" in sentence and "District B" in sentence

    def test_sentence_names_the_higher_group_first(self, evictions):
        assert evictions.sentence().startswith("District B is higher")

    def test_sentence_flips_when_the_data_flip(self):
        res = bf.compare_proportions(
            [51, 34], [240, 220], labels=["District A", "District B"]
        )
        assert res.sentence().startswith("District A is higher")

    def test_translate_states_both_claims(self, evictions):
        text = flat(evictions.translate())
        assert "What the posterior says" in text
        assert "not the probability that the null is true" in text

    def test_translate_names_the_ci_misreading(self, evictions):
        assert "only available for a credible interval" in flat(evictions.translate())


class TestSummary:
    def test_posterior_comes_before_the_frequentist_twin(self, evictions):
        text = flat(evictions.summary())
        assert text.index("BAYESIAN") < text.index("FREQUENTIST")

    def test_every_line_fits_in_a_terminal(self, evictions):
        for line in str(evictions.summary()).splitlines():
            assert len(line) <= 78, f"line too wide ({len(line)}): {line!r}"

    def test_sensitivity_lines_also_fit(self, evictions):
        for line in str(evictions.sensitivity()).splitlines():
            assert len(line) <= 78, f"line too wide ({len(line)}): {line!r}"

    def test_reports_monte_carlo_error_when_sampled(self, evictions):
        assert "Monte Carlo error" in flat(evictions.summary())

    def test_says_so_when_exact(self):
        text = flat(bf.proportion(34, 220).summary())
        assert "exact" in text
        assert "Monte Carlo error" not in text

    def test_states_the_prior_it_used(self, evictions):
        assert "Beta(1, 1)" in flat(evictions.summary())

    def test_unit_appears_once_per_interval(self, evictions):
        line = next(
            line
            for line in str(evictions.summary()).splitlines()
            if "credible interval" in line
        )
        assert line.count("percentage points") == 1

    def test_repr_is_short_and_points_at_summary(self, evictions):
        text = repr(evictions)
        assert len(text.splitlines()) == 2
        assert ".summary()" in text


class TestToDict:
    def test_carries_the_headline_numbers(self, evictions):
        out = evictions.to_dict()
        for key in (
            "point_median",
            "interval_low",
            "interval_high",
            "probability_of_direction",
            "p_value",
            "bf10",
            "prior",
        ):
            assert key in out

    def test_values_are_plain_floats(self, evictions):
        out = evictions.to_dict()
        assert isinstance(out["interval_low"], float)
        assert isinstance(out["point_median"], float)

    def test_round_trips_through_json(self, evictions):
        import json

        text = json.dumps(evictions.to_dict())
        assert json.loads(text)["analysis"] == "compare_proportions"


class TestPedagogy:
    """Tests that pin numbers a lecture depends on.

    If a refactor breaks a lecture, CI should say so rather than a student.
    """

    @pytest.mark.pedagogy
    def test_week4_eviction_example_holds(self, evictions):
        # The week 4 pivot: the Bayesian answer is actionable, the frequentist
        # answer is "not significant", and both are computed from the same
        # 34/220 vs 51/240.
        assert evictions.probability(">", 0) == pytest.approx(0.94, abs=0.01)
        assert evictions.frequentist.pvalue > 0.05
        assert not evictions.frequentist.significant

    @pytest.mark.pedagogy
    def test_week4_intervals_nearly_coincide(self, evictions):
        # The numbers agree; only the sentences differ. That is the lecture.
        lo, hi = evictions.interval(kind="eti")
        flo, fhi = evictions.frequentist.interval
        assert lo == pytest.approx(flo, abs=0.01)
        assert hi == pytest.approx(fhi, abs=0.01)

    @pytest.mark.pedagogy
    def test_week4_summary_explains_the_apparent_disagreement(self, evictions):
        assert "Why they look like they disagree" in flat(evictions.summary())

    @pytest.mark.pedagogy
    def test_week7_small_cells_degrade_gracefully(self):
        # chi-square warns here; the exact posterior does not need to.
        res = bf.compare_proportions([1, 8], [30, 32])
        lo, hi = res.interval()
        assert np.isfinite([lo, hi]).all()
        assert any("fewer than 5" in note for note in res.notes)

    @pytest.mark.pedagogy
    def test_week3_prior_stops_mattering_as_n_grows(self):
        def spread(n):
            successes = int(0.3 * n)
            points = [
                bf.proportion(successes, n, prior=p).point()
                for p in bf.priors.SENSITIVITY_LADDER
            ]
            return max(points) - min(points)

        assert spread(20) > 5 * spread(2000)
