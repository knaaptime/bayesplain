"""Tests for the teaching-only helpers.

These matter more than their size suggests: every one of them produces a
number that gets written on a board in front of a class, so the properties
asserted here are the ones a lecture would be wrong without.
"""

import numpy as np
import pytest

import bayesplain as bp


def flat(report) -> str:
    """Collapse whitespace, so prose assertions survive line wrapping."""
    return " ".join(str(report).split())


class TestNaturalFrequencies:
    @pytest.fixture
    def grid(self):
        return bp.teach.natural_frequencies(
            base_rate=0.05,
            sensitivity=0.90,
            specificity=0.90,
            n=1000,
            case_label="permits",
            condition_label="a violation",
        )

    def test_every_cell_is_a_whole_number(self, grid):
        assert grid.table.dtype.kind in "iu"
        assert all(float(v).is_integer() for v in grid.table.ravel())

    def test_rows_and_columns_sum_exactly(self, grid):
        assert grid.table.sum() == grid.n
        assert grid.with_condition == grid.true_positives + grid.false_negatives
        assert grid.flagged == grid.true_positives + grid.false_positives

    def test_matches_bayes_rule(self, grid):
        # The whole point: counting and the formula are the same operation.
        assert grid.posterior_given_flag == pytest.approx(
            grid.bayes_rule_check(), abs=0.005
        )

    @pytest.mark.pedagogy
    def test_the_base_rate_lesson_actually_lands(self, grid):
        # 90% accurate both ways, and still wrong about two thirds of the time
        # it fires. If this stops being true the week 3 lecture is wrong.
        assert grid.flagged == 140
        assert grid.posterior_given_flag == pytest.approx(0.321, abs=0.001)
        assert grid.posterior_given_flag < 0.5

    @pytest.mark.pedagogy
    def test_raising_the_base_rate_rescues_the_same_detector(self, grid):
        richer = bp.teach.natural_frequencies(0.16, 0.90, 0.90, 1000)
        assert richer.posterior_given_flag > 0.6
        assert richer.sensitivity == grid.sensitivity
        assert richer.specificity == grid.specificity

    def test_summary_prints_no_probability_in_the_table(self, grid):
        table = str(grid.summary()).split("Of the")[0]
        assert "%" not in table
        assert "0." not in table

    def test_summary_fits_a_terminal(self, grid):
        for line in str(grid.summary()).splitlines():
            assert len(line) <= 78, f"too wide ({len(line)}): {line!r}"

    def test_long_labels_do_not_break_the_layout(self):
        wide = bp.teach.natural_frequencies(
            0.05,
            0.9,
            0.9,
            1000,
            case_label="households",
            condition_label="eligibility for the assistance program",
            flag_label="screened in",
        )
        for line in str(wide.summary()).splitlines():
            assert len(line) <= 78, f"too wide ({len(line)}): {line!r}"

    def test_moral_changes_with_the_numbers(self, grid):
        strong = bp.teach.natural_frequencies(0.5, 0.95, 0.95, 1000)
        assert "more likely than not that the case is fine" in flat(grid.summary())
        assert "strong evidence" in flat(strong.summary())

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            (
                {"base_rate": 0, "sensitivity": 0.9, "specificity": 0.9},
                "between 0 and 1",
            ),
            ({"base_rate": 1.0, "sensitivity": 0.9, "specificity": 0.9}, "below 1"),
            (
                {"base_rate": 90, "sensitivity": 0.9, "specificity": 0.9},
                "0.9 rather than 90",
            ),
            (
                {"base_rate": 0.1, "sensitivity": 1.5, "specificity": 0.9},
                "between 0 and 1",
            ),
            (
                {"base_rate": 0.1, "sensitivity": 0.9, "specificity": 0},
                "between 0 and 1",
            ),
        ],
    )
    def test_invalid_inputs_rejected(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            bp.teach.natural_frequencies(**kwargs)

    def test_tiny_n_rejected(self):
        with pytest.raises(ValueError, match="at least 100"):
            bp.teach.natural_frequencies(0.1, 0.9, 0.9, n=10)


class TestGridPosterior:
    @pytest.fixture
    def four(self):
        candidates = [0.1, 0.2, 0.3, 0.4]
        return bp.teach.grid_posterior(
            grid=candidates,
            likelihood=bp.teach.binomial_likelihood(7, 20, candidates),
        )

    def test_discrete_posterior_sums_to_one(self, four):
        assert four.posterior.sum() == pytest.approx(1.0)

    def test_posterior_is_prior_times_likelihood_normalised(self, four):
        expected = four.prior * four.likelihood
        assert np.allclose(four.posterior, expected / expected.sum())

    def test_flat_prior_leaves_the_likelihood_shape_alone(self, four):
        assert np.allclose(np.argsort(four.posterior), np.argsort(four.likelihood))

    def test_prior_is_normalised_even_if_given_unnormalised(self):
        candidates = [0.2, 0.4]
        post = bp.teach.grid_posterior(
            grid=candidates,
            prior=[3, 7],
            likelihood=bp.teach.binomial_likelihood(1, 2, candidates),
        )
        assert post.prior.sum() == pytest.approx(1.0)
        assert post.prior[0] == pytest.approx(0.3)

    @pytest.mark.pedagogy
    def test_a_fine_grid_reproduces_the_closed_form(self):
        # The payoff of the week 3 exercise: the mechanism students run by hand
        # and the one-liner they use afterwards agree.
        fine = np.linspace(0.0005, 0.9995, 2000)
        post = bp.teach.grid_posterior(
            grid=fine,
            likelihood=bp.teach.binomial_likelihood(7, 20, fine),
            kind="continuous",
        )
        exact = bp.proportion(7, 20)
        assert post.mean() == pytest.approx(exact.point("mean"), abs=1e-4)
        lo, hi = post.interval()
        elo, ehi = exact.interval(kind="eti")
        assert lo == pytest.approx(elo, abs=2e-3)
        assert hi == pytest.approx(ehi, abs=2e-3)

    def test_continuous_posterior_integrates_to_one(self):
        fine = np.linspace(0.001, 0.999, 500)
        post = bp.teach.grid_posterior(
            grid=fine,
            likelihood=bp.teach.binomial_likelihood(3, 10, fine),
            kind="continuous",
        )
        assert np.trapezoid(post.posterior, post.grid) == pytest.approx(1.0, abs=1e-6)

    def test_informative_prior_pulls_the_posterior(self):
        candidates = np.linspace(0.05, 0.95, 19)
        lik = bp.teach.binomial_likelihood(9, 10, candidates)
        flat = bp.teach.grid_posterior(grid=candidates, likelihood=lik)
        low = bp.teach.grid_posterior(
            grid=candidates, prior=np.exp(-8 * candidates), likelihood=lik
        )
        assert low.mean() < flat.mean()

    def test_table_shows_the_normalising_constant(self, four):
        text = " ".join(str(four.table()).split())
        assert "divided by its own total" in text

    def test_table_summarises_a_fine_grid_rather_than_dumping_it(self):
        fine = np.linspace(0.01, 0.99, 500)
        post = bp.teach.grid_posterior(
            grid=fine, likelihood=bp.teach.binomial_likelihood(5, 10, fine)
        )
        assert len(str(post.table()).splitlines()) < 40
        assert "grid points in all" in flat(post.table())

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"grid": [0.5], "likelihood": [1.0]}, "at least 2"),
            ({"grid": [0.5, 0.2], "likelihood": [1.0, 1.0]}, "strictly increasing"),
            ({"grid": [0.2, 0.5], "likelihood": [1.0]}, "line up one to one"),
            ({"grid": [0.2, 0.5], "likelihood": [-1.0, 1.0]}, "cannot be negative"),
            ({"grid": [0.2, 0.5], "likelihood": [0.0, 0.0]}, "nothing to normalise"),
            ({"grid": [0.2, 0.5]}, "likelihood is required"),
        ],
    )
    def test_invalid_inputs_rejected(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            bp.teach.grid_posterior(**kwargs)

    def test_bad_kind_rejected(self):
        with pytest.raises(ValueError, match="discrete"):
            bp.teach.grid_posterior(
                grid=[0.2, 0.5], likelihood=[1.0, 1.0], kind="smooth"
            )

    def test_binomial_likelihood_rejects_rates_outside_the_unit_interval(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            bp.teach.binomial_likelihood(2, 5, [0.5, 1.5])


class TestSequential:
    @pytest.fixture
    def run(self):
        rng = np.random.default_rng(0)
        return bp.teach.sequential(rng.binomial(1, 0.3, 200), step=20)

    def test_one_posterior_per_chunk(self, run):
        assert len(run.posteriors) == 10
        assert run.checkpoints[-1] == 200

    def test_interval_narrows_monotonically_in_the_long_run(self, run):
        assert run.widths[-1] < run.widths[0] / 2

    def test_estimate_converges_toward_the_truth(self, run):
        assert run.means[-1] == pytest.approx(0.3, abs=0.08)

    @pytest.mark.pedagogy
    def test_precision_improves_like_the_square_root(self, run):
        # Quadruple the data, halve the width -- the claim the table makes.
        i, j = 1, 7  # 40 observations vs 160
        assert run.checkpoints[j] == 4 * run.checkpoints[i]
        assert run.widths[j] == pytest.approx(run.widths[i] / 2, rel=0.25)

    def test_a_ragged_tail_still_gets_a_checkpoint(self):
        run = bp.teach.sequential([1, 0] * 11, step=10)  # 22 observations
        assert run.checkpoints.tolist() == [10, 20, 22]

    def test_booleans_are_accepted(self):
        pytest.importorskip("pandas")
        df = bp.datasets.load_collisions()
        run = bp.teach.sequential(df.alcohol_involved.head(100), step=25)
        assert len(run.posteriors) == 4

    def test_prior_shifts_the_early_estimates_most(self):
        outcomes = [1] * 5 + [0] * 5
        flat = bp.teach.sequential(outcomes, step=5, prior="uninformed")
        firm = bp.teach.sequential(outcomes, step=5, prior="skeptical")
        assert abs(firm.means[0] - 0.5) < abs(flat.means[0] - 0.5)

    def test_table_and_repr_render(self, run):
        assert "THE POSTERIOR AFTER EACH CHUNK" in flat(run.table())
        assert "SequentialUpdate" in repr(run)

    @pytest.mark.parametrize(
        ("outcomes", "match"),
        [
            ([1], "at least 2"),
            ([0, 1, 2], "0/1 or boolean"),
            ([0.5, 0.5], "0/1 or boolean"),
        ],
    )
    def test_invalid_outcomes_rejected(self, outcomes, match):
        with pytest.raises(ValueError, match=match):
            bp.teach.sequential(outcomes)

    def test_bad_step_rejected(self):
        with pytest.raises(ValueError, match="step must be at least 1"):
            bp.teach.sequential([0, 1, 0, 1], step=0)


class TestPrecisionPlanning:
    def test_the_returned_n_actually_meets_the_target(self):
        plan = bp.teach.precision_planning(0.10, 0.15)
        post = bp.core.beta_binomial.posterior(
            round(plan.required_n * 0.15), plan.required_n
        )
        assert post.ppf(0.975) - post.ppf(0.025) <= 0.10

    def test_one_fewer_observation_would_not(self):
        plan = bp.teach.precision_planning(0.10, 0.15)
        post = bp.core.beta_binomial.posterior(
            round((plan.required_n - 1) * 0.15), plan.required_n - 1
        )
        assert post.ppf(0.975) - post.ppf(0.025) > 0.10

    def test_a_narrower_target_needs_more_data(self):
        wide = bp.teach.precision_planning(0.10, 0.3)
        narrow = bp.teach.precision_planning(0.05, 0.3)
        assert narrow.required_n > 3 * wide.required_n

    def test_rates_near_the_middle_are_hardest(self):
        middle = bp.teach.precision_planning(0.05, 0.5)
        edge = bp.teach.precision_planning(0.05, 0.05)
        assert middle.required_n > edge.required_n

    @pytest.mark.pedagogy
    def test_agreement_beats_strength(self):
        # The lesson: what helps is a prior that agrees with what you find,
        # not merely a confident one. An equally firm prior in the wrong place
        # costs you data.
        flat = bp.teach.precision_planning(0.10, 0.15, prior="uninformed")
        agrees = bp.teach.precision_planning(
            0.10, 0.15, prior=bp.priors.from_previous_study(15, 100)
        )
        disagrees = bp.teach.precision_planning(0.10, 0.15, prior="skeptical")
        assert agrees.required_n < flat.required_n
        assert disagrees.required_n > flat.required_n

    def test_summary_renders_and_fits(self):
        plan = bp.teach.precision_planning(0.08, 0.2)
        for line in str(plan.summary()).splitlines():
            assert len(line) <= 78, f"too wide ({len(line)}): {line!r}"

    def test_impossible_target_is_rejected_with_a_reason(self):
        with pytest.raises(ValueError, match="too demanding"):
            bp.teach.precision_planning(0.0005, 0.5, max_n=5000)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"target_width": 0}, "between 0 and 1"),
            ({"target_width": 1.5}, "between 0 and 1"),
            ({"target_width": 0.1, "expected_rate": 1.5}, "between 0 and 1"),
        ],
    )
    def test_invalid_inputs_rejected(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            bp.teach.precision_planning(**kwargs)


class TestPlots:
    def test_every_teaching_object_can_draw_itself(self):
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg")

        candidates = [0.1, 0.2, 0.3, 0.4]
        post = bp.teach.grid_posterior(
            grid=candidates,
            likelihood=bp.teach.binomial_likelihood(7, 20, candidates),
        )
        assert post.plot() is not None
        rng = np.random.default_rng(0)
        assert bp.teach.sequential(rng.binomial(1, 0.4, 100)).plot() is not None
