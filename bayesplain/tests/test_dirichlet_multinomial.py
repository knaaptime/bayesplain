"""Correctness tests for the Gunel-Dickey contingency-table core.

Three independent lines of evidence:

1. **Golden masters from Jamil et al. (2017).** The paper reports Bayes factors
   for worked examples under each sampling scheme; those numbers are asserted
   here directly.
2. **Brute-force integration.** For a two-column table each marginal likelihood
   is a one-dimensional integral that ``scipy.integrate.quad`` evaluates to
   machine precision, so the closed form can be checked against it without
   reference to any published value.
3. **Behavioural properties.** Invariances and monotonicities that must hold
   whatever the formula.
"""

import numpy as np
import pytest
from scipy import integrate, special, stats

from bayesplain.core import dirichlet_multinomial as dm

# ---------------------------------------------------------------------------
# Data from Jamil et al. (2017)
# ---------------------------------------------------------------------------

#: Hraba & Grant (1970) doll preference. Rows are the child's race, and the
#: row totals (89 African American, 71 white children) were fixed by the
#: sampling design, so this is the independent multinomial scheme.
#: Paper reports log BF10 = 23.03 and chi-square = 46.71.
DOLLS = np.array([[62.0, 27.0], [11.0, 60.0]])

#: Simulation 1 of the paper: the table (3, 3, 2, 5) scaled by c = 10.
#: Paper reports BF10 = 9.19 (Poisson) and BF10 = 3.04 (hypergeometric).
SIM_C10 = np.array([[3.0, 3.0], [2.0, 5.0]]) * 10


class TestGoldenMasters:
    @pytest.mark.validation
    def test_dolls_independent_multinomial(self):
        got = dm.log_bayes_factor_independence(
            DOLLS, 1.0, scheme="independent_multinomial", fixed="rows"
        )
        assert got == pytest.approx(23.03, abs=0.005)

    @pytest.mark.validation
    def test_dolls_chi_square_matches_too(self):
        # Confirms the table is oriented the way the paper had it.
        chi2, _, dof, _ = stats.chi2_contingency(DOLLS, correction=False)
        assert chi2 == pytest.approx(46.71, abs=0.01)
        assert dof == 1

    @pytest.mark.validation
    def test_simulation_poisson(self):
        got = dm.log_bayes_factor_independence(SIM_C10, 1.0, scheme="poisson")
        assert np.exp(got) == pytest.approx(9.19, abs=0.01)

    @pytest.mark.validation
    def test_simulation_hypergeometric(self):
        got = dm.log_bayes_factor_independence(SIM_C10, 1.0, scheme="hypergeometric")
        assert np.exp(got) == pytest.approx(3.04, abs=0.01)

    @pytest.mark.validation
    def test_hypergeometric_matches_jeffreys_closed_form(self):
        # Equation 17: an independent closed form for the 2x2 case, using the
        # SMALLEST of the four marginal totals in the leading factorial.
        def jeffreys(table):
            table = np.asarray(table, dtype=float)
            margins = np.concatenate([table.sum(1), table.sum(0)])
            smallest, others = margins.min(), np.sort(margins)[1:]
            return (
                special.gammaln(table.ravel() + 1).sum()
                + special.gammaln(table.sum() + 1)
                - special.gammaln(smallest + 2)
                - special.gammaln(others + 1).sum()
            )

        for table in (SIM_C10, DOLLS, np.array([[3.0, 7.0], [6.0, 4.0]])):
            got = dm.log_bayes_factor_independence(table, 1.0, scheme="hypergeometric")
            assert got == pytest.approx(jeffreys(table), abs=1e-9)

    @pytest.mark.validation
    def test_evidence_ordering_across_schemes(self):
        # The paper's Table 3: evidence against independence decreases with
        # successive conditioning on margins and totals.
        values = [
            dm.log_bayes_factor_independence(SIM_C10, 1.0, scheme=s) for s in dm.SCHEMES
        ]
        assert values == sorted(values, reverse=True)

    @pytest.mark.validation
    def test_hypergeometric_supports_the_null_most_strongly(self):
        # Simulation 2: uniform counts give maximal support for independence.
        # The BF10 ordering P > M > I > H is a fixed algebraic relation (the
        # paper's Table 3) and does not depend on the data, so it still holds
        # here. What the paper calls the reversal is that the *strength of
        # evidence for the favoured hypothesis* now runs the other way: with
        # every BF10 below 1, the smallest of them is the largest BF01.
        uniform = np.full((2, 2), 100.0)
        values = {
            s: dm.log_bayes_factor_independence(uniform, 1.0, scheme=s)
            for s in dm.SCHEMES
        }
        assert list(values.values()) == sorted(values.values(), reverse=True)
        assert all(v < 0 for v in values.values())
        assert min(values, key=values.get) == "hypergeometric"
        assert max(values, key=values.get) == "poisson"


# ---------------------------------------------------------------------------
# Independent reimplementation by quadrature
# ---------------------------------------------------------------------------


def _marginal_dependent(table, a1, a2):
    """p(data | rows have their own rate), by numerical integration."""
    total = 1.0
    for row in np.asarray(table, dtype=float):
        successes, trials = row[0], row.sum()
        value, _ = integrate.quad(
            lambda p: stats.binom.pmf(successes, trials, p) * stats.beta.pdf(p, a1, a2),
            0.0,
            1.0,
        )
        total *= value
    return total


def _marginal_independent(table, a1, a2):
    """p(data | all rows share one rate), by numerical integration."""
    rows = np.asarray(table, dtype=float)

    def integrand(p):
        out = stats.beta.pdf(p, a1, a2)
        for row in rows:
            out *= stats.binom.pmf(row[0], row.sum(), p)
        return out

    value, _ = integrate.quad(integrand, 0.0, 1.0)
    return value


class TestAgainstQuadrature:
    @pytest.mark.parametrize(
        "table",
        [
            [[3, 7], [6, 4]],
            [[34, 186], [51, 189]],
            [[1, 19], [15, 5]],
            [[10, 10], [10, 10]],
            [[0, 12], [7, 5]],
        ],
    )
    @pytest.mark.parametrize(("a1", "a2"), [(1.0, 1.0), (2.0, 5.0), (0.5, 0.5)])
    def test_independent_multinomial_matches_integration(self, table, a1, a2):
        expected = _marginal_dependent(table, a1, a2) / _marginal_independent(
            table, a1, a2
        )
        got = np.exp(dm.log_bayes_factor_independence(table, [a1, a2]))
        assert got == pytest.approx(expected, rel=1e-7)

    def test_three_rows_also_matches(self):
        table = [[8, 12], [14, 6], [3, 17]]
        expected = _marginal_dependent(table, 1.0, 1.0) / _marginal_independent(
            table, 1.0, 1.0
        )
        got = np.exp(dm.log_bayes_factor_independence(table, 1.0))
        assert got == pytest.approx(expected, rel=1e-7)

    @pytest.mark.slow
    def test_three_column_table_matches_monte_carlo(self):
        # No one-dimensional reduction is available once there are three
        # columns, so integrate the two marginal likelihoods by sampling from
        # the Dirichlet priors directly.
        table = np.array([[6, 9, 5], [12, 3, 5]], dtype=float)
        alpha = np.ones(3)
        rng = np.random.default_rng(0)
        n_sim = 2_000_000

        def log_multinomial(counts, probs):
            coef = special.gammaln(counts.sum() + 1) - special.gammaln(counts + 1).sum()
            return coef + (counts * np.log(probs)).sum(axis=-1)

        own = rng.dirichlet(alpha, size=(n_sim, 2))
        dep = np.exp(
            log_multinomial(table[0], own[:, 0]) + log_multinomial(table[1], own[:, 1])
        ).mean()
        shared = rng.dirichlet(alpha, size=n_sim)
        ind = np.exp(
            log_multinomial(table[0], shared) + log_multinomial(table[1], shared)
        ).mean()

        got = np.exp(dm.log_bayes_factor_independence(table, 1.0))
        assert got == pytest.approx(dep / ind, rel=0.02)


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------


class TestBehaviour:
    @pytest.mark.parametrize("scheme", dm.SCHEMES)
    def test_perfectly_proportional_rows_favour_independence(self, scheme):
        table = [[20, 30], [40, 60]]  # identical rates, exactly
        assert dm.log_bayes_factor_independence(table, scheme=scheme) < 0

    @pytest.mark.parametrize("scheme", dm.SCHEMES)
    def test_strongly_divergent_rows_favour_association(self, scheme):
        table = [[90, 10], [10, 90]]
        assert dm.log_bayes_factor_independence(table, scheme=scheme) > np.log(1e6)

    @pytest.mark.parametrize("scheme", dm.SCHEMES)
    def test_invariant_to_row_order(self, scheme):
        table = np.array([[34, 186], [51, 189]])
        a = dm.log_bayes_factor_independence(table, scheme=scheme)
        b = dm.log_bayes_factor_independence(table[::-1], scheme=scheme)
        assert a == pytest.approx(b)

    @pytest.mark.parametrize("scheme", dm.SCHEMES)
    def test_invariant_to_column_relabelling(self, scheme):
        table = np.array([[34, 186], [51, 189]])
        a = dm.log_bayes_factor_independence(table, 1.0, scheme=scheme)
        b = dm.log_bayes_factor_independence(table[:, ::-1], 1.0, scheme=scheme)
        assert a == pytest.approx(b)

    def test_transposing_swaps_which_margin_is_fixed(self):
        table = np.array([[34, 186], [51, 189]])
        rows = dm.log_bayes_factor_independence(table, fixed="rows")
        cols = dm.log_bayes_factor_independence(table.T, fixed="columns")
        assert rows == pytest.approx(cols)

    def test_more_data_at_the_same_rates_strengthens_the_evidence(self):
        small = dm.log_bayes_factor_independence([[15, 35], [25, 25]])
        large = dm.log_bayes_factor_independence([[150, 350], [250, 250]])
        assert large > small

    def test_larger_concentration_weakens_evidence_for_association(self):
        # Higher `a` pulls H1's predictions toward H0, per Jamil et al. p. 5.
        table = [[34, 186], [51, 189]]
        flat = dm.log_bayes_factor_independence(table, 1.0)
        tight = dm.log_bayes_factor_independence(table, 10.0)
        assert tight < flat


class TestValidation:
    @pytest.mark.parametrize(
        "table",
        [
            [[1, 2, 3]],
            [[1], [2]],
            [[1, 2], [3, -1]],
            [[1.5, 2], [3, 4]],
            [[0, 0], [3, 4]],
            [[0, 5], [0, 4]],
        ],
    )
    def test_bad_tables_rejected(self, table):
        with pytest.raises(ValueError):
            dm.validate_table(table)

    def test_one_dimensional_input_rejected(self):
        with pytest.raises(ValueError, match="two-dimensional"):
            dm.validate_table([1, 2, 3])

    def test_wrong_length_concentration_rejected(self):
        with pytest.raises(ValueError, match="one value per column"):
            dm.log_bayes_factor_independence([[3, 7], [6, 4]], [1.0, 1.0, 1.0])

    def test_nonpositive_concentration_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            dm.log_bayes_factor_independence([[3, 7], [6, 4]], 0.0)

    def test_unknown_scheme_lists_the_options(self):
        with pytest.raises(ValueError, match="scheme must be one of"):
            dm.log_bayes_factor_independence([[3, 7], [6, 4]], scheme="binomial")

    def test_bad_fixed_margin_rejected(self):
        with pytest.raises(ValueError, match="'rows' or 'columns'"):
            dm.log_bayes_factor_independence([[3, 7], [6, 4]], fixed="diagonal")

    def test_asymmetric_concentration_rejected_where_undefined(self):
        with pytest.raises(ValueError, match="symmetric concentration only"):
            dm.log_bayes_factor_independence(
                [[3, 7], [6, 4]], [1.0, 5.0], scheme="joint_multinomial"
            )

    def test_uniform_vector_concentration_accepted_everywhere(self):
        scalar = dm.log_bayes_factor_independence(
            [[3, 7], [6, 4]], 2.0, scheme="joint_multinomial"
        )
        vector = dm.log_bayes_factor_independence(
            [[3, 7], [6, 4]], [2.0, 2.0], scheme="joint_multinomial"
        )
        assert scalar == pytest.approx(vector)

    def test_hypergeometric_rejects_larger_tables(self):
        with pytest.raises(ValueError, match="2x2 tables only"):
            dm.log_bayes_factor_independence(
                [[3, 7, 2], [6, 4, 1]], scheme="hypergeometric"
            )


class TestPosteriorCells:
    def test_draws_are_valid_probability_tables(self):
        rng = np.random.default_rng(1)
        draws = dm.posterior_cell_draws([[3, 7], [6, 4]], size=500, rng=rng)
        assert draws.shape == (500, 2, 2)
        assert np.allclose(draws.sum(axis=(1, 2)), 1.0)
        assert (draws >= 0).all()

    def test_posterior_mean_approaches_observed_shares(self):
        rng = np.random.default_rng(2)
        table = np.array([[300, 700], [600, 400]], dtype=float)
        draws = dm.posterior_cell_draws(table, size=20_000, rng=rng)
        assert np.allclose(draws.mean(axis=0), table / table.sum(), atol=0.01)


class TestEffectSizes:
    def test_cramers_v_zero_under_exact_independence(self):
        probs = np.outer([0.4, 0.6], [0.3, 0.7])
        assert dm.cramers_v(probs) == pytest.approx(0.0)

    def test_cramers_v_one_under_perfect_association(self):
        probs = np.array([[0.5, 0.0], [0.0, 0.5]])
        assert dm.cramers_v(probs) == pytest.approx(1.0)

    def test_cramers_v_handles_a_stack(self):
        rng = np.random.default_rng(3)
        draws = dm.posterior_cell_draws([[30, 70], [60, 40]], size=1000, rng=rng)
        values = dm.cramers_v(draws)
        assert values.shape == (1000,)
        assert ((values >= 0) & (values <= 1)).all()

    def test_effect_size_posterior_excludes_zero_for_strong_association(self):
        # The payoff of the Dirichlet route: an interval on effect size, which
        # a chi-square test cannot produce.
        rng = np.random.default_rng(4)
        draws = dm.posterior_cell_draws([[90, 10], [10, 90]], size=20_000, rng=rng)
        assert np.quantile(dm.cramers_v(draws), 0.025) > 0.5

    def test_cramers_v_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match="table or a"):
            dm.cramers_v(np.ones(4))

    def test_log_odds_ratio_zero_under_independence(self):
        probs = np.outer([0.4, 0.6], [0.3, 0.7])
        assert dm.log_odds_ratio(probs) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.validation
    def test_log_odds_ratio_posterior_matches_the_dolls_example(self):
        # Paper reports a posterior median log odds ratio of 2.47 with a 95%
        # credible interval of (1.73, 3.26).
        rng = np.random.default_rng(5)
        draws = dm.posterior_cell_draws(DOLLS, 1.0, size=200_000, rng=rng)
        values = dm.log_odds_ratio(draws)
        assert np.median(values) == pytest.approx(2.47, abs=0.05)
        lo, hi = np.quantile(values, [0.025, 0.975])
        assert lo == pytest.approx(1.73, abs=0.08)
        assert hi == pytest.approx(3.26, abs=0.08)

    def test_log_odds_ratio_rejects_larger_tables(self):
        with pytest.raises(ValueError, match="2x2 table"):
            dm.log_odds_ratio(np.ones((3, 3)) / 9)
