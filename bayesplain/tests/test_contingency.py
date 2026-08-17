"""Tests for the public contingency-table analysis."""

import numpy as np
import pytest

import bayesplain as bp

DOLLS = [[62, 27], [11, 60]]
MODE_BY_DISTRICT = [[120, 45, 35], [90, 80, 30], [60, 40, 100]]


class TestDollsExample:
    @pytest.fixture
    def dolls(self):
        return bp.contingency(
            DOLLS,
            row_labels=["Black children", "white children"],
            col_labels=["black doll", "white doll"],
        )

    @pytest.mark.validation
    def test_bayes_factor_matches_the_paper(self, dolls):
        assert dolls.bayes_factor().log_bf10 == pytest.approx(23.03, abs=0.005)

    @pytest.mark.validation
    def test_effect_size_posterior_matches_the_paper(self, dolls):
        assert dolls.point("median") == pytest.approx(2.47, abs=0.05)
        lo, hi = dolls.interval(kind="eti")
        assert lo == pytest.approx(1.73, abs=0.08)
        assert hi == pytest.approx(3.26, abs=0.08)

    def test_2x2_defaults_to_the_log_odds_ratio(self, dolls):
        assert dolls.direction_reference == 0.0
        assert "log odds ratio" in dolls.quantity

    def test_bayes_factor_reads_as_association_not_difference(self, dolls):
        assert "association" in dolls.bayes_factor().interpretation


class TestEffectChoice:
    def test_larger_table_defaults_to_cramers_v(self):
        res = bp.contingency(MODE_BY_DISTRICT)
        assert "Cramér's V" in res.quantity
        assert res.direction_reference == bp.SMALL_EFFECT_V

    def test_cramers_v_is_bounded(self):
        res = bp.contingency(MODE_BY_DISTRICT)
        assert (res.draws >= 0).all()
        assert (res.draws <= 1).all()

    def test_cramers_v_explains_that_zero_is_not_a_null(self):
        res = bp.contingency(MODE_BY_DISTRICT)
        assert any("cannot be negative" in note for note in res.notes)

    def test_threshold_can_be_overridden(self):
        res = bp.contingency(MODE_BY_DISTRICT, threshold=0.25)
        assert res.direction_reference == 0.25

    def test_log_odds_ratio_rejected_for_larger_tables(self):
        with pytest.raises(ValueError, match="only defined for a 2x2"):
            bp.contingency(MODE_BY_DISTRICT, effect="log_odds_ratio")

    def test_unknown_effect_rejected(self):
        with pytest.raises(ValueError, match="effect must be"):
            bp.contingency(DOLLS, effect="phi")


class TestSchemes:
    @pytest.mark.parametrize("scheme", bp.core.dirichlet_multinomial.SCHEMES)
    def test_every_scheme_runs(self, scheme):
        res = bp.contingency(DOLLS, scheme=scheme)
        assert np.isfinite(res.log_bf10)

    def test_scheme_changes_the_bayes_factor_but_not_the_posterior(self):
        a = bp.contingency(DOLLS, scheme="poisson")
        b = bp.contingency(DOLLS, scheme="hypergeometric")
        assert a.log_bf10 != pytest.approx(b.log_bf10)
        # The effect-size posterior does not depend on the sampling scheme.
        assert np.array_equal(a.draws, b.draws)

    def test_summary_names_the_scheme_in_the_caveat(self):
        res = bp.contingency(DOLLS, scheme="joint_multinomial")
        assert "joint multinomial" in res.bayes_factor().caveat


class TestSparseTables:
    def test_small_cells_get_a_note(self):
        res = bp.contingency([[1, 20], [3, 18]])
        assert any("expected counts below 5" in note for note in res.notes)

    def test_posterior_still_finite_with_a_zero_cell(self):
        res = bp.contingency([[0, 20], [7, 14]])
        lo, hi = res.interval()
        assert np.isfinite([lo, hi]).all()

    def test_tiny_table_flags_prior_dependence(self):
        res = bp.contingency([[2, 3], [4, 1]])
        assert any("prior is doing visible work" in note for note in res.notes)


class TestLabels:
    def test_labels_appear_in_the_quantity(self):
        res = bp.contingency(DOLLS, row_labels=["A", "B"], col_labels=["yes", "no"])
        assert "yes" in res.quantity and "A" in res.quantity

    def test_wrong_label_count_rejected(self):
        with pytest.raises(ValueError, match="row_labels has 3 names"):
            bp.contingency(DOLLS, row_labels=["a", "b", "c"])
        with pytest.raises(ValueError, match="col_labels has 1 name"):
            bp.contingency(DOLLS, col_labels=["only"])

    def test_pandas_crosstab_labels_are_picked_up(self):
        pd = pytest.importorskip("pandas")
        table = pd.DataFrame(
            DOLLS,
            index=["Black children", "white children"],
            columns=["black doll", "white doll"],
        )
        res = bp.contingency(table)
        assert "black doll" in res.quantity
        assert "Black children" in res.components

    def test_components_are_row_profiles(self):
        res = bp.contingency(DOLLS, row_labels=["A", "B"])
        assert set(res.components) == {"A", "B"}
        # Row A prefers the first column; row B does not.
        assert res.components["A"].mean() > res.components["B"].mean()


class TestPriorAndSensitivity:
    def test_table_prior_presets_resolve(self):
        for name in bp.priors.TABLE_PRIORS:
            assert bp.priors.resolve_table(name).name == name

    def test_number_resolves_to_custom(self):
        assert bp.priors.resolve_table(3.0).a == 3.0

    def test_symmetric_beta_prior_is_accepted(self):
        resolved = bp.priors.resolve_table(bp.priors.BetaPrior(2.0, 2.0))
        assert resolved.a == 2.0

    def test_asymmetric_beta_prior_rejected(self):
        with pytest.raises(ValueError, match="must be symmetric"):
            bp.priors.resolve_table(bp.priors.BetaPrior(2.0, 5.0))

    def test_unknown_name_lists_options(self):
        with pytest.raises(ValueError, match="Available presets"):
            bp.priors.resolve_table("vague")

    def test_sensitivity_runs_the_ladder(self):
        text = " ".join(str(bp.contingency(DOLLS).sensitivity()).split())
        for name in bp.priors.SENSITIVITY_LADDER:
            assert name in text

    def test_stronger_prior_shrinks_the_effect(self):
        weak = bp.contingency([[8, 12], [14, 6]], prior="uninformed").point()
        strong = bp.contingency([[8, 12], [14, 6]], prior="skeptical").point()
        assert abs(strong) < abs(weak)


class TestOutput:
    def test_summary_lines_fit(self):
        res = bp.contingency(
            DOLLS,
            row_labels=["Black children", "white children"],
            col_labels=["black doll", "white doll"],
        )
        for line in str(res.summary()).splitlines():
            assert len(line) <= 78, f"line too wide ({len(line)}): {line!r}"

    def test_tiny_p_value_does_not_print_as_zero_percent(self):
        res = bp.contingency(DOLLS)
        assert "0.0% of the time" not in res.frequentist.claims()

    def test_prior_posterior_plot_refuses_with_a_clear_message(self):
        pytest.importorskip("matplotlib")
        res = bp.contingency(DOLLS)
        with pytest.raises(ValueError, match="no single density curve"):
            res.plot(kind="prior_posterior")

    def test_to_dict_round_trips(self):
        import json

        out = bp.contingency(DOLLS).to_dict()
        assert json.loads(json.dumps(out))["analysis"] == "contingency"
