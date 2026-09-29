"""Tests for goodness of fit on one categorical variable."""

import numpy as np
import pytest
from scipy import stats

import bayesplain as bp
from bayesplain.core import beta_binomial as bb
from bayesplain.core import dirichlet_multinomial as dm

DAYS = {"Mon": 31, "Tue": 28, "Wed": 30, "Thu": 33, "Fri": 45, "Sat": 38, "Sun": 25}


class TestCore:
    def test_two_categories_reduce_to_beta_binomial(self):
        got = dm.log_bayes_factor_goodness_of_fit([34, 186], [0.1, 0.9])
        want = bb.log_bayes_factor_point_null(34, 220, 0.1, 1.0, 1.0)
        assert got == pytest.approx(want, abs=1e-9)

    def test_cohens_w_is_zero_at_the_expected_shares(self):
        assert dm.cohens_w([0.2, 0.3, 0.5], [0.2, 0.3, 0.5])[0] == pytest.approx(0)

    def test_cohens_w_matches_the_chi_square_identity(self):
        counts = np.array([31, 28, 30, 33, 45, 38, 25])
        observed = counts / counts.sum()
        chi2 = stats.chisquare(counts).statistic
        w = dm.cohens_w(observed, np.ones(7))[0]
        assert w == pytest.approx(np.sqrt(chi2 / counts.sum()))


class TestCategories:
    def test_twin_is_the_chi_square_goodness_of_fit_test(self):
        res = bp.categories(DAYS)
        want = stats.chisquare(list(DAYS.values()))
        assert res.frequentist.statistic == pytest.approx(want.statistic)
        assert res.frequentist.pvalue == pytest.approx(want.pvalue)

    def test_shares_table_flags_the_category_that_stands_out(self):
        res = bp.categories(DAYS)
        assert res.shares_table["Fri"]["probability_above_expected"] > 0.95
        assert "Fri" in str(res.shares())

    def test_marginal_shares_are_exact_betas(self):
        res = bp.categories(DAYS)
        total = sum(DAYS.values())
        fri = res.components["Fri"]
        assert fri.mean() == pytest.approx((45 + 1) / (total + 7))

    def test_exactly_expected_data_do_not_report_a_departure(self):
        # The noise-floor bar is the point: data matching the expected shares
        # exactly must land near 0.5, not claim a confident departure.
        res = bp.categories([33] * 7)
        assert res.probability(">") == pytest.approx(0.5, abs=0.05)

    def test_large_samples_fall_back_to_cohens_small_bar(self):
        res = bp.categories([3300] * 7)
        assert res.direction_reference == pytest.approx(bp.SMALL_EFFECT_W)

    def test_explicit_threshold_is_respected(self):
        res = bp.categories(DAYS, threshold=0.3)
        assert res.direction_reference == 0.3

    def test_raw_labels_are_tallied(self):
        raw = ["dry"] * 40 + ["wet"] * 9 + ["icy"] * 1
        res = bp.categories(raw, expected={"dry": 0.8, "wet": 0.15, "icy": 0.05})
        assert res.shares_table["dry"]["observed"] == 40

    def test_pandas_value_counts_keep_their_names(self):
        pd = pytest.importorskip("pandas")
        counts = pd.Series(["a"] * 5 + ["b"] * 8 + ["c"] * 3).value_counts()
        res = bp.categories(counts)
        assert set(res.shares_table) == {"a", "b", "c"}

    def test_plain_list_of_counts_gets_default_names(self):
        res = bp.categories([10, 20, 30])
        assert list(res.shares_table) == ["category 1", "category 2", "category 3"]

    def test_uneven_expected_shares_centre_the_prior(self):
        # With a prior centred on the expected shares, data matching them
        # exactly leave each posterior share centred there too.
        shares = np.array([0.6, 0.3, 0.1])
        res = bp.categories(list(shares * 1000), expected=shares)
        for name, share in zip(res.shares_table, shares):
            assert res.components[name].mean() == pytest.approx(share, abs=1e-9)

    @pytest.mark.parametrize(
        "data, expected, match",
        [
            ([5], None, "at least two categories"),
            ([5, -1], None, "non-negative"),
            ([5.5, 3], None, "whole numbers"),
            ([5, 3], [1, 0], "must be positive"),
            ({"a": 3, "b": 4}, {"a": 1}, "no share for"),
        ],
    )
    def test_bad_inputs_are_explained(self, data, expected, match):
        with pytest.raises(ValueError, match=match):
            bp.categories(data, expected=expected)

    def test_sensitivity_and_plot(self):
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        res = bp.categories(DAYS)
        assert "Dirichlet" in str(res.sensitivity())
        assert res.plot(kind="shares") is not None
