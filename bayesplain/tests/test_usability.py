"""Tests for the conveniences: data frames, missing values, plots, intervals."""

import numpy as np
import pytest
from scipy import stats

import bayesplain as bp
from bayesplain.core import intervals

pd = pytest.importorskip("pandas")


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(11)
    n = 90
    return pd.DataFrame(
        {
            "district": np.repeat(["North", "South", "East"], n // 3),
            "rent": rng.normal(1500, 200, n),
            "income": rng.normal(60, 12, n),
            "renter": rng.integers(0, 2, n).astype(bool),
        }
    )


class TestDataFrames:
    def test_mean_reads_a_column_and_labels_itself(self, frame):
        res = bp.mean("rent", data=frame)
        assert "rent" in res.quantity
        assert res.point() == pytest.approx(bp.mean(frame.rent).point())

    def test_compare_means_long_format(self, frame):
        res = bp.compare_means("rent", by="renter", data=frame)
        assert res.lower_label == "renter=False"
        assert res.higher_label == "renter=True"
        want = bp.compare_means(
            frame.rent[~frame.renter], frame.rent[frame.renter]
        ).point()
        assert res.point() == pytest.approx(want)

    def test_compare_means_wide_format(self, frame):
        res = bp.compare_means("rent", "income", data=frame, paired=True)
        assert "income − rent" in res.quantity

    def test_correlation_and_regression_take_names(self, frame):
        assert bp.correlation("income", "rent", data=frame).quantity == (
            "correlation between income and rent"
        )
        assert (
            "rent per unit of income"
            in bp.regression("income", "rent", data=frame).quantity
        )

    def test_compare_groups_with_named_columns(self, frame):
        res = bp.compare_groups(frame, by="district", values="rent")
        assert res.group_names == ["East", "North", "South"]

    def test_explicit_labels_win(self, frame):
        res = bp.correlation("income", "rent", data=frame, labels=["a", "b"])
        assert res.quantity == "correlation between a and b"

    @pytest.mark.parametrize(
        "call, match",
        [
            (lambda f: bp.mean("nope", data=f), "no column named 'nope'"),
            (lambda f: bp.mean("rent"), "no data= was given"),
            (lambda f: bp.mean(f.rent, data=f), "should be a column name"),
            (
                lambda f: bp.compare_means("rent", by="district", data=f),
                "exactly two groups",
            ),
            (
                lambda f: bp.compare_means("rent", "income", by="renter", data=f),
                "either y or by",
            ),
            (
                lambda f: bp.compare_means("rent", by="renter", data=f, paired=True),
                "side by side",
            ),
            (lambda f: bp.compare_means("rent", data=f), "needs a second sample"),
            (lambda f: bp.compare_groups(f, by="district"), "name both columns"),
        ],
    )
    def test_mistakes_are_explained(self, frame, call, match):
        with pytest.raises(ValueError, match=match):
            call(frame)


class TestMissingValues:
    def test_each_analysis_says_what_it_dropped(self):
        nan = np.nan
        cases = [
            (bp.mean([1, 2, nan, 4, 5]), "1 missing value was dropped"),
            (
                bp.compare_means([1, 2, nan, 4], [3, 4, 5, nan, nan]),
                "3 missing values were dropped",
            ),
            (
                bp.compare_means([1, 2, nan, 4, 6], [3, 5, 5, 6, nan], paired=True),
                "2 incomplete pairs were dropped",
            ),
            (
                bp.correlation([1, 2, nan, 4, 5, 6], [2, 1, 3, 5, 4, 7]),
                "1 incomplete pair was dropped",
            ),
            (
                bp.regression([1, 2, nan, 4, 5, 6], [2, 1, 3, 5, 4, 7]),
                "1 incomplete row was dropped",
            ),
            (
                bp.compare_groups({"a": [1, 2, nan, 3], "b": [3, 4, 5], "c": [5, 6]}),
                "1 missing value was dropped",
            ),
        ]
        for res, phrase in cases:
            assert any(phrase in note for note in res.notes), phrase

    def test_complete_data_get_no_note(self):
        assert not any("dropped" in note for note in bp.mean([1, 2, 3, 4]).notes)


class TestExactIntervals:
    def test_zero_successes_start_at_exactly_zero(self):
        assert bp.proportion(0, 20).interval()[0] == 0.0
        assert bp.proportion(20, 20).interval()[1] == 1.0

    @pytest.mark.parametrize(
        "dist",
        [
            stats.beta(35, 187),
            stats.t(9, 34.6, 2.4),
            stats.gamma(14.5, scale=0.25),
            stats.betaprime(9.5, 31.5, scale=2.0),
        ],
    )
    def test_analytic_hdi_matches_a_large_sample(self, dist):
        exact = intervals.hdi_from_dist(dist)
        sampled = intervals.hdi_from_draws(
            dist.rvs(2_000_000, random_state=np.random.default_rng(0))
        )
        assert exact == pytest.approx(sampled, rel=0.01)
        assert dist.cdf(exact[1]) - dist.cdf(exact[0]) == pytest.approx(0.95)

    def test_exact_results_have_deterministic_hdis(self):
        a = bp.proportion(34, 220, seed=1).interval()
        b = bp.proportion(34, 220, seed=2).interval()
        assert a == b

    def test_summary_names_the_median(self):
        assert "best estimate (median)" in str(bp.proportion(34, 220).summary())


def _every_result():
    rng = np.random.default_rng(3)
    x, y = rng.normal(30, 8, 40), rng.normal(33, 9, 40)
    groups = {"a": rng.normal(10, 2, 20), "b": rng.normal(11, 2, 25), "c": x[:15]}
    return {
        "proportion": bp.proportion(34, 220),
        "compare_proportions": bp.compare_proportions([34, 51], [220, 240]),
        "several_proportions": bp.compare_proportions([3, 9, 20], [40, 50, 60]),
        "rate": bp.rate(14, 4, reference=2.0, prior=(2, 1)),
        "compare_rates": bp.compare_rates([31, 9], [4, 2]),
        "contingency": bp.contingency([[20, 10], [8, 22]]),
        "categories": bp.categories([10, 14, 21]),
        "mean": bp.mean(x),
        "log_mean": bp.mean(np.abs(x), reference=20, log=True),
        "compare_means": bp.compare_means(x, y),
        "paired": bp.compare_means(x, y, paired=True),
        "log_ratio": bp.compare_means(np.abs(x), np.abs(y), log=True),
        "correlation": bp.correlation(x, y),
        "regression": bp.regression(x, y),
        "compare_groups": bp.compare_groups(groups, pool=True),
    }


@pytest.mark.parametrize("name, result", list(_every_result().items()))
def test_every_plot_kind_draws(name, result):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for kind in result.plot_kinds():
        ax = result.plot(kind=kind)
        assert ax is not None, (name, kind)
        plt.close("all")
    # A threshold shades the tail, on either side of the mass.
    lo, hi = result.interval()
    result.plot(threshold=lo)
    result.plot(threshold=hi)
    plt.close("all")
