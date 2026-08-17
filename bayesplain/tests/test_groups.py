"""Tests for many-group comparison, partial pooling, and the shipped datasets."""

import numpy as np
import pytest

import bayesplain as bp
from bayesplain.core import hierarchical


@pytest.fixture(scope="module")
def districts():
    rng = np.random.default_rng(1)
    return {
        "North": rng.normal(12.0, 3.0, 45),
        "South": rng.normal(15.5, 3.0, 40),
        "East": rng.normal(12.5, 3.0, 50),
        "West": rng.normal(13.0, 3.0, 38),
        "Harbor": rng.normal(17.0, 4.0, 7),  # tiny group, looks best
    }


class TestShrinkage:
    def test_noisier_groups_are_pulled_harder(self):
        out = hierarchical.shrink([10.0, 12.0, 25.0], [1.0, 1.0, 8.0])
        assert out["weights"][2] < out["weights"][0]

    def test_shrunk_estimate_lies_between_original_and_grand_mean(self):
        out = hierarchical.shrink([10.0, 12.0, 25.0], [1.0, 1.0, 8.0])
        grand = out["grand_mean"]
        assert grand < out["means"][2] < 25.0

    def test_precise_groups_barely_move(self):
        out = hierarchical.shrink([10.0, 20.0, 30.0], [0.05, 0.05, 0.05])
        assert np.allclose(out["means"], [10.0, 20.0, 30.0], atol=0.05)
        assert np.all(out["weights"] > 0.99)

    def test_identical_groups_collapse_to_the_grand_mean(self):
        # No spread beyond noise, so tau-squared is zero and pooling is total.
        out = hierarchical.shrink([10.0, 10.05, 9.95], [1.0, 1.0, 1.0])
        assert out["tau2"] == 0.0
        assert np.allclose(out["means"], out["grand_mean"])

    def test_shrinking_never_increases_uncertainty(self):
        se = np.array([1.0, 2.0, 5.0])
        out = hierarchical.shrink([10.0, 14.0, 22.0], se)
        assert np.all(out["standard_errors"] <= se)

    def test_between_group_variance_is_non_negative(self):
        assert hierarchical.between_group_variance([1.0, 1.0], [5.0, 5.0]) >= 0

    @pytest.mark.parametrize(
        ("means", "errors", "match"),
        [
            ([1.0], [1.0], "at least 2 groups"),
            ([1.0, 2.0], [0.0, 1.0], "must be positive"),
            ([1.0, 2.0], [1.0], "same shape"),
        ],
    )
    def test_invalid_input_rejected(self, means, errors, match):
        with pytest.raises(ValueError, match=match):
            hierarchical.between_group_variance(means, errors)


class TestCompareGroups:
    def test_components_cover_every_group(self, districts):
        res = bp.compare_groups(districts)
        assert len(res.components) == len(districts)

    def test_headline_quantity_is_the_spread(self, districts):
        res = bp.compare_groups(districts)
        assert "spread" in res.quantity
        assert (res.draws >= 0).all()

    def test_two_groups_are_redirected(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="use compare_means"):
            bp.compare_groups([rng.normal(size=20), rng.normal(size=20)])

    def test_accepts_a_sequence_with_labels(self):
        rng = np.random.default_rng(2)
        res = bp.compare_groups(
            [rng.normal(size=30) for _ in range(3)], labels=["a", "b", "c"]
        )
        assert res.group_names == ["a", "b", "c"]

    def test_accepts_long_form_with_by(self):
        rng = np.random.default_rng(3)
        values = rng.normal(size=90)
        keys = np.repeat(["x", "y", "z"], 30)
        res = bp.compare_groups(values, by=keys)
        assert res.group_names == ["x", "y", "z"]
        assert res.group_sizes.tolist() == [30, 30, 30]

    def test_mismatched_by_length_rejected(self):
        with pytest.raises(ValueError, match="line up one to one"):
            bp.compare_groups(np.zeros(10), by=np.repeat(["a", "b"], 3))

    def test_wrong_label_count_rejected(self):
        rng = np.random.default_rng(4)
        with pytest.raises(ValueError, match="labels has 2 names"):
            bp.compare_groups(
                [rng.normal(size=20) for _ in range(3)], labels=["a", "b"]
            )


class TestPooling:
    def test_smallest_group_moves_the_most(self, districts):
        pooled = bp.compare_groups(districts, pool=True)
        weights = dict(zip(pooled.group_names, pooled.pooling["weights"]))
        assert weights["Harbor"] == min(weights.values())
        assert weights["Harbor"] < 0.7

    def test_pooling_can_change_the_ranking(self, districts):
        raw = bp.compare_groups(districts)
        pooled = bp.compare_groups(districts, pool=True)

        def top(result):
            return max(
                result.group_names,
                key=lambda g: np.median(result.group_draws[g]),
            )

        # The week 9 lesson: the smallest site wins until it is shrunk.
        assert top(raw) == "Harbor"
        assert top(pooled) != "Harbor"

    def test_pooling_narrows_the_noisiest_group(self, districts):
        raw = bp.compare_groups(districts)
        pooled = bp.compare_groups(districts, pool=True)
        width = lambda res: np.diff(  # noqa: E731
            np.quantile(res.group_draws["Harbor"], [0.025, 0.975])
        )[0]
        assert width(pooled) < width(raw)

    def test_pooling_is_reported_in_the_notes(self, districts):
        pooled = bp.compare_groups(districts, pool=True)
        assert any("partial pooling is on" in note for note in pooled.notes)

    def test_small_group_warning_without_pooling(self, districts):
        res = bp.compare_groups(districts)
        assert any("smallest group has 7" in note for note in res.notes)


class TestPairwise:
    def test_lists_every_pair_by_default(self, districts):
        text = str(bp.compare_groups(districts).pairwise())
        assert text.count("−") >= 10  # 5 choose 2

    def test_can_be_restricted(self, districts):
        text = str(bp.compare_groups(districts).pairwise(only=[("South", "North")]))
        assert "South − North" in text
        assert "East − West" not in text

    def test_unknown_group_rejected(self, districts):
        with pytest.raises(ValueError, match="unknown group"):
            bp.compare_groups(districts).pairwise(only=[("South", "Nowhere")])

    def test_reports_bayes_factors_per_pair(self, districts):
        text = str(bp.compare_groups(districts).pairwise())
        assert "BF10 (JZS)" in text

    def test_rope_adds_practical_verdicts(self, districts):
        text = str(bp.compare_groups(districts).pairwise(rope=(-1.0, 1.0)))
        assert "AGAINST A ROPE" in text
        assert "practically" in text

    def test_explains_why_no_correction_is_applied(self, districts):
        text = " ".join(str(bp.compare_groups(districts).pairwise()).split())
        assert "not a test" in text
        assert "pool=True, not a correction" in text


class TestDeliberateOmissions:
    def test_no_omnibus_bayes_factor_and_it_says_why(self, districts):
        res = bp.compare_groups(districts)
        with pytest.raises(NotImplementedError, match="omnibus"):
            res.bayes_factor()

    def test_sensitivity_explains_there_is_no_prior_to_vary(self, districts):
        res = bp.compare_groups(districts)
        with pytest.raises(NotImplementedError, match="no prior to vary"):
            res.sensitivity()

    def test_footer_suggests_the_methods_that_work(self, districts):
        text = " ".join(str(bp.compare_groups(districts).summary()).split())
        assert ".pairwise()" in text
        assert ".sensitivity()" not in text

    def test_plot_kinds_include_the_group_views(self, districts):
        kinds = bp.compare_groups(districts).plot_kinds()
        assert "forest" in kinds and "pairwise" in kinds


class TestDatasets:
    def test_all_advertised_datasets_load(self):
        pytest.importorskip("pandas")
        for name in bp.datasets.available():
            df = bp.datasets.load(name)
            assert len(df) == bp.datasets.DATASETS[name].rows

    def test_documented_columns_are_present(self):
        pytest.importorskip("pandas")
        for name in bp.datasets.available():
            df = bp.datasets.load(name)
            documented = set(bp.datasets.DATASETS[name].columns)
            assert documented == set(df.columns), name

    def test_unknown_dataset_lists_the_options(self):
        with pytest.raises(ValueError, match="Available:"):
            bp.datasets.load("evictions")

    def test_describe_covers_every_dataset(self):
        for name in bp.datasets.available():
            assert name in bp.datasets.describe(name)
        assert all(n in bp.datasets.describe() for n in bp.datasets.available())

    @pytest.mark.pedagogy
    def test_collisions_supports_the_two_group_comparison(self):
        pytest.importorskip("pandas")
        df = bp.datasets.load_collisions()
        counts = df.groupby("county").alcohol_involved.agg(["sum", "size"])
        res = bp.compare_proportions(
            counts["sum"].to_numpy(),
            counts["size"].to_numpy(),
            labels=list(counts.index),
        )
        # San Diego's alcohol-involvement rate is higher, clearly.
        assert res.probability(">", 0) > 0.99

    @pytest.mark.pedagogy
    def test_parcels_supports_the_pooling_lesson(self):
        pytest.importorskip("pandas")
        df = bp.datasets.load_parcels()
        groups = {
            name: sub.assessed_value.to_numpy()
            for name, sub in df.groupby("construction")
        }
        res = bp.compare_groups(groups, unit="dollars", pool=True)
        assert len(res.components) == 6
        assert res.pooling["tau2"] > 0
