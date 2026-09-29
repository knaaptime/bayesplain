"""Tests for prior resolution: every accepted spelling, and every refusal."""

import pytest

import bayesplain as bp
from bayesplain import priors

# (resolver, default name, a preset name, a valid custom spec, a bad spec)
RESOLVERS = [
    (priors.resolve_proportion, "uninformed", "skeptical", (2, 5), "nonsense"),
    (priors.resolve_table, "uninformed", "gentle", 3.0, "nonsense"),
    (priors.resolve_effect_size, "conventional", "generous", 0.4, "nonsense"),
    (priors.resolve_correlation, "uninformed", "modest", 0.8, "nonsense"),
    (priors.resolve_rate, "jeffreys", "uninformed", (2, 1.5), "nonsense"),
]


@pytest.mark.parametrize("resolve, default, preset, custom, bad", RESOLVERS)
class TestResolvers:
    def test_none_gives_the_default(self, resolve, default, preset, custom, bad):
        assert resolve(None).name == default

    def test_presets_are_case_insensitive(self, resolve, default, preset, custom, bad):
        assert resolve(f"  {preset.upper()} ").name == preset

    def test_an_instance_passes_through(self, resolve, default, preset, custom, bad):
        prior = resolve(preset)
        assert resolve(prior) is prior

    def test_custom_values_are_labelled_custom(
        self, resolve, default, preset, custom, bad
    ):
        prior = resolve(custom)
        assert prior.name == "custom"
        assert "—" in prior.label
        assert repr(prior).startswith(type(prior).__name__)

    def test_unknown_names_list_the_options(
        self, resolve, default, preset, custom, bad
    ):
        with pytest.raises(ValueError, match="Available presets"):
            resolve(bad)

    def test_unreadable_specs_are_refused(self, resolve, default, preset, custom, bad):
        with pytest.raises(ValueError, match="could not read"):
            resolve(object())


@pytest.mark.parametrize(
    "build",
    [
        lambda: priors.BetaPrior(0, 1),
        lambda: priors.ConcentrationPrior(-1),
        lambda: priors.EffectSizePrior(0),
        lambda: priors.CorrelationPrior(0),
        lambda: priors.RatePrior(0, 1),
        lambda: priors.RatePrior(1, -1),
    ],
)
def test_nonpositive_parameters_are_refused(build):
    with pytest.raises(ValueError, match="positive"):
        build()


def test_asymmetric_beta_has_no_table_equivalent():
    with pytest.raises(ValueError, match="symmetric"):
        priors.resolve_table(priors.BetaPrior(2, 5))


def test_symmetric_beta_carries_over_to_a_table():
    table = priors.resolve_table(priors.resolve_proportion("gentle"))
    assert table.a == 2.0 and table.name == "gentle"


def test_densities_exist_only_where_they_mean_something():
    assert priors.resolve_proportion("gentle").dist() is not None
    assert priors.resolve_correlation("modest").dist() is not None
    assert priors.resolve_table("gentle").dist() is None
    assert priors.resolve_effect_size("modest").dist() is None
    assert priors.resolve_rate("jeffreys").dist() is None
    assert priors.from_previous_period(30, 5).dist() is not None


def test_rate_prior_mean():
    assert priors.resolve_rate("jeffreys").prior_mean == float("inf")
    assert priors.from_previous_period(30, 5).prior_mean == pytest.approx(6.1)


def test_describe_and_available():
    assert priors.available() == list(priors.PROPORTION_PRIORS)
    assert "Beta(10, 10)" in priors.describe()
    assert priors.describe((2, 5)).startswith("custom:")


def test_previous_study_validates_its_counts():
    with pytest.raises(ValueError):
        bp.priors.from_previous_study(successes=10, n=5)
