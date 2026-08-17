"""Named priors, so nobody has to type a shape parameter to get started.

A student who has never seen a Beta distribution should still be able to state
an assumption and be held to it. So priors are addressed by name -- ``"gentle"``,
``"skeptical"`` -- with the numbers they resolve to printed in every summary
and one keystroke away for anyone who wants them.

The names are chosen to describe *what the assumption says*, not what family it
belongs to. ``"uninformed"`` means every rate between 0 and 1 starts out
equally plausible. ``"skeptical"`` means large effects are assumed unlikely
until the data insist. Both are defensible; neither is neutral, and the summary
output says so.

Examples
--------
>>> import bayesplain as bf
>>> bf.priors.resolve_proportion("gentle")
BetaPrior(a=2.0, b=2.0, name='gentle')
>>> bf.priors.resolve_proportion((0.5, 0.5)).name
'custom'
>>> bf.priors.describe("gentle").startswith("gentle: mild pull toward the middle")
True
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy import stats

__all__ = [
    "BetaPrior",
    "ConcentrationPrior",
    "EffectSizePrior",
    "EFFECT_SIZE_PRIORS",
    "EFFECT_SENSITIVITY_LADDER",
    "resolve_effect_size",
    "CorrelationPrior",
    "CORRELATION_PRIORS",
    "CORRELATION_SENSITIVITY_LADDER",
    "resolve_correlation",
    "PROPORTION_PRIORS",
    "TABLE_PRIORS",
    "SENSITIVITY_LADDER",
    "resolve_proportion",
    "resolve_table",
    "from_previous_study",
    "describe",
    "available",
]


# ---------------------------------------------------------------------------
# Prior objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BetaPrior:
    """A Beta prior over a proportion, carrying its own explanation.

    Attributes
    ----------
    a, b : float
        Shape parameters. Read them as "prior successes" and "prior failures":
        ``Beta(2, 2)`` behaves like having already seen one success and one
        failure before the study began.
    name : str
        Short label used in printed output.
    rationale : str
        Plain-English statement of what assuming this prior commits you to.
    """

    a: float
    b: float
    name: str = "custom"
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.a <= 0 or self.b <= 0:
            raise ValueError(
                f"Beta prior shapes must be positive, got a={self.a}, b={self.b}."
            )

    @property
    def label(self) -> str:
        """Name and parameters together, for printing in a summary table."""
        return f"{self.name} — Beta({self.a:g}, {self.b:g})"

    @property
    def prior_mean(self) -> float:
        """The rate this prior expects before seeing any data."""
        return self.a / (self.a + self.b)

    @property
    def prior_weight(self) -> float:
        """Roughly how many observations of data this prior is worth."""
        return self.a + self.b

    def dist(self):
        """Return the prior as a frozen scipy distribution."""
        return stats.beta(self.a, self.b)

    def __repr__(self) -> str:
        return f"BetaPrior(a={self.a!r}, b={self.b!r}, name={self.name!r})"


# ---------------------------------------------------------------------------
# The named presets
# ---------------------------------------------------------------------------

PROPORTION_PRIORS: dict[str, BetaPrior] = {
    "uninformed": BetaPrior(
        1.0,
        1.0,
        "uninformed",
        "every rate between 0% and 100% starts out equally plausible",
    ),
    "jeffreys": BetaPrior(
        0.5,
        0.5,
        "jeffreys",
        "the conventional reference choice; slightly favours rates near 0 "
        "and 1, and is invariant to how the question is parameterised",
    ),
    "gentle": BetaPrior(
        2.0,
        2.0,
        "gentle",
        "mild pull toward the middle, worth about 2 observations either way; "
        "rules out nothing but keeps tiny samples from implying certainty",
    ),
    "skeptical": BetaPrior(
        10.0,
        10.0,
        "skeptical",
        "assumes the rate is probably near 50% and takes real data to move; "
        "worth about 20 observations",
    ),
}

#: Presets ordered by how much they constrain the answer, used by
#: ``Result.sensitivity()`` to show how a conclusion moves with the assumption.
SENSITIVITY_LADDER: tuple[str, ...] = (
    "uninformed",
    "jeffreys",
    "gentle",
    "skeptical",
)


@dataclass(frozen=True)
class ConcentrationPrior:
    """A symmetric Dirichlet prior over the cells of a contingency table.

    The same preset names as :class:`BetaPrior`, resolving to the concentration
    ``a`` that Gunel and Dickey's Bayes factors take. The correspondence is
    exact rather than analogous: for a 2x2 table built from successes and
    failures, concentration ``a`` on the columns *is* a ``Beta(a, a)`` prior on
    each group's rate.

    Larger values pull the alternative's predictions toward the null. Jamil et
    al. (2017) put it well: ``a = 10`` behaves like a flat ``a = 1`` prior that
    has already been updated with nine hypothetical observations in every cell.

    Attributes
    ----------
    a : float
        The concentration.
    name : str
        Short label used in printed output.
    rationale : str
        Plain-English statement of what this assumption commits you to.
    """

    a: float
    name: str = "custom"
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.a <= 0:
            raise ValueError(f"concentration must be positive, got {self.a}.")

    @property
    def label(self) -> str:
        """Name and parameter together, for printing in a summary table."""
        return f"{self.name} — Dirichlet({self.a:g})"

    def dist(self):
        """Return ``None``: a table prior has no one-dimensional density.

        The prior lives on the simplex of cell probabilities, so there is no
        single curve to overlay on a posterior plot the way there is for a
        rate. Plotting code checks for ``None`` and skips the prior layer.
        """
        return None

    def __repr__(self) -> str:
        return f"ConcentrationPrior(a={self.a!r}, name={self.name!r})"


@dataclass(frozen=True)
class EffectSizePrior:
    """A Cauchy prior on standardised effect size, for means and correlations.

    This one behaves differently from the other two, and the difference is
    worth stating plainly wherever it appears: **it does not touch the
    estimate**. The posterior for a mean uses the standard reference prior and
    is fixed by the data alone. This prior enters only the Bayes factor, where
    it says how large an effect you would expect *if* there is one.

    That makes the means analyses a clean demonstration of the package's main
    claim. Vary this prior across its whole range and the credible interval
    does not move by a single digit, while the Bayes factor moves substantially
    -- because one is locating a value and the other is grading the evidence
    for a model.

    Attributes
    ----------
    scale : float
        Width of the Cauchy. Larger values spread prior mass over bigger
        effects, which costs the alternative when the observed effect is
        small.
    name : str
        Short label used in printed output.
    rationale : str
        Plain-English statement of what this assumption commits you to.
    """

    scale: float
    name: str = "custom"
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError(f"Cauchy prior scale must be positive, got {self.scale}.")

    @property
    def label(self) -> str:
        """Name and parameter together, for printing in a summary table."""
        return f"{self.name} — Cauchy({self.scale:g})"

    def dist(self):
        """Return the prior on standardised effect size as a frozen scipy dist.

        Note this lives on the *standardised* scale, not the scale of the data,
        so it cannot be overlaid directly on a posterior for a difference in
        dollars or minutes. Plotting code treats it accordingly.
        """
        return None

    def __repr__(self) -> str:
        return f"EffectSizePrior(scale={self.scale!r}, name={self.name!r})"


#: Effect-size priors. These affect the Bayes factor only, never the estimate.
EFFECT_SIZE_PRIORS: dict[str, EffectSizePrior] = {
    "modest": EffectSizePrior(
        0.5,
        "modest",
        "expects small to medium effects; appropriate when big swings would "
        "be surprising in this setting",
    ),
    "conventional": EffectSizePrior(
        0.707,
        "conventional",
        "the standard default, matching R's BayesFactor package; a medium "
        "effect is the single most likely size",
    ),
    "uninformed": EffectSizePrior(
        1.0,
        "uninformed",
        "large effects are as plausible as small ones going in",
    ),
    "generous": EffectSizePrior(
        1.414,
        "generous",
        "expects a large effect if there is one at all; makes the Bayes "
        "factor harder to move for a small observed difference",
    ),
}

#: Effect-size presets ordered by width, for ``Result.sensitivity()``.
EFFECT_SENSITIVITY_LADDER: tuple[str, ...] = (
    "modest",
    "conventional",
    "uninformed",
    "generous",
)


#: Table priors, matched one-for-one with the proportion presets above.
TABLE_PRIORS: dict[str, ConcentrationPrior] = {
    "uninformed": ConcentrationPrior(
        1.0,
        "uninformed",
        "every combination of cell probabilities starts out equally plausible",
    ),
    "jeffreys": ConcentrationPrior(
        0.5,
        "jeffreys",
        "the conventional reference choice; puts slightly more weight on "
        "tables concentrated in a few cells",
    ),
    "gentle": ConcentrationPrior(
        2.0,
        "gentle",
        "worth about one extra observation per cell; keeps sparse tables from "
        "implying certainty",
    ),
    "skeptical": ConcentrationPrior(
        10.0,
        "skeptical",
        "worth about nine extra observations per cell, which pulls the "
        "association toward zero unless the data insist otherwise",
    ),
}


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def resolve_proportion(prior) -> BetaPrior:
    """Turn whatever the user passed as ``prior=`` into a ``BetaPrior``.

    Accepts a preset name, an ``(a, b)`` pair, an existing ``BetaPrior``, or
    ``None`` for the default.

    Parameters
    ----------
    prior : str, tuple, BetaPrior, or None
        The prior specification. ``None`` resolves to ``"uninformed"``.

    Returns
    -------
    BetaPrior
        The resolved prior.

    Raises
    ------
    ValueError
        If a name is given that is not a known preset, or a tuple of the wrong
        length.
    """
    if prior is None:
        return PROPORTION_PRIORS["uninformed"]
    if isinstance(prior, BetaPrior):
        return prior
    if isinstance(prior, str):
        key = prior.strip().lower()
        if key not in PROPORTION_PRIORS:
            options = ", ".join(repr(k) for k in PROPORTION_PRIORS)
            raise ValueError(
                f"unknown prior {prior!r}. Available presets: {options}. You "
                "can also pass an (a, b) pair, or "
                "bayesplain.priors.from_previous_study(...)."
            )
        return PROPORTION_PRIORS[key]
    try:
        a, b = prior
    except (TypeError, ValueError) as err:
        raise ValueError(
            f"could not read {prior!r} as a prior. Pass a preset name, an "
            "(a, b) pair, or a BetaPrior."
        ) from err
    return BetaPrior(float(a), float(b), "custom", f"custom Beta({a:g}, {b:g}) prior")


def resolve_table(prior) -> ConcentrationPrior:
    """Turn whatever the user passed as ``prior=`` into a ``ConcentrationPrior``.

    Accepts a preset name, a positive number, an existing
    :class:`ConcentrationPrior`, or ``None`` for the default. A
    :class:`BetaPrior` is also accepted when its two shapes are equal, so that
    the same ``prior=`` argument can be handed to a proportion analysis and a
    table analysis and mean the same thing.

    Parameters
    ----------
    prior : str, float, ConcentrationPrior, BetaPrior, or None
        The prior specification. ``None`` resolves to ``"uninformed"``.

    Returns
    -------
    ConcentrationPrior
        The resolved prior.

    Raises
    ------
    ValueError
        If a name is not a known preset, or a ``BetaPrior`` with unequal shapes
        is given, since that has no symmetric table equivalent.
    """
    if prior is None:
        return TABLE_PRIORS["uninformed"]
    if isinstance(prior, ConcentrationPrior):
        return prior
    if isinstance(prior, BetaPrior):
        if prior.a != prior.b:
            raise ValueError(
                f"a table prior must be symmetric across cells, but "
                f"Beta({prior.a:g}, {prior.b:g}) is not. Pass a single "
                "concentration, or one of: "
                f"{', '.join(repr(k) for k in TABLE_PRIORS)}."
            )
        return ConcentrationPrior(prior.a, prior.name, prior.rationale)
    if isinstance(prior, str):
        key = prior.strip().lower()
        if key not in TABLE_PRIORS:
            options = ", ".join(repr(k) for k in TABLE_PRIORS)
            raise ValueError(
                f"unknown prior {prior!r}. Available presets: {options}. You "
                "can also pass a single positive number."
            )
        return TABLE_PRIORS[key]
    try:
        a = float(prior)
    except (TypeError, ValueError) as err:
        raise ValueError(
            f"could not read {prior!r} as a table prior. Pass a preset name or "
            "a single positive number."
        ) from err
    return ConcentrationPrior(a, "custom", f"custom Dirichlet({a:g}) prior")


def from_previous_study(successes: float, n: float) -> BetaPrior:
    """Build a prior from an earlier study's counts.

    Treats the earlier study as data observed under a flat prior, giving
    ``Beta(1 + successes, 1 + failures)``. This is the honest way to say "we
    already know something about this rate", and it makes the strength of the
    assumption legible: a prior study of 80 cases is worth 80 cases, no more.

    Parameters
    ----------
    successes : int
        Successes in the earlier study.
    n : int
        Trials in the earlier study.

    Returns
    -------
    BetaPrior
        The resulting prior.

    Examples
    --------
    >>> from_previous_study(successes=12, n=80).prior_mean
    0.15853658536585366
    """
    from .core.beta_binomial import validate_counts

    successes, n = validate_counts(successes, n)
    return BetaPrior(
        1.0 + successes,
        1.0 + (n - successes),
        "prior study",
        f"an earlier study that saw {successes} of {n} "
        f"({successes / n:.1%}), treated as evidence in hand",
    )


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


def available() -> list[str]:
    """List the preset prior names.

    Returns
    -------
    list of str
        Names accepted by ``prior=``.
    """
    return list(PROPORTION_PRIORS)


def describe(prior=None) -> str:
    """Explain in one line what a prior assumes.

    Parameters
    ----------
    prior : str, tuple, BetaPrior, or None, optional
        Prior specification. ``None`` describes every preset.

    Returns
    -------
    str
        A single line for a specific prior, or one line per preset.
    """
    if prior is None:
        return "\n".join(
            f"{p.name:<12} Beta({p.a:g}, {p.b:g})  {p.rationale}"
            for p in PROPORTION_PRIORS.values()
        )
    resolved = resolve_proportion(prior)
    rationale = resolved.rationale or f"Beta({resolved.a:g}, {resolved.b:g})"
    return f"{resolved.name}: {rationale}"


def resolve_effect_size(prior) -> EffectSizePrior:
    """Turn whatever the user passed as ``prior=`` into an ``EffectSizePrior``.

    Accepts a preset name, a positive number read as the Cauchy scale, an
    existing :class:`EffectSizePrior`, or ``None`` for the conventional
    default.

    Parameters
    ----------
    prior : str, float, EffectSizePrior, or None
        The prior specification. ``None`` resolves to ``"conventional"``.

    Returns
    -------
    EffectSizePrior
        The resolved prior.

    Raises
    ------
    ValueError
        If a name is not a known preset, or the value is not a positive
        number.
    """
    if prior is None:
        return EFFECT_SIZE_PRIORS["conventional"]
    if isinstance(prior, EffectSizePrior):
        return prior
    if isinstance(prior, str):
        key = prior.strip().lower()
        if key not in EFFECT_SIZE_PRIORS:
            options = ", ".join(repr(k) for k in EFFECT_SIZE_PRIORS)
            raise ValueError(
                f"unknown prior {prior!r}. Available presets: {options}. You "
                "can also pass a positive number, read as the Cauchy scale on "
                "standardised effect size."
            )
        return EFFECT_SIZE_PRIORS[key]
    try:
        scale = float(prior)
    except (TypeError, ValueError) as err:
        raise ValueError(
            f"could not read {prior!r} as an effect-size prior. Pass a preset "
            "name or a positive number."
        ) from err
    return EffectSizePrior(scale, "custom", f"custom Cauchy({scale:g}) prior")


@dataclass(frozen=True)
class CorrelationPrior:
    """A stretched-beta prior on a correlation coefficient.

    Parameterised by a width ``kappa``: the prior density is proportional to
    ``(1 - rho**2) ** (1/kappa - 1)`` on (-1, 1). At ``kappa = 1`` it is flat,
    meaning every correlation from -1 to 1 is equally plausible before seeing
    data. Below 1 it concentrates near zero; above 1 it pushes mass toward the
    extremes.

    Attributes
    ----------
    kappa : float
        Prior width.
    name : str
        Short label used in printed output.
    rationale : str
        Plain-English statement of what this assumption commits you to.
    """

    kappa: float
    name: str = "custom"
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.kappa <= 0:
            raise ValueError(f"kappa must be positive, got {self.kappa}.")

    @property
    def label(self) -> str:
        """Name and parameter together, for printing in a summary table."""
        return f"{self.name} — width {self.kappa:g}"

    def dist(self):
        """Return the prior as a frozen scipy distribution on (-1, 1)."""
        shape = 1.0 / self.kappa
        return stats.beta(shape, shape, loc=-1.0, scale=2.0)

    def __repr__(self) -> str:
        return f"CorrelationPrior(kappa={self.kappa!r}, name={self.name!r})"


#: Correlation priors, ordered from most to least concentrated near zero.
CORRELATION_PRIORS: dict[str, CorrelationPrior] = {
    "concentrated": CorrelationPrior(
        0.3,
        "concentrated",
        "assumes any real relationship is weak; takes strong data to move",
    ),
    "modest": CorrelationPrior(
        0.5,
        "modest",
        "expects a weak to moderate relationship if there is one at all",
    ),
    "uninformed": CorrelationPrior(
        1.0,
        "uninformed",
        "every correlation between -1 and 1 starts out equally plausible",
    ),
    "generous": CorrelationPrior(
        1.5,
        "generous",
        "expects a strong relationship if any, which makes the Bayes factor "
        "harder to move for a weak observed correlation",
    ),
}

#: Correlation presets ordered by width, for ``Result.sensitivity()``.
CORRELATION_SENSITIVITY_LADDER: tuple[str, ...] = (
    "concentrated",
    "modest",
    "uninformed",
    "generous",
)


def resolve_correlation(prior) -> CorrelationPrior:
    """Turn whatever the user passed as ``prior=`` into a ``CorrelationPrior``.

    Accepts a preset name, a positive number read as ``kappa``, an existing
    :class:`CorrelationPrior`, or ``None`` for the flat default.

    Parameters
    ----------
    prior : str, float, CorrelationPrior, or None
        The prior specification. ``None`` resolves to ``"uninformed"``.

    Returns
    -------
    CorrelationPrior
        The resolved prior.

    Raises
    ------
    ValueError
        If a name is not a known preset, or the value is not a positive
        number.
    """
    if prior is None:
        return CORRELATION_PRIORS["uninformed"]
    if isinstance(prior, CorrelationPrior):
        return prior
    if isinstance(prior, str):
        key = prior.strip().lower()
        if key not in CORRELATION_PRIORS:
            options = ", ".join(repr(k) for k in CORRELATION_PRIORS)
            raise ValueError(
                f"unknown prior {prior!r}. Available presets: {options}. You "
                "can also pass a positive number, read as the prior width."
            )
        return CORRELATION_PRIORS[key]
    try:
        kappa = float(prior)
    except (TypeError, ValueError) as err:
        raise ValueError(
            f"could not read {prior!r} as a correlation prior. Pass a preset "
            "name or a positive number."
        ) from err
    return CorrelationPrior(kappa, "custom", f"custom width {kappa:g} prior")
