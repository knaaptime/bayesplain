"""Two things moving together, and how much of that we actually know.

The frequentist counterpart is Pearson's r with its p-value (week 8). The
Bayesian version treats the correlation as a quantity to be estimated, which
turns "is there a relationship" into "how strong is it, and how sure are we" --
a question that has an answer a planner can use.

Two warnings belong with this analysis and nowhere else in the course, so they
are attached to every result as notes rather than left to the lecture.

**Ecological correlation.** A correlation computed on tracts, districts, or
counties is a statement about *those areas*, not about the people in them.
Robinson (1950) showed the two can differ in sign, not merely in size. Nothing
in a correlation coefficient warns you when you have crossed that line.

**The modifiable areal unit problem.** The same underlying data, aggregated to
different boundaries, produces different correlations. Redraw the districts and
the number changes. This is not measurement error to be averaged away; it means
the correlation is partly a property of the geography you chose.

And the ordinary caveat, which the package will not stop repeating: nothing
here makes a finding causal.
"""

from __future__ import annotations

import numpy as np

from . import frequentist, priors
from ._config import get_draws, make_rng
from .core import correlation as core_corr
from .core import grid as grid_utils
from .result import Result

__all__ = ["correlation"]


def correlation(
    x,
    y,
    prior="uninformed",
    labels=None,
    aggregated: bool = False,
    n_draws: int | None = None,
    seed="unset",
) -> Result:
    """Estimate the correlation between two variables.

    The posterior is computed from the exact sampling density of the
    correlation coefficient, evaluated on a grid and sampled by inverting its
    cumulative distribution. Not a conjugate update and not MCMC -- a
    one-dimensional integral, which is all this problem needs.

    Parameters
    ----------
    x, y : array_like
        Paired observations. Rows where either value is missing are dropped.
    prior : str, float, or CorrelationPrior, default 'uninformed'
        Stretched-beta prior width on the correlation. The default is flat on
        (-1, 1). See :func:`bayesplain.priors.describe`.
    labels : sequence of str, optional
        Names of the two variables, used in output.
    aggregated : bool, default False
        Set to ``True`` when the rows are geographic areas rather than
        individuals. This adds the ecological-correlation and modifiable-areal-
        unit warnings to the result, because a correlation between tract
        averages is not a correlation between people.
    n_draws : int, optional
        Number of draws. Defaults to the package setting.
    seed : int or None, optional
        Seed for the draws.

    Returns
    -------
    Result
        Call ``.summary()`` for the full report.

    Examples
    --------
    >>> import numpy as np, bayesplain as bp
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=60)
    >>> y = 0.5 * x + rng.normal(size=60)
    >>> res = bp.correlation(x, y, labels=["density", "transit use"])
    >>> res.probability(">", 0) > 0.95
    True
    """
    a, b = core_corr.validate_pair(x, y)
    n = a.size
    resolved = priors.resolve_correlation(prior)
    n_draws = get_draws() if n_draws is None else int(n_draws)
    rng = make_rng(seed)

    if labels is None:
        labels = ("x", "y")
    labels = tuple(str(item) for item in labels)
    if len(labels) != 2:
        raise ValueError(f"labels needs exactly two names, got {len(labels)}.")

    twin = frequentist.correlation(a, b)
    r = twin.statistic

    rho_grid, density = core_corr.posterior_on_grid(r, n, kappa=resolved.kappa)
    draws = grid_utils.sample_from_grid(density, rho_grid, size=n_draws, rng=rng)
    log_bf10 = core_corr.log_bayes_factor(r, n, kappa=resolved.kappa)

    notes = _build_notes(n, aggregated)

    def _refit(spec):
        return correlation(
            a,
            b,
            prior=spec,
            labels=labels,
            aggregated=aggregated,
            n_draws=n_draws,
            seed=seed,
        )

    return Result(
        quantity=f"correlation between {labels[0]} and {labels[1]}",
        draws=draws,
        posterior=None,  # grid-based; no closed-form scipy distribution
        prior=resolved,
        subject="correlation",
        analysis="correlation",
        frequentist=twin,
        log_bf10=log_bf10,
        bf_alternative="a relationship",
        bf_caveat=(
            "Compares 'the two variables are related' against 'they are "
            "completely unrelated'. A correlation of exactly zero is rarely "
            "what anyone believes, and this number moves with the prior width "
            "— run .sensitivity()."
        ),
        unit="",
        display_scale=1.0,
        decimals=3,
        direction_reference=0.0,
        components=None,
        custom_plots={"scatter": _scatter_plot(a, b, labels)},
        refit=_refit,
        prior_ladder=priors.CORRELATION_SENSITIVITY_LADDER,
        n_draws=n_draws,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_notes(n: int, aggregated: bool) -> list[str]:
    """Caveats that belong with every correlation, and some with only a few."""
    notes = [
        "A correlation is not a cause. Nothing in this analysis rules out a "
        "third variable driving both, or the arrow pointing the other way."
    ]
    if aggregated:
        notes.append(
            "These rows are areas, not people. A correlation between area "
            "averages can differ from the correlation among individuals, "
            "sometimes in sign (Robinson 1950), and it will change if the "
            "boundaries are redrawn. Report the geography you used."
        )
    if n < 25:
        notes.append(
            f"only {n} pairs, so the interval is wide and the posterior is "
            "noticeably asymmetric — read the interval, not the point estimate"
        )
    return notes


def _scatter_plot(a, b, labels):
    """Build the scatter-plot callable this result exposes as kind='scatter'."""

    def draw(ax):
        ax.scatter(a, b, s=22, alpha=0.55, color="#4a4e69", edgecolor="none")
        # A least-squares line, purely to guide the eye. The posterior above
        # is about the correlation, not about this line's slope.
        if np.ptp(a) > 0:
            slope, intercept = np.polyfit(a, b, 1)
            span = np.array([a.min(), a.max()])
            ax.plot(span, slope * span + intercept, color="#c9184a", lw=1.5)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.spines[["top", "right"]].set_visible(False)
        return ax

    return draw
