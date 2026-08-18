"""Tools that exist only to be taught with.

Nothing in this module belongs in a research package. Each function here shows a
mechanism that the analysis functions deliberately hide, so that a class can see
it work once before trusting a one-liner that does it for them.

Four things:

:func:`natural_frequencies`
    Bayes' rule as a grid of whole numbers, with no probability anywhere in the
    output. For the base-rate lesson, before any formula appears.
:func:`grid_posterior`
    Prior times likelihood, normalised, with the arithmetic laid out in a table
    you can read across. The mechanism behind every posterior in the package.
:func:`sequential`
    The posterior after each chunk of data, for watching an interval tighten as
    evidence accumulates.
:func:`precision_planning`
    How many observations until the interval is narrow enough to act on. The
    honest replacement for a power calculation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from . import priors
from .core import beta_binomial, intervals
from .core import grid as grid_utils
from .result import _Report, _wrap

__all__ = [
    "natural_frequencies",
    "binomial_likelihood",
    "grid_posterior",
    "sequential",
    "precision_planning",
    "NaturalFrequencies",
    "GridPosterior",
    "SequentialUpdate",
    "PrecisionPlan",
]


# ---------------------------------------------------------------------------
# Bayes' rule as counting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NaturalFrequencies:
    """A 2x2 grid of whole counts, and what it implies.

    :ivar table: ``[[true positive, false negative], [false positive, true
        negative]]`` as integers.
    :ivar n: Total number of imagined cases.
    :ivar base_rate: The prevalence that was assumed.
    :ivar sensitivity: The detection rate that was assumed.
    :ivar specificity: The correct-clearance rate that was assumed.
    :ivar case_label: What one row of the imagined data is.
    :ivar condition_label: The thing being detected.
    :ivar flag_label: What the detector does when it fires.
    """

    table: np.ndarray
    n: int
    base_rate: float
    sensitivity: float
    specificity: float
    case_label: str = "cases"
    condition_label: str = "the condition"
    flag_label: str = "flagged"

    @property
    def true_positives(self) -> int:
        """Cases that have the condition and were flagged."""
        return int(self.table[0, 0])

    @property
    def false_negatives(self) -> int:
        """Cases that have the condition and were missed."""
        return int(self.table[0, 1])

    @property
    def false_positives(self) -> int:
        """Cases that do not have the condition but were flagged anyway."""
        return int(self.table[1, 0])

    @property
    def true_negatives(self) -> int:
        """Cases that do not have the condition and were correctly cleared."""
        return int(self.table[1, 1])

    @property
    def flagged(self) -> int:
        """Total number of cases the detector flagged."""
        return self.true_positives + self.false_positives

    @property
    def with_condition(self) -> int:
        """Total number of cases that actually have the condition."""
        return self.true_positives + self.false_negatives

    @property
    def posterior_given_flag(self) -> float:
        """Share of flagged cases that actually have the condition.

        This is the number the exercise is about, and it is obtained by
        dividing one count by another -- no formula required.
        """
        return self.true_positives / self.flagged if self.flagged else float("nan")

    @property
    def posterior_given_no_flag(self) -> float:
        """Share of un-flagged cases that have the condition anyway."""
        cleared = self.false_negatives + self.true_negatives
        return self.false_negatives / cleared if cleared else float("nan")

    def bayes_rule_check(self) -> float:
        """Compute the same answer from the formula, for comparison.

        Returns
        -------
        float
            ``P(condition | flag)`` by Bayes' rule. Agrees with
            :attr:`posterior_given_flag` up to the rounding needed to keep the
            grid in whole numbers, which is the point: the counting and the
            formula are the same operation.
        """
        hit = self.base_rate * self.sensitivity
        false_alarm = (1.0 - self.base_rate) * (1.0 - self.specificity)
        return hit / (hit + false_alarm)

    def summary(self) -> _Report:
        """Print the grid and read the answer off it.

        Returns
        -------
        _Report
            Printable text with no probability in the table itself.
        """
        flagged_col = self.flag_label
        cleared_col = f"not {self.flag_label}"

        # The condition is named once in a caption rather than repeated down
        # the stub, so a long label cannot push the table past terminal width.
        stub = 7
        col = max(len(flagged_col), len(cleared_col), 9) + 3
        line = "  " + "-" * (stub + 3 * col)

        rows = [
            f"IMAGINE {self.n:,} {self.case_label.upper()}",
            "",
            "Nothing below is a probability. Every number is a count of "
            f"{self.case_label}.",
            f"Rows: does it have {self.condition_label}?",
            "",
            f"  {'':<{stub}}{flagged_col:>{col}}{cleared_col:>{col}}{'total':>{col}}",
            f"  {'yes':<{stub}}{self.true_positives:>{col},}"
            f"{self.false_negatives:>{col},}{self.with_condition:>{col},}",
            f"  {'no':<{stub}}{self.false_positives:>{col},}"
            f"{self.true_negatives:>{col},}"
            f"{self.n - self.with_condition:>{col},}",
            line,
            f"  {'total':<{stub}}{self.flagged:>{col},}"
            f"{self.n - self.flagged:>{col},}{self.n:>{col},}",
            "",
        ]
        rows += _wrap(
            f"Of the {self.flagged:,} {self.case_label} the tool "
            f"{self.flag_label}, {self.true_positives:,} actually have "
            f"{self.condition_label}. So a {self.flag_label} case has "
            f"{self.condition_label} {self.true_positives:,} times out of "
            f"{self.flagged:,} — about {self.posterior_given_flag:.0%}.",
            prefix="  ",
        )
        rows += [""]
        rows += _wrap(
            f"Before the tool looked at anything, {self.with_condition:,} of "
            f"{self.n:,} {self.case_label} had {self.condition_label}: about "
            f"{self.base_rate:.0%}. Being {self.flag_label} moves that to "
            f"{self.posterior_given_flag:.0%}.",
            prefix="  ",
        )
        rows += [""]
        rows += _wrap(self._moral(), prefix="  ")
        return _Report("\n".join(rows))

    def _moral(self) -> str:
        """State what this particular set of numbers should teach."""
        post = self.posterior_given_flag
        if post < 0.5:
            return (
                "Read that again: even after the tool fires, it is still more "
                "likely than not that the case is fine. A detector can be "
                "accurate and still be wrong most of the time it fires, and "
                "the reason is the base rate — there are simply far more cases "
                "without the condition for it to make mistakes on. Anyone "
                "acting on the flag alone is acting on a coin flip or worse."
            )
        if post < 0.8:
            return (
                "The flag is informative but not decisive: a meaningful share "
                "of flagged cases are fine. Whether that is good enough "
                "depends entirely on what happens to a case once it is "
                "flagged, which is a policy question and not a statistical one."
            )
        return (
            "Here the flag is strong evidence. Worth noticing what makes it "
            "so: not the detector's accuracy alone, but that accuracy "
            "combined with a base rate high enough that false alarms cannot "
            "swamp the true ones. Change the base rate and the same detector "
            "tells you much less."
        )

    def __repr__(self) -> str:
        return (
            f"<NaturalFrequencies: {self.true_positives:,} of "
            f"{self.flagged:,} {self.flag_label} {self.case_label} have "
            f"{self.condition_label} ({self.posterior_given_flag:.0%}); "
            f"base rate {self.base_rate:.0%}>"
            "\nCall .summary() to see the grid."
        )


def natural_frequencies(
    base_rate: float,
    sensitivity: float,
    specificity: float,
    n: int = 1000,
    case_label: str = "cases",
    condition_label: str = "the condition",
    flag_label: str = "flagged",
) -> NaturalFrequencies:
    """Lay out a detection problem as whole counts instead of probabilities.

    People are far better at reasoning about "144 of these 312" than about
    ``P(A|B)``, and the two are mathematically identical. This builds the grid
    of counts so a class can find the answer by dividing one number by another,
    and only afterwards be told they have just applied Bayes' rule.

    Parameters
    ----------
    base_rate : float
        Share of cases that actually have the condition, in (0, 1).
    sensitivity : float
        Share of cases *with* the condition that get flagged, in (0, 1].
    specificity : float
        Share of cases *without* the condition that correctly do not get
        flagged, in (0, 1].
    n : int, default 1000
        How many cases to imagine. A round number is the point; 1,000 keeps
        every cell a whole number for most realistic inputs.
    case_label : str, default 'cases'
        Plural noun for one row of the imagined data, e.g. ``"permits"``.
    condition_label : str, default 'the condition'
        The thing being detected, e.g. ``"a code violation"``.
    flag_label : str, default 'flagged'
        Past-tense verb for what the detector does, e.g. ``"flagged"``.

    Returns
    -------
    NaturalFrequencies
        Call ``.summary()`` to print the grid.

    Examples
    --------
    An inspection-targeting tool that is 90% accurate both ways, looking for a
    violation that 5% of permits actually have:

    >>> import bayesplain as bp
    >>> grid = bp.teach.natural_frequencies(
    ...     base_rate=0.05, sensitivity=0.90, specificity=0.90,
    ...     n=1000, case_label="permits", condition_label="a violation",
    ... )
    >>> grid.flagged
    140
    >>> round(grid.posterior_given_flag, 3)
    0.321

    Ninety percent accurate both ways, and yet only about a third of the
    permits it flags actually have a violation. Raise the base rate to 16% and
    the same tool gets it right 63% of the time — the detector did not change,
    the world did.

    Notes
    -----
    Counts are rounded so that every row and column sums exactly, which means
    :attr:`NaturalFrequencies.posterior_given_flag` can differ from
    :meth:`NaturalFrequencies.bayes_rule_check` in the third decimal place. That
    discrepancy is worth showing rather than hiding: it is the cost of insisting
    on whole permits, and it is smaller than any decision would turn on.
    """
    for name, value in (
        ("base_rate", base_rate),
        ("sensitivity", sensitivity),
        ("specificity", specificity),
    ):
        if not 0.0 < value <= 1.0:
            raise ValueError(
                f"{name} must be a share between 0 and 1, got {value}. Pass "
                "0.9 rather than 90."
            )
    if base_rate >= 1.0:
        raise ValueError(
            f"base_rate must be below 1, got {base_rate}: if every case has "
            "the condition there is nothing to detect."
        )
    n = int(n)
    if n < 100:
        raise ValueError(
            f"n must be at least 100 for the grid to stay in whole numbers, "
            f"got {n}. 1,000 is the conventional choice."
        )

    with_condition = int(round(n * base_rate))
    without_condition = n - with_condition
    true_positives = int(round(with_condition * sensitivity))
    true_negatives = int(round(without_condition * specificity))

    table = np.array(
        [
            [true_positives, with_condition - true_positives],
            [without_condition - true_negatives, true_negatives],
        ],
        dtype=int,
    )
    if table.sum() != n:  # pragma: no cover - guarded by the arithmetic above
        raise RuntimeError("the grid failed to sum to n; this is a bug.")

    return NaturalFrequencies(
        table=table,
        n=n,
        base_rate=float(base_rate),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        case_label=case_label,
        condition_label=condition_label,
        flag_label=flag_label,
    )


# ---------------------------------------------------------------------------
# The mechanism behind every posterior in the package
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GridPosterior:
    """Prior, likelihood, and posterior over a grid, with the arithmetic shown.

    :ivar grid: The candidate values considered.
    :ivar prior: Prior weight on each, normalised.
    :ivar likelihood: Likelihood of the data at each candidate.
    :ivar unnormalized: ``prior * likelihood``, before dividing through.
    :ivar posterior: The normalised posterior.
    :ivar kind: ``'discrete'`` if the posterior sums to 1, ``'continuous'`` if
        it integrates to 1 over the grid.
    :ivar label: What the grid is a grid of, for printing.
    """

    grid: np.ndarray
    prior: np.ndarray
    likelihood: np.ndarray
    unnormalized: np.ndarray
    posterior: np.ndarray
    kind: str = "discrete"
    label: str = "rate"

    @property
    def total(self) -> float:
        """The normalising constant -- the number everything got divided by."""
        return float(self.unnormalized.sum())

    def mean(self) -> float:
        """Posterior mean.

        Returns
        -------
        float
            A weighted average of the grid values.
        """
        if self.kind == "discrete":
            return float((self.grid * self.posterior).sum())
        return grid_utils.grid_mean(self.posterior, self.grid)

    def interval(self, level: float = 0.95) -> tuple[float, float]:
        """Equal-tailed credible interval from the grid.

        Parameters
        ----------
        level : float, default 0.95
            Probability the interval should contain.

        Returns
        -------
        tuple of float
            Lower and upper bounds.
        """
        tail = (1.0 - level) / 2.0
        if self.kind == "continuous":
            lo, hi = grid_utils.grid_quantile(
                self.posterior, self.grid, [tail, 1.0 - tail]
            )
            return float(lo), float(hi)
        cumulative = np.cumsum(self.posterior)
        lo = self.grid[np.searchsorted(cumulative, tail)]
        hi = self.grid[min(np.searchsorted(cumulative, 1.0 - tail), self.grid.size - 1)]
        return float(lo), float(hi)

    def table(self, max_rows: int = 12) -> _Report:
        """Show the four columns whose arithmetic produces the posterior.

        Parameters
        ----------
        max_rows : int, default 12
            Grid rows to print. A fine grid is summarised rather than dumped.

        Returns
        -------
        _Report
            Printable table: grid, prior, likelihood, their product, and the
            product divided by its own total.
        """
        step = max(1, self.grid.size // max_rows)
        shown = range(0, self.grid.size, step)

        lines = [
            "PRIOR x LIKELIHOOD, THEN DIVIDE BY THE TOTAL",
            "",
            f"{self.label:>12}{'prior':>12}{'likelihood':>14}"
            f"{'prior x lik':>14}{'posterior':>12}",
            "-" * 64,
        ]
        for i in shown:
            lines.append(
                f"{self.grid[i]:>12.4g}{self.prior[i]:>12.4g}"
                f"{self.likelihood[i]:>14.4g}{self.unnormalized[i]:>14.4g}"
                f"{self.posterior[i]:>12.4g}"
            )
        if step > 1:
            lines.append(
                f"  ... {self.grid.size:,} grid points in all, every {step}th shown"
            )
        lines += ["-" * 64]

        check = (
            self.posterior.sum()
            if self.kind == "discrete"
            else float(np.trapezoid(self.posterior, self.grid))
        )
        verb = "sum" if self.kind == "discrete" else "integral"
        lines += [
            f"{'total':>12}{'':>12}{'':>14}{self.total:>14.4g}{check:>12.4g}",
            "",
        ]
        lines += _wrap(
            f"The fourth column is just the second times the third. The fifth "
            f"is the fourth divided by its own total ({self.total:.4g}), which "
            f"is what makes the posterior {verb} to 1. That division is the "
            f"only thing 'normalising' means, and it is the entire mechanism "
            f"behind every posterior in this package.",
            prefix="  ",
        )
        return _Report("\n".join(lines))

    def plot(self, ax=None):
        """Draw prior, likelihood, and posterior on shared axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. Created if omitted.

        Returns
        -------
        matplotlib.axes.Axes
            The axes drawn on.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as err:  # pragma: no cover - environment dependent
            raise ImportError(
                "plotting needs matplotlib. Install it with "
                "`pip install bayesplain[plot]`."
            ) from err
        if ax is None:
            _, ax = plt.subplots(figsize=(7.5, 4.0))

        style = dict(marker="o", ms=4) if self.grid.size <= 30 else {}
        # Prior and likelihood are put on a common scale so the shapes can be
        # compared; only the posterior's height is meaningful.
        ax.plot(
            self.grid,
            self.prior / self.prior.max(),
            color="#9a8c98",
            ls="--",
            lw=1.6,
            label="prior",
            **style,
        )
        ax.plot(
            self.grid,
            self.likelihood / self.likelihood.max(),
            color="#c9184a",
            ls=":",
            lw=1.6,
            label="likelihood",
            **style,
        )
        ax.plot(
            self.grid,
            self.posterior / self.posterior.max(),
            color="#22223b",
            lw=2.2,
            label="posterior",
            **style,
        )
        ax.fill_between(
            self.grid,
            self.posterior / self.posterior.max(),
            color="#4a4e69",
            alpha=0.15,
        )
        ax.set_xlabel(self.label)
        ax.set_ylabel("relative height")
        ax.set_yticks([])
        ax.legend(frameon=False)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_ylim(bottom=0)
        return ax

    def __repr__(self) -> str:
        lo, hi = self.interval()
        return (
            f"<GridPosterior over {self.grid.size} values of {self.label}: "
            f"mean {self.mean():.4g}, 95% interval [{lo:.4g}, {hi:.4g}]>"
            "\nCall .table() to see the arithmetic."
        )


def binomial_likelihood(successes: int, n: int, grid) -> np.ndarray:
    """Likelihood of binomial data at each candidate rate on a grid.

    A convenience so that a first grid-approximation exercise is about the
    update, not about remembering where ``scipy.stats.binom`` lives.

    Parameters
    ----------
    successes : int
        Number of successes observed.
    n : int
        Number of trials.
    grid : array_like
        Candidate rates, in [0, 1].

    Returns
    -------
    ndarray
        ``P(successes out of n | rate)`` at each grid point.
    """
    successes, n = beta_binomial.validate_counts(successes, n)
    grid = np.asarray(grid, dtype=float)
    if np.any(grid < 0) or np.any(grid > 1):
        raise ValueError("a grid of rates must lie in [0, 1].")
    return stats.binom.pmf(successes, n, grid)


def grid_posterior(
    grid,
    prior=None,
    likelihood=None,
    kind: str = "discrete",
    label: str = "rate",
) -> GridPosterior:
    """Multiply prior by likelihood over a grid, then divide by the total.

    The whole of Bayesian updating, with nothing hidden. Every posterior this
    package produces is this operation, either done in closed form or done on a
    finer grid; running it once by hand is what makes the closed forms
    believable rather than magical.

    Parameters
    ----------
    grid : array_like
        The candidate values under consideration. A handful of named
        hypotheses for a first lesson; a few hundred points to approximate a
        continuous posterior.
    prior : array_like, optional
        Relative prior weight on each candidate. Need not sum to 1 -- it is
        normalised here, which is itself worth pointing out. Defaults to flat.
    likelihood : array_like
        Likelihood of the observed data at each candidate. See
        :func:`binomial_likelihood` for the common case.
    kind : {'discrete', 'continuous'}, default 'discrete'
        Whether the grid enumerates distinct hypotheses (posterior sums to 1)
        or approximates a continuous parameter (posterior integrates to 1).
    label : str, default 'rate'
        What the grid is a grid of, for printed output.

    Returns
    -------
    GridPosterior
        Call ``.table()`` to see the arithmetic, ``.plot()`` to see the shapes.

    Examples
    --------
    Four candidate violation rates, and 7 violations found in 20 inspections:

    >>> import bayesplain as bp
    >>> candidates = [0.1, 0.2, 0.3, 0.4]
    >>> post = bp.teach.grid_posterior(
    ...     grid=candidates,
    ...     likelihood=bp.teach.binomial_likelihood(7, 20, candidates),
    ... )
    >>> float(round(post.posterior.sum(), 10))
    1.0
    >>> round(post.mean(), 4)
    0.3278

    A fine grid reproduces the closed form, which is the point of the exercise:

    >>> import numpy as np
    >>> fine = np.linspace(0.001, 0.999, 999)
    >>> post = bp.teach.grid_posterior(
    ...     grid=fine,
    ...     likelihood=bp.teach.binomial_likelihood(7, 20, fine),
    ...     kind="continuous",
    ... )
    >>> exact = bp.proportion(7, 20).point("mean")
    >>> abs(post.mean() - exact) < 1e-4
    True
    """
    grid = np.asarray(grid, dtype=float).ravel()
    if grid.size < 2:
        raise ValueError(f"a grid needs at least 2 candidate values, got {grid.size}.")
    if np.any(np.diff(grid) <= 0):
        raise ValueError(
            "grid values must be strictly increasing, so the table reads in "
            "order and the interval calculation makes sense."
        )
    if kind not in {"discrete", "continuous"}:
        raise ValueError(f"kind must be 'discrete' or 'continuous', got {kind!r}.")
    if likelihood is None:
        raise ValueError(
            "likelihood is required: it is what the data contributes. See "
            "bayesplain.teach.binomial_likelihood for the usual case."
        )

    likelihood = np.asarray(likelihood, dtype=float).ravel()
    prior = (
        np.ones_like(grid) if prior is None else np.asarray(prior, dtype=float).ravel()
    )
    for name, arr in (("prior", prior), ("likelihood", likelihood)):
        if arr.shape != grid.shape:
            raise ValueError(
                f"{name} has {arr.size} values but the grid has {grid.size}; "
                "they must line up one to one."
            )
        if np.any(arr < 0):
            raise ValueError(f"{name} cannot be negative.")

    prior_total = prior.sum()
    if prior_total <= 0:
        raise ValueError("the prior must put positive weight somewhere.")
    prior = prior / prior_total

    unnormalized = prior * likelihood
    total = unnormalized.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError(
            "prior times likelihood is zero everywhere on this grid, so there "
            "is nothing to normalise. Usually this means the grid does not "
            "cover the values the data support."
        )

    if kind == "discrete":
        posterior = unnormalized / total
    else:
        posterior = grid_utils.normalize_log_density(
            np.log(np.where(unnormalized > 0, unnormalized, np.nan)), grid
        )

    return GridPosterior(
        grid=grid,
        prior=prior,
        likelihood=likelihood,
        unnormalized=unnormalized,
        posterior=posterior,
        kind=kind,
        label=label,
    )


# ---------------------------------------------------------------------------
# Watching an interval tighten
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SequentialUpdate:
    """The posterior after each successive chunk of data.

    :ivar checkpoints: Cumulative number of observations at each step.
    :ivar successes: Cumulative successes at each step.
    :ivar posteriors: The Beta posterior at each step.
    :ivar prior: The prior everything started from.
    :ivar level: Credible level used for the reported intervals.
    :ivar label: What is being estimated, for printed output.
    """

    checkpoints: np.ndarray
    successes: np.ndarray
    posteriors: list = field(default_factory=list)
    prior: object = None
    level: float = 0.95
    label: str = "rate"

    @property
    def means(self) -> np.ndarray:
        """Posterior mean after each chunk."""
        return np.array([p.mean() for p in self.posteriors])

    @property
    def bounds(self) -> np.ndarray:
        """Interval bounds after each chunk, shape ``(steps, 2)``."""
        tail = (1.0 - self.level) / 2.0
        return np.array([[p.ppf(tail), p.ppf(1.0 - tail)] for p in self.posteriors])

    @property
    def widths(self) -> np.ndarray:
        """Interval width after each chunk."""
        return self.bounds[:, 1] - self.bounds[:, 0]

    def table(self) -> _Report:
        """Show the estimate and its interval tightening step by step.

        Returns
        -------
        _Report
            Printable table.
        """
        pct = f"{self.level:.0%}"
        lines = [
            f"THE POSTERIOR AFTER EACH CHUNK OF DATA ({self.label})",
            "",
            f"{'seen':>8}{'successes':>12}{'estimate':>12}"
            f"{pct + ' interval':>22}{'width':>10}",
            "-" * 66,
        ]
        bounds, widths, means = self.bounds, self.widths, self.means
        for i, n in enumerate(self.checkpoints):
            span = f"{bounds[i, 0]:.3f} to {bounds[i, 1]:.3f}"
            lines.append(
                f"{int(n):>8,}{int(self.successes[i]):>12,}"
                f"{means[i]:>12.3f}{span:>22}{widths[i]:>10.3f}"
            )
        lines += [
            "",
        ]
        shrink = widths[0] / widths[-1] if widths[-1] > 0 else float("inf")
        growth = self.checkpoints[-1] / self.checkpoints[0]
        lines += _wrap(
            f"The interval narrowed by a factor of {shrink:.1f} while the data "
            f"grew by a factor of {growth:.0f}. That is roughly the square root "
            f"of the data — quadrupling the sample halves the interval, which "
            f"is why the last few observations feel so much less useful than "
            f"the first few, and why 'collect more data' has diminishing "
            f"returns you can actually quantify.",
            prefix="  ",
        )
        return _Report("\n".join(lines))

    def plot(self, ax=None):
        """Draw the estimate and its interval as data accumulates.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. Created if omitted.

        Returns
        -------
        matplotlib.axes.Axes
            The axes drawn on.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as err:  # pragma: no cover - environment dependent
            raise ImportError(
                "plotting needs matplotlib. Install it with "
                "`pip install bayesplain[plot]`."
            ) from err
        if ax is None:
            _, ax = plt.subplots(figsize=(7.5, 4.0))

        bounds = self.bounds
        ax.fill_between(
            self.checkpoints,
            bounds[:, 0],
            bounds[:, 1],
            color="#4a4e69",
            alpha=0.20,
            label=f"{self.level:.0%} credible",
        )
        ax.plot(self.checkpoints, self.means, color="#22223b", lw=2)
        ax.plot(
            self.checkpoints,
            self.successes / self.checkpoints,
            color="#c9184a",
            ls=":",
            lw=1.5,
            label="observed share",
        )
        ax.set_xlabel("observations seen")
        ax.set_ylabel(self.label)
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        return ax

    def __repr__(self) -> str:
        return (
            f"<SequentialUpdate: {len(self.posteriors)} steps, "
            f"{int(self.checkpoints[-1]):,} observations, interval width "
            f"{self.widths[0]:.3f} -> {self.widths[-1]:.3f}>"
            "\nCall .table() or .plot()."
        )


def sequential(
    outcomes,
    step: int = 10,
    prior="uninformed",
    level: float = 0.95,
    label: str = "rate",
) -> SequentialUpdate:
    """Recompute the posterior after each chunk of incoming data.

    For the lesson where evidence arrives a bit at a time and the class watches
    the estimate move and the interval close. Also the cheapest way to make
    two abstractions concrete at once: that a posterior is a running summary of
    everything seen so far, and that precision improves like the square root of
    the sample rather than in proportion to it.

    Parameters
    ----------
    outcomes : array_like
        A sequence of 0/1 or boolean outcomes, **in the order observed**. A
        boolean pandas column works directly.
    step : int, default 10
        Chunk size. Ten is a good classroom pace: enough steps to see a trend,
        few enough to read the table aloud.
    prior : str, tuple, or BetaPrior, default 'uninformed'
        Prior on the rate, before any data.
    level : float, default 0.95
        Credible level for the reported intervals.
    label : str, default 'rate'
        What is being estimated, for printed output.

    Returns
    -------
    SequentialUpdate
        Call ``.table()`` or ``.plot()``.

    Examples
    --------
    >>> import bayesplain as bp
    >>> outcomes = [1, 0, 0, 1, 0] * 20
    >>> run = bp.teach.sequential(outcomes, step=20)
    >>> len(run.posteriors)
    5
    >>> bool(run.widths[-1] < run.widths[0])
    True
    """
    arr = np.asarray(outcomes).ravel()
    if arr.dtype == bool:
        arr = arr.astype(int)
    arr = arr[np.isfinite(arr.astype(float))]
    if arr.size < 2:
        raise ValueError(f"need at least 2 outcomes to show an update, got {arr.size}.")
    if not np.isin(arr, (0, 1)).all():
        raise ValueError(
            "outcomes must be 0/1 or boolean — one entry per observation, in "
            "the order observed. For summary counts instead of a sequence, use "
            "bayesplain.proportion()."
        )
    step = int(step)
    if step < 1:
        raise ValueError(f"step must be at least 1, got {step}.")

    resolved = priors.resolve_proportion(prior)
    edges = list(range(step, arr.size + 1, step))
    if not edges or edges[-1] != arr.size:
        edges.append(arr.size)

    checkpoints = np.array(edges, dtype=float)
    successes = np.array([arr[:k].sum() for k in edges], dtype=float)
    posteriors = [
        beta_binomial.posterior(int(s), int(k), resolved.a, resolved.b)
        for s, k in zip(successes, checkpoints)
    ]

    return SequentialUpdate(
        checkpoints=checkpoints,
        successes=successes,
        posteriors=posteriors,
        prior=resolved,
        level=float(level),
        label=label,
    )


# ---------------------------------------------------------------------------
# The honest replacement for a power calculation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrecisionPlan:
    """How much data a target precision requires.

    :ivar required_n: Observations needed to reach the target width.
    :ivar target_width: The interval width that was asked for.
    :ivar expected_rate: The rate the calculation assumed.
    :ivar level: Credible level.
    :ivar schedule: ``(n, width)`` pairs along the way.
    :ivar prior: The prior assumed.
    """

    required_n: int
    target_width: float
    expected_rate: float
    level: float
    schedule: np.ndarray
    prior: object = None

    def summary(self) -> _Report:
        """Report the required sample size and the trade-off around it.

        Returns
        -------
        _Report
            Printable text.
        """
        pct = f"{self.level:.0%}"
        lines = [
            "HOW MUCH DATA IS ENOUGH?",
            "",
        ]
        lines += _wrap(
            f"To get a {pct} credible interval no wider than "
            f"{self.target_width:.3f} ({self.target_width * 100:.1f} "
            f"percentage points), assuming the true rate is near "
            f"{self.expected_rate:.0%}, you need about "
            f"{self.required_n:,} observations.",
            prefix="  ",
        )
        lines += ["", f"{'n':>10}{'interval width':>18}", "-" * 30]
        for n, width in self.schedule:
            marker = "  <- target" if n >= self.required_n else ""
            lines.append(f"{int(n):>10,}{width:>18.4f}{marker}")
        lines += [""]
        lines += _wrap(
            "This is the question a power calculation is usually reaching for, "
            "asked in a way that survives contact with a decision. Power asks "
            "how often you would detect an effect of a size you had to guess "
            "at; this asks how precisely you will know the answer, which is "
            "what determines whether you can act on it. Notice the shape of "
            "the table: quadrupling the sample halves the width, so there is "
            "always a point past which more data stops being worth its cost.",
            prefix="  ",
        )
        return _Report("\n".join(lines))

    def __repr__(self) -> str:
        return (
            f"<PrecisionPlan: about {self.required_n:,} observations for a "
            f"{self.level:.0%} interval of width {self.target_width:.3f} "
            f"near a rate of {self.expected_rate:.0%}>"
            "\nCall .summary() for the schedule."
        )


def precision_planning(
    target_width: float,
    expected_rate: float = 0.5,
    level: float = 0.95,
    prior="uninformed",
    max_n: int = 1_000_000,
) -> PrecisionPlan:
    """Find the sample size that makes an interval narrow enough to act on.

    Computed exactly rather than through a normal approximation: for each
    candidate ``n`` the actual Beta posterior interval is measured, and the
    search stops at the first ``n`` that meets the target.

    Parameters
    ----------
    target_width : float
        The widest interval you could still act on, on the rate's own scale.
        ``0.05`` means five percentage points.
    expected_rate : float, default 0.5
        The rate to plan around. 0.5 is the conservative choice, since a rate
        near one half is the hardest to pin down; plan around a rate closer to
        what you expect if you have grounds for one.
    level : float, default 0.95
        Credible level.
    prior : str, tuple, or BetaPrior, default 'uninformed'
        Prior on the rate. Note that strength alone does not help: a prior
        centred near ``expected_rate`` reduces the data required, while an
        equally strong prior centred elsewhere increases it, since it pulls the
        posterior toward a region of higher variance. Making that distinction
        visible is much of the point of this method.
    max_n : int, default 1_000_000
        Give up beyond this, rather than searching forever for an
        unattainable precision.

    Returns
    -------
    PrecisionPlan
        Call ``.summary()``.

    Examples
    --------
    >>> import bayesplain as bp
    >>> plan = bp.teach.precision_planning(target_width=0.10, expected_rate=0.15)
    >>> plan.required_n
    195

    What matters is not how strong the prior is but whether it agrees with what
    you expect to find. A prior centred on the rate you are planning around
    halves the data you need; one of similar strength centred somewhere else
    *increases* it, because it drags the posterior toward the middle where a
    rate is hardest to pin down:

    >>> agrees = bp.priors.from_previous_study(successes=15, n=100)
    >>> bp.teach.precision_planning(0.10, 0.15, prior=agrees).required_n
    95
    >>> bp.teach.precision_planning(0.10, 0.15, prior="skeptical").required_n
    207
    """
    if not 0.0 < target_width < 1.0:
        raise ValueError(
            f"target_width must be between 0 and 1, got {target_width}. Pass "
            "0.05 for five percentage points."
        )
    if not 0.0 <= expected_rate <= 1.0:
        raise ValueError(f"expected_rate must be between 0 and 1, got {expected_rate}.")
    resolved = priors.resolve_proportion(prior)

    def width_at(n: int) -> float:
        successes = int(round(n * expected_rate))
        post = beta_binomial.posterior(successes, n, resolved.a, resolved.b)
        return float(
            intervals.interval(level=level, kind="eti", dist=post)[1]
            - intervals.interval(level=level, kind="eti", dist=post)[0]
        )

    # Geometric bracket, then bisect: the width falls like 1/sqrt(n), so a
    # linear scan would waste most of its work at the wrong scale.
    low, high = 2, 4
    while width_at(high) > target_width:
        low, high = high, high * 2
        if high > max_n:
            raise ValueError(
                f"a {level:.0%} interval of width {target_width} near a rate "
                f"of {expected_rate:.0%} needs more than {max_n:,} "
                "observations. Either the target is too demanding or the rate "
                "is too close to 0 or 1 to pin down that tightly."
            )
    while low + 1 < high:
        middle = (low + high) // 2
        if width_at(middle) > target_width:
            low = middle
        else:
            high = middle
    required = high

    marks = sorted(
        {
            max(2, required // 8),
            max(2, required // 4),
            max(2, required // 2),
            required,
            required * 2,
        }
    )
    schedule = np.array([[n, width_at(n)] for n in marks], dtype=float)

    return PrecisionPlan(
        required_n=int(required),
        target_width=float(target_width),
        expected_rate=float(expected_rate),
        level=float(level),
        schedule=schedule,
        prior=resolved,
    )
