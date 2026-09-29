"""Summarising a posterior: intervals, threshold probabilities, Monte Carlo error.

Two kinds of interval appear throughout the package:

**Equal-tailed (ETI)** cuts the same probability off each end, so a 95% ETI
runs from the 2.5th to the 97.5th percentile. It is the one that lines up with
a frequentist confidence interval numerically, which makes it the better
choice when the point of a table is to compare the two side by side.

**Highest-density (HDI)** is the *shortest* interval containing the stated
probability. Every value inside it is more plausible than every value outside,
which is what people usually mean when they gesture at "the range the answer
is in". For a skewed posterior -- a rate near 0 or 1, say -- the two intervals
differ visibly, and that difference is worth showing students once.

Pure functions only.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize

__all__ = [
    "eti_from_draws",
    "hdi_from_dist",
    "hdi_from_draws",
    "interval",
    "monte_carlo_se",
    "probability_from_draws",
]


# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------


def _validate_level(level: float) -> float:
    if not 0.0 < level < 1.0:
        raise ValueError(
            f"level must be strictly between 0 and 1, got {level}. Pass 0.95 "
            "rather than 95."
        )
    return float(level)


def eti_from_draws(draws, level: float = 0.95) -> tuple[float, float]:
    """Equal-tailed interval from posterior draws.

    Parameters
    ----------
    draws : array_like
        Posterior draws.
    level : float, default 0.95
        Probability the interval should contain.

    Returns
    -------
    tuple of float
        Lower and upper bounds.
    """
    level = _validate_level(level)
    arr = np.asarray(draws, dtype=float).ravel()
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(arr, [tail, 1.0 - tail])
    return float(lo), float(hi)


def hdi_from_draws(draws, level: float = 0.95) -> tuple[float, float]:
    """Highest-density interval from posterior draws.

    Finds the shortest window that contains ``level`` of the draws, by sliding
    a fixed-width window across the sorted sample. Valid for unimodal
    posteriors, which covers everything this package produces.

    Parameters
    ----------
    draws : array_like
        Posterior draws.
    level : float, default 0.95
        Probability the interval should contain.

    Returns
    -------
    tuple of float
        Lower and upper bounds of the shortest such interval.
    """
    level = _validate_level(level)
    arr = np.sort(np.asarray(draws, dtype=float).ravel())
    n = arr.size
    if n < 2:
        raise ValueError("need at least 2 draws to compute an interval.")

    n_included = max(2, int(np.ceil(level * n)))
    if n_included >= n:
        return float(arr[0]), float(arr[-1])

    widths = arr[n_included - 1 :] - arr[: n - n_included + 1]
    start = int(np.argmin(widths))
    return float(arr[start]), float(arr[start + n_included - 1])


def hdi_from_dist(dist, level: float = 0.95) -> tuple[float, float]:
    """Highest-density interval of an analytic, unimodal distribution.

    The shortest interval holding ``level`` of the mass, found by choosing how
    much probability to leave in the lower tail. Exact up to the optimiser's
    tolerance, with no Monte Carlo error -- and when the density piles up
    against a boundary, as a rate posterior does after zero successes, the
    interval starts exactly at that boundary rather than a hair above it.

    Parameters
    ----------
    dist : scipy.stats frozen distribution
        Continuous and unimodal (or monotone).
    level : float, default 0.95
        Probability the interval should contain.

    Returns
    -------
    tuple of float
        Lower and upper bounds.

    Examples
    --------
    >>> from scipy import stats
    >>> hdi_from_dist(stats.beta(1, 21))[0]
    0.0
    """
    level = _validate_level(level)
    spare = 1.0 - level

    def width(lower_tail: float) -> float:
        return float(dist.ppf(lower_tail + level) - dist.ppf(lower_tail))

    candidates = [0.0, spare]
    found = optimize.minimize_scalar(
        width, bounds=(0.0, spare), method="bounded", options={"xatol": 1e-10}
    )
    if found.success:
        candidates.append(float(found.x))
    widths = [width(c) for c in candidates]
    finite = [(w, c) for w, c in zip(widths, candidates) if np.isfinite(w)]
    if not finite:
        # An unbounded tail with no finite interval; fall back to equal tails.
        return float(dist.ppf(spare / 2)), float(dist.ppf(1 - spare / 2))
    best = min(finite)[1]
    return float(dist.ppf(best)), float(dist.ppf(best + level))


def interval(
    draws=None,
    level: float = 0.95,
    kind: str = "hdi",
    dist=None,
) -> tuple[float, float]:
    """Posterior interval, taken analytically when that is possible.

    If a frozen scipy distribution is supplied, the bounds come from it -- the
    quantile function for an equal-tailed interval, a one-dimensional
    optimisation for a highest-density one -- and carry no Monte Carlo error
    at all. Otherwise they are estimated from draws.

    Parameters
    ----------
    draws : array_like, optional
        Posterior draws. Required unless an ETI is requested with ``dist``.
    level : float, default 0.95
        Probability the interval should contain.
    kind : {'hdi', 'eti'}, default 'hdi'
        Highest-density or equal-tailed.
    dist : scipy.stats.rv_continuous_frozen, optional
        Analytic posterior, when one exists.

    Returns
    -------
    tuple of float
        Lower and upper bounds.
    """
    level = _validate_level(level)
    kind = kind.lower()
    if kind not in {"hdi", "eti"}:
        raise ValueError(f"kind must be 'hdi' or 'eti', got {kind!r}.")

    if dist is not None:
        if kind == "hdi":
            return hdi_from_dist(dist, level)
        tail = (1.0 - level) / 2.0
        return float(dist.ppf(tail)), float(dist.ppf(1.0 - tail))
    if draws is None:
        raise ValueError("need either draws or an analytic distribution.")
    if kind == "eti":
        return eti_from_draws(draws, level)
    return hdi_from_draws(draws, level)


# ---------------------------------------------------------------------------
# Threshold probabilities
# ---------------------------------------------------------------------------


def probability_from_draws(draws, op: str, value) -> float:
    """Posterior probability that a quantity satisfies a comparison.

    Parameters
    ----------
    draws : array_like
        Posterior draws.
    op : {'>', '>=', '<', '<=', 'between', 'outside'}
        Comparison to evaluate.
    value : float or tuple of float
        Threshold, or a ``(low, high)`` pair for ``'between'`` and
        ``'outside'``.

    Returns
    -------
    float
        The share of posterior probability satisfying the comparison.
    """
    arr = np.asarray(draws, dtype=float).ravel()
    if op in {"between", "outside"}:
        try:
            low, high = value
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"op={op!r} needs a (low, high) pair, got {value!r}."
            ) from err
        if low > high:
            low, high = high, low
        inside = float(np.mean((arr >= low) & (arr <= high)))
        return inside if op == "between" else 1.0 - inside

    comparisons = {
        ">": lambda a, v: a > v,
        ">=": lambda a, v: a >= v,
        "<": lambda a, v: a < v,
        "<=": lambda a, v: a <= v,
    }
    if op not in comparisons:
        raise ValueError(
            f"op must be one of '>', '>=', '<', '<=', 'between', 'outside', got {op!r}."
        )
    return float(np.mean(comparisons[op](arr, float(value))))


# ---------------------------------------------------------------------------
# Monte Carlo error
# ---------------------------------------------------------------------------


def monte_carlo_se(draws) -> float:
    """Compute the standard error of a posterior mean from i.i.d. draws.

    Because the draws are independent -- these are exact samples from a
    closed-form distribution, not a Markov chain -- this is simply
    ``sd / sqrt(n)``, with no effective-sample-size correction needed. That
    absence is itself the point: there is no autocorrelation to discount.

    Parameters
    ----------
    draws : array_like
        Posterior draws.

    Returns
    -------
    float
        Monte Carlo standard error of the mean.
    """
    arr = np.asarray(draws, dtype=float).ravel()
    if arr.size < 2:
        return float("nan")
    return float(arr.std(ddof=1) / np.sqrt(arr.size))
