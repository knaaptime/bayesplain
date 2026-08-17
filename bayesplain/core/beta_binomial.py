"""Beta-Binomial conjugate analysis for proportions.

Everything in this module is exact: the posterior is a Beta distribution
written down in closed form, and the marginal likelihoods are ratios of beta
functions. No integration, no sampling, no approximation of any kind.

Pure functions only. Nothing here knows it is being used to teach.
"""

from __future__ import annotations

import numpy as np
from scipy import special, stats

__all__ = [
    "posterior",
    "draws",
    "log_marginal_likelihood",
    "log_bayes_factor_point_null",
    "prior_predictive_mean",
    "validate_counts",
]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_counts(successes: float, n: float) -> tuple[int, int]:
    """Check that a successes/trials pair describes a possible experiment.

    Parameters
    ----------
    successes : int
        Number of successes observed. Must be a non-negative whole number.
    n : int
        Number of trials. Must be a positive whole number, at least as large
        as ``successes``.

    Returns
    -------
    tuple of int
        The validated ``(successes, n)`` pair, coerced to Python ints.

    Raises
    ------
    ValueError
        If the counts are negative, non-integral, or if ``successes > n``.
    """
    for name, value in (("successes", successes), ("n", n)):
        if value != int(value):
            raise ValueError(
                f"{name} must be a whole number of observations, got {value!r}. "
                "If you have a rate rather than counts, multiply it by the "
                "number of observations first."
            )
    successes, n = int(successes), int(n)
    if n <= 0:
        raise ValueError(f"n must be at least 1, got {n}.")
    if successes < 0:
        raise ValueError(f"successes must be non-negative, got {successes}.")
    if successes > n:
        raise ValueError(
            f"successes ({successes}) cannot exceed the number of trials ({n})."
        )
    return successes, n


def _validate_shape(a: float, b: float) -> None:
    if a <= 0 or b <= 0:
        raise ValueError(f"Beta prior shapes must be positive, got a={a}, b={b}.")


# ---------------------------------------------------------------------------
# Posterior
# ---------------------------------------------------------------------------


def posterior(successes: float, n: float, a: float = 1.0, b: float = 1.0):
    """Exact Beta posterior for an unknown proportion.

    With a ``Beta(a, b)`` prior and ``successes`` out of ``n`` observed, the
    posterior is ``Beta(a + successes, b + n - successes)``. That is the whole
    computation -- conjugacy means the prior and posterior are the same family,
    so observing data just adds counts to the two shape parameters.

    Parameters
    ----------
    successes : int
        Number of successes observed.
    n : int
        Number of trials.
    a, b : float, default 1.0
        Shape parameters of the Beta prior. ``Beta(1, 1)`` is flat on [0, 1].

    Returns
    -------
    scipy.stats.rv_continuous_frozen
        Frozen Beta posterior, supporting ``.mean()``, ``.ppf()``, ``.rvs()``
        and the rest of the scipy distribution interface.

    Notes
    -----
    Because the posterior is available in closed form, quantities like the
    mean and the equal-tailed interval should be read off analytically rather
    than estimated from draws. Draws are only needed for quantities built from
    *more than one* posterior, such as a difference between two groups.
    """
    successes, n = validate_counts(successes, n)
    _validate_shape(a, b)
    return stats.beta(a + successes, b + n - successes)


def prior_predictive_mean(a: float = 1.0, b: float = 1.0) -> float:
    """Mean of a ``Beta(a, b)`` prior, i.e. the rate expected before data.

    Parameters
    ----------
    a, b : float, default 1.0
        Shape parameters of the Beta prior.

    Returns
    -------
    float
        ``a / (a + b)``.
    """
    _validate_shape(a, b)
    return a / (a + b)


# ---------------------------------------------------------------------------
# Marginal likelihood and Bayes factors
# ---------------------------------------------------------------------------


def log_marginal_likelihood(
    successes: float, n: float, a: float = 1.0, b: float = 1.0
) -> float:
    r"""Log marginal likelihood of binomial data under a Beta prior.

    Integrating the binomial likelihood against a ``Beta(a, b)`` prior gives
    the beta-binomial probability mass

    .. math::

        p(x) = \binom{n}{x} \frac{B(a + x,\; b + n - x)}{B(a, b)}

    which is closed form -- there is no integral left to evaluate
    numerically.

    Parameters
    ----------
    successes : int
        Number of successes observed.
    n : int
        Number of trials.
    a, b : float, default 1.0
        Shape parameters of the Beta prior.

    Returns
    -------
    float
        The natural log of the marginal likelihood, including the binomial
        coefficient. Computed in log space so that large ``n`` does not
        overflow.
    """
    x, n = validate_counts(successes, n)
    _validate_shape(a, b)
    log_choose = (
        special.gammaln(n + 1) - special.gammaln(x + 1) - special.gammaln(n - x + 1)
    )
    return float(log_choose + special.betaln(a + x, b + n - x) - special.betaln(a, b))


def log_bayes_factor_point_null(
    successes: float,
    n: float,
    p0: float = 0.5,
    a: float = 1.0,
    b: float = 1.0,
) -> float:
    """Log Bayes factor comparing an unknown rate against a fixed value.

    Compares ``H1``: the rate is unknown with a ``Beta(a, b)`` prior, against
    ``H0``: the rate is exactly ``p0``. Both marginal likelihoods are closed
    form, so this is an exact ratio rather than an estimate.

    Parameters
    ----------
    successes : int
        Number of successes observed.
    n : int
        Number of trials.
    p0 : float, default 0.5
        The value the null hypothesis fixes the rate at.
    a, b : float, default 1.0
        Shape parameters of the Beta prior under ``H1``.

    Returns
    -------
    float
        ``log(BF10)``. Positive values favour an unknown rate; negative
        values favour ``p0``.

    Raises
    ------
    ValueError
        If ``p0`` is not strictly inside (0, 1), or if it is a boundary value
        that the data contradict outright.
    """
    x, n = validate_counts(successes, n)
    _validate_shape(a, b)
    if not 0.0 < p0 < 1.0:
        raise ValueError(
            f"p0 must be strictly between 0 and 1, got {p0}. A null of exactly "
            "0 or 1 is contradicted by any opposing observation, so the Bayes "
            "factor is not finite."
        )
    log_alt = log_marginal_likelihood(x, n, a, b)
    log_null = float(stats.binom.logpmf(x, n, p0))
    return log_alt - log_null


# ---------------------------------------------------------------------------
# Draws
# ---------------------------------------------------------------------------


def draws(
    successes: float,
    n: float,
    a: float = 1.0,
    b: float = 1.0,
    size: int = 100_000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw independent samples from the exact Beta posterior.

    These are i.i.d. draws from a distribution known in closed form -- the
    same operation as rolling a die, not a Markov chain. There is nothing to
    check for convergence and no burn-in to discard.

    Parameters
    ----------
    successes : int
        Number of successes observed.
    n : int
        Number of trials.
    a, b : float, default 1.0
        Shape parameters of the Beta prior.
    size : int, default 100_000
        Number of draws.
    rng : numpy.random.Generator, optional
        Random generator. Supply one for reproducibility.

    Returns
    -------
    ndarray
        Array of ``size`` draws from the posterior.
    """
    rng = np.random.default_rng() if rng is None else rng
    post = posterior(successes, n, a, b)
    return post.rvs(size=size, random_state=rng)
