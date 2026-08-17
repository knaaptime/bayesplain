r"""Posteriors for means, and the JZS Bayes factor for a difference between them.

With the standard reference prior :math:`p(\mu, \sigma^2) \propto 1/\sigma^2`,
the posterior for an unknown mean is exactly a scaled Student-t:

.. math::

    \mu \mid \text{data} \sim t_{n-1}\!\left(\bar{x},\; s^2/n\right)

Closed form -- no sampling, nothing to converge. It is worth noticing that this
interval coincides numerically with the frequentist t interval on the same
data. The arithmetic is identical; what differs is what you are allowed to say
about the result, and that is the whole lesson of the week where means are
introduced.

Two groups
----------
With **equal variances** assumed, the difference of the two means is again
exactly a scaled Student-t, pooling the two sample variances.

With **unequal variances** -- the honest default, and what Welch's test already
assumes on the frequentist side -- this is the Behrens-Fisher problem. Each
group's mean still has an exact Student-t posterior, but the difference of two
Student-t variables has no tidy closed form, so the difference is obtained by
drawing from each and subtracting. Those draws are independent samples from
distributions known in closed form: still not MCMC, still nothing to converge.

The Bayes factor
----------------
The JZS Bayes factor of Rouder et al. (2009) places a Cauchy prior of scale
:math:`r` on the standardised effect size and a Jeffreys prior on the variance.
Integrating out the variance analytically leaves a one-dimensional integral
over the Cauchy's scale mixture parameter :math:`g`:

.. math::

    \mathrm{BF}_{10} = \frac{\int_0^\infty (1+Ng)^{-1/2}
        \left[1 + \frac{t^2}{(1+Ng)\nu}\right]^{-(\nu+1)/2} \pi(g)\, dg}
        {\left[1 + t^2/\nu\right]^{-(\nu+1)/2}}

with :math:`\pi(g)` an inverse-gamma(1/2, :math:`r^2`/2) density. One
dimension, so ``scipy.integrate.quad`` handles it directly -- no sampler
required. For a two-sample test, :math:`N` becomes the effective sample size
:math:`n_1 n_2/(n_1+n_2)` and :math:`\nu = n_1 + n_2 - 2`.

References
----------
Rouder, J. N., Speckman, P. L., Sun, D., Morey, R. D., and Iverson, G. (2009).
Bayesian t tests for accepting and rejecting the null hypothesis.
*Psychonomic Bulletin & Review*, 16(2), 225-237.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, stats

__all__ = [
    "DEFAULT_CAUCHY_SCALE",
    "summarise",
    "mean_posterior",
    "pooled_difference_posterior",
    "difference_draws",
    "log_bayes_factor_ttest",
    "validate_sample",
]

#: The conventional Cauchy prior width on standardised effect size, matching
#: the current default of R's BayesFactor package. Rouder et al. (2009)
#: originally used 1.0.
DEFAULT_CAUCHY_SCALE = 0.707


# ---------------------------------------------------------------------------
# Validation and summaries
# ---------------------------------------------------------------------------


def validate_sample(x, name: str = "x") -> np.ndarray:
    """Check that a sample is usable, dropping missing values.

    Parameters
    ----------
    x : array_like
        Observations. Non-finite values are dropped.
    name : str, default 'x'
        Name to use in error messages.

    Returns
    -------
    ndarray
        The finite observations, as a 1-D float array.

    Raises
    ------
    ValueError
        If fewer than two finite observations remain, or if they are all
        identical (which leaves the variance unidentified).
    """
    arr = np.asarray(x, dtype=float).ravel()
    finite = arr[np.isfinite(arr)]
    if finite.size < 2:
        raise ValueError(
            f"{name} needs at least 2 finite observations to estimate a mean "
            f"and a spread, got {finite.size}."
        )
    if np.ptp(finite) == 0:
        raise ValueError(
            f"every value in {name} is identical ({finite[0]:g}), so there is "
            "no spread to estimate and the posterior is undefined."
        )
    return finite


def summarise(x, name: str = "x") -> tuple[int, float, float]:
    """Reduce a sample to the three numbers every later step needs.

    Parameters
    ----------
    x : array_like
        Observations.
    name : str, default 'x'
        Name to use in error messages.

    Returns
    -------
    tuple
        ``(n, mean, sd)`` with ``sd`` the sample standard deviation using the
        ``n - 1`` denominator.
    """
    arr = validate_sample(x, name)
    return int(arr.size), float(arr.mean()), float(arr.std(ddof=1))


# ---------------------------------------------------------------------------
# Posteriors
# ---------------------------------------------------------------------------


def mean_posterior(n: int, mean: float, sd: float):
    """Exact Student-t posterior for one unknown mean.

    Parameters
    ----------
    n : int
        Number of observations.
    mean : float
        Sample mean.
    sd : float
        Sample standard deviation (``n - 1`` denominator).

    Returns
    -------
    scipy.stats.rv_continuous_frozen
        Frozen scaled Student-t with ``n - 1`` degrees of freedom, centred on
        the sample mean with scale ``sd / sqrt(n)``.

    Notes
    -----
    Under the reference prior this is exact. Its equal-tailed interval
    coincides numerically with the usual t confidence interval, which makes
    this the cleanest place to show that the two frameworks can agree on every
    digit and still license different sentences.
    """
    if n < 2:
        raise ValueError(f"need at least 2 observations, got {n}.")
    if sd <= 0:
        raise ValueError(f"sd must be positive, got {sd}.")
    return stats.t(df=n - 1, loc=mean, scale=sd / np.sqrt(n))


def pooled_difference_posterior(
    n1: int, mean1: float, sd1: float, n2: int, mean2: float, sd2: float
):
    """Exact Student-t posterior for a difference of means, equal variances.

    Parameters
    ----------
    n1, n2 : int
        Group sizes.
    mean1, mean2 : float
        Group means.
    sd1, sd2 : float
        Group standard deviations.

    Returns
    -------
    scipy.stats.rv_continuous_frozen
        Frozen posterior for ``mean2 - mean1``, with ``n1 + n2 - 2`` degrees of
        freedom.
    """
    if n1 < 2 or n2 < 2:
        raise ValueError("each group needs at least 2 observations.")
    df = n1 + n2 - 2
    pooled_var = ((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / df
    scale = np.sqrt(pooled_var * (1.0 / n1 + 1.0 / n2))
    return stats.t(df=df, loc=mean2 - mean1, scale=scale)


def difference_draws(
    n1: int,
    mean1: float,
    sd1: float,
    n2: int,
    mean2: float,
    sd2: float,
    size: int = 100_000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw the difference of two means without assuming equal variances.

    The Behrens-Fisher case. Each group's mean posterior is an exact Student-t
    that can be sampled directly; the difference of two of them has no closed
    form, so the draws are subtracted pairwise.

    Parameters
    ----------
    n1, n2 : int
        Group sizes.
    mean1, mean2 : float
        Group means.
    sd1, sd2 : float
        Group standard deviations.
    size : int, default 100_000
        Number of draws.
    rng : numpy.random.Generator, optional
        Random generator.

    Returns
    -------
    ndarray
        Draws of ``mean2 - mean1``.
    """
    rng = np.random.default_rng() if rng is None else rng
    post1 = mean_posterior(n1, mean1, sd1)
    post2 = mean_posterior(n2, mean2, sd2)
    return post2.rvs(size=size, random_state=rng) - post1.rvs(
        size=size, random_state=rng
    )


# ---------------------------------------------------------------------------
# The JZS Bayes factor
# ---------------------------------------------------------------------------


def log_bayes_factor_ttest(
    t: float,
    n_effective: float,
    df: float,
    scale: float = DEFAULT_CAUCHY_SCALE,
) -> float:
    """Log JZS Bayes factor for a t statistic.

    Parameters
    ----------
    t : float
        The t statistic.
    n_effective : float
        Sample size entering the effect-size scaling: ``n`` for a one-sample
        or paired test, ``n1 * n2 / (n1 + n2)`` for two independent samples.
    df : float
        Degrees of freedom: ``n - 1`` or ``n1 + n2 - 2``.
    scale : float, default 0.707
        Width of the Cauchy prior on standardised effect size. Larger values
        spread prior mass over bigger effects, which costs the alternative
        when the observed effect is small.

    Returns
    -------
    float
        ``log(BF10)``. Positive favours a difference; negative favours none.

    Notes
    -----
    The integrand is evaluated relative to the null likelihood so that it stays
    near unity regardless of ``t`` and ``df``, which keeps the quadrature
    stable for large samples.
    """
    if df <= 0:
        raise ValueError(f"df must be positive, got {df}.")
    if n_effective <= 0:
        raise ValueError(f"n_effective must be positive, got {n_effective}.")
    if scale <= 0:
        raise ValueError(f"Cauchy prior scale must be positive, got {scale}.")

    t2 = float(t) ** 2
    log_null = -0.5 * (df + 1.0) * np.log1p(t2 / df)
    prior_g = stats.invgamma(a=0.5, scale=scale**2 / 2.0)

    def integrand(g):
        shrink = 1.0 + n_effective * g
        log_alt = -0.5 * np.log(shrink) - 0.5 * (df + 1.0) * np.log1p(
            t2 / (shrink * df)
        )
        return np.exp(log_alt - log_null) * prior_g.pdf(g)

    value, _ = integrate.quad(integrand, 0.0, np.inf, limit=200)
    if not np.isfinite(value) or value <= 0:
        raise RuntimeError(
            f"the JZS integral did not converge for t={t}, df={df}, scale={scale}."
        )
    return float(np.log(value))
