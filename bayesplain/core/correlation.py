r"""Posterior and Bayes factor for a correlation coefficient.

Everything here is built from one thing: the exact sampling density of the
observed correlation :math:`r` given the true correlation :math:`\rho` and the
sample size :math:`n`,

.. math::

    f(r \mid \rho, n) = \frac{(n-2)\,\Gamma(n-1)\,(1-\rho^2)^{(n-1)/2}
                              (1-r^2)^{(n-4)/2}}
                             {\sqrt{2\pi}\,\Gamma(n-\tfrac12)\,
                              (1-\rho r)^{n-\frac32}}
        \; {}_2F_1\!\left(\tfrac12, \tfrac12; n-\tfrac12;
                          \tfrac{1+\rho r}{2}\right)

which ``scipy.special.hyp2f1`` evaluates directly. Given that density, the
posterior over :math:`\rho` and the Bayes factor against :math:`\rho = 0`
follow by one-dimensional integration:

.. math::

    p(\rho \mid r) \propto f(r \mid \rho, n)\,\pi(\rho), \qquad
    \mathrm{BF}_{10} = \frac{\int_{-1}^{1} f(r \mid \rho, n)\,\pi(\rho)\,d\rho}
                            {f(r \mid 0, n)}

Ly et al. (2016) give closed forms for these in terms of the same
hypergeometric function. Evaluating the integrals on a grid instead costs
nothing at this size, reuses the grid machinery the teaching module already
needs, and keeps one code path for any prior width rather than a special case
per value of :math:`\kappa`.

The prior
---------
:math:`\pi(\rho)` is a stretched beta on :math:`(-1, 1)`: if
:math:`(\rho+1)/2 \sim \mathrm{Beta}(1/\kappa, 1/\kappa)` then
:math:`\pi(\rho) \propto (1-\rho^2)^{1/\kappa - 1}`. At :math:`\kappa = 1` this
is flat, meaning every correlation from -1 to 1 starts out equally plausible.
Smaller :math:`\kappa` concentrates the prior near zero.

References
----------
Ly, A., Verhagen, J., and Wagenmakers, E.-J. (2016). Harold Jeffreys's default
Bayes factor hypothesis tests: Explanation, extension, and application in
psychology. *Journal of Mathematical Psychology*, 72, 19-32.
"""

from __future__ import annotations

import numpy as np
from scipy import special

from . import grid as grid_utils

__all__ = [
    "DEFAULT_KAPPA",
    "log_sampling_density",
    "log_prior_density",
    "rho_grid",
    "posterior_on_grid",
    "log_bayes_factor",
    "validate_pair",
]

#: Prior width. ``kappa = 1`` is flat on (-1, 1), the conventional default.
DEFAULT_KAPPA = 1.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_pair(x, y) -> tuple[np.ndarray, np.ndarray]:
    """Check two paired samples and drop rows where either value is missing.

    Parameters
    ----------
    x, y : array_like
        Paired observations of equal length.

    Returns
    -------
    tuple of ndarray
        The complete pairs.

    Raises
    ------
    ValueError
        If the lengths differ, fewer than four complete pairs remain, or
        either variable is constant.
    """
    a = np.asarray(x, dtype=float).ravel()
    b = np.asarray(y, dtype=float).ravel()
    if a.size != b.size:
        raise ValueError(
            f"x and y must be the same length, got {a.size} and {b.size}. "
            "A correlation needs paired observations."
        )
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if a.size < 4:
        raise ValueError(
            f"need at least 4 complete pairs to estimate a correlation, got {a.size}."
        )
    for name, arr in (("x", a), ("y", b)):
        if np.ptp(arr) == 0:
            raise ValueError(
                f"every value of {name} is identical, so it cannot correlate "
                "with anything."
            )
    return a, b


# ---------------------------------------------------------------------------
# The exact sampling density
# ---------------------------------------------------------------------------


def log_sampling_density(r, rho, n: int):
    """Log density of an observed correlation given the true one.

    Parameters
    ----------
    r : float
        Observed sample correlation, strictly inside (-1, 1).
    rho : float or array_like
        True correlation(s) to evaluate at, strictly inside (-1, 1).
    n : int
        Number of paired observations.

    Returns
    -------
    float or ndarray
        ``log f(r | rho, n)``, matching the shape of ``rho``.

    Notes
    -----
    Exact, not an approximation: this is the Hotelling density, and the
    hypergeometric factor is what makes it exact rather than the more familiar
    Fisher-z normal approximation.
    """
    if n < 4:
        raise ValueError(f"need at least 4 observations, got {n}.")
    r = float(r)
    if not -1.0 < r < 1.0:
        raise ValueError(
            f"the observed correlation must be strictly inside (-1, 1), got "
            f"{r}. A correlation of exactly ±1 leaves nothing to estimate."
        )
    rho = np.asarray(rho, dtype=float)

    constant = (
        np.log(n - 2)
        + special.gammaln(n - 1)
        - 0.5 * np.log(2.0 * np.pi)
        - special.gammaln(n - 0.5)
    )
    log_density = (
        constant
        + 0.5 * (n - 1) * np.log1p(-(rho**2))
        + 0.5 * (n - 4) * np.log1p(-(r**2))
        - (n - 1.5) * np.log1p(-rho * r)
        + np.log(special.hyp2f1(0.5, 0.5, n - 0.5, (1.0 + rho * r) / 2.0))
    )
    return log_density


def log_prior_density(rho, kappa: float = DEFAULT_KAPPA):
    """Log density of the stretched-beta prior on a correlation.

    Parameters
    ----------
    rho : float or array_like
        Values in (-1, 1).
    kappa : float, default 1.0
        Prior width. 1.0 is flat; smaller values concentrate near zero.

    Returns
    -------
    ndarray
        Log prior density, normalised over (-1, 1).
    """
    if kappa <= 0:
        raise ValueError(f"kappa must be positive, got {kappa}.")
    rho = np.asarray(rho, dtype=float)
    shape = 1.0 / kappa
    # (rho + 1) / 2 ~ Beta(shape, shape), with the Jacobian of the stretch.
    return (
        (shape - 1.0) * np.log1p(-(rho**2))
        - special.betaln(shape, shape)
        - (2.0 * shape - 1.0) * np.log(2.0)
    )


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------


def rho_grid(r: float, n: int, points: int = 1201) -> np.ndarray:
    """Build a grid over (-1, 1) that resolves a sharply peaked posterior.

    A uniform grid wastes most of its points far from the peak and, for large
    samples, may not place enough of them near it. This combines a coarse
    sweep of the whole interval with a fine sweep concentrated where the
    posterior actually has mass, located using the Fisher-z approximation.

    Parameters
    ----------
    r : float
        Observed correlation.
    n : int
        Number of pairs.
    points : int, default 1201
        Approximate number of grid points in each of the two components.

    Returns
    -------
    ndarray
        Strictly increasing grid, strictly inside (-1, 1).
    """
    edge = 1.0 - 1e-9
    coarse = np.linspace(-edge, edge, points)

    # Fisher's z is roughly normal with sd 1/sqrt(n - 3); eight of those either
    # side covers the posterior with room to spare.
    z_hat = np.arctanh(np.clip(r, -0.999999, 0.999999))
    spread = 8.0 / np.sqrt(max(n - 3, 1))
    fine = np.tanh(np.linspace(z_hat - spread, z_hat + spread, points))

    combined = np.unique(np.clip(np.concatenate([coarse, fine]), -edge, edge))
    return combined


# ---------------------------------------------------------------------------
# Posterior and Bayes factor
# ---------------------------------------------------------------------------


def posterior_on_grid(
    r: float, n: int, kappa: float = DEFAULT_KAPPA, points: int = 1201
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate and normalise the posterior over the true correlation.

    Parameters
    ----------
    r : float
        Observed correlation.
    n : int
        Number of pairs.
    kappa : float, default 1.0
        Prior width.
    points : int, default 1201
        Grid resolution.

    Returns
    -------
    tuple of ndarray
        ``(grid, density)`` with the density normalised to integrate to 1.
    """
    grid = rho_grid(r, n, points)
    log_post = log_sampling_density(r, grid, n) + log_prior_density(grid, kappa)
    return grid, grid_utils.normalize_log_density(log_post, grid)


def log_bayes_factor(
    r: float, n: int, kappa: float = DEFAULT_KAPPA, points: int = 1201
) -> float:
    """Log Bayes factor for a real correlation against no correlation.

    Parameters
    ----------
    r : float
        Observed correlation.
    n : int
        Number of pairs.
    kappa : float, default 1.0
        Prior width. Wider priors weaken the evidence for a small observed
        correlation.
    points : int, default 1201
        Grid resolution for the marginal-likelihood integral.

    Returns
    -------
    float
        ``log(BF10)``. Positive favours a relationship; negative favours none.
    """
    grid = rho_grid(r, n, points)
    log_joint = log_sampling_density(r, grid, n) + log_prior_density(grid, kappa)
    log_null = float(log_sampling_density(r, 0.0, n))

    # Integrate in log space. Shifting by the peak before exponentiating keeps
    # the integrand near unity no matter how large the evidence is: with a few
    # thousand observations and a strong correlation, exp(log_joint - log_null)
    # overflows a float64 outright.
    peak = float(np.max(log_joint[np.isfinite(log_joint)]))
    integral = float(np.trapezoid(np.exp(log_joint - peak), grid))
    if not np.isfinite(integral) or integral <= 0:
        raise RuntimeError(
            f"the correlation Bayes factor integral did not converge for "
            f"r={r}, n={n}, kappa={kappa}."
        )
    return float(peak + np.log(integral) - log_null)
