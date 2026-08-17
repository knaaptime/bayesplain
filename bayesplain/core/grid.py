"""Grid approximation: the general fallback for awkward one-parameter posteriors.

Evaluate an unnormalised log density on a dense grid, normalise it by
numerical integration, and sample from it by inverting the cumulative
distribution. That handles any single-parameter posterior whose density can be
written down, whether or not it has a conjugate form -- the correlation
coefficient being the case this package needs it for.

It is also, deliberately, the same machinery behind
``bayesplain.teach.grid_posterior``. When students step through prior times
likelihood over a grid of candidate values in week 2, they are running the
production code path, not a simplified illustration of it.

Pure functions only.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "normalize_log_density",
    "grid_mean",
    "grid_quantile",
    "sample_from_grid",
]


def normalize_log_density(log_density, grid) -> np.ndarray:
    """Turn an unnormalised log density on a grid into a proper density.

    Subtracts the maximum before exponentiating, so densities that would
    otherwise underflow to zero are handled without loss of precision, then
    divides by the trapezoidal integral.

    Parameters
    ----------
    log_density : array_like
        Unnormalised log density evaluated at each grid point.
    grid : array_like
        Strictly increasing grid points.

    Returns
    -------
    ndarray
        Normalised density values, integrating to 1 over the grid.
    """
    grid = np.asarray(grid, dtype=float)
    log_density = np.asarray(log_density, dtype=float)
    if grid.shape != log_density.shape:
        raise ValueError(
            f"grid and log_density must have the same shape, got "
            f"{grid.shape} and {log_density.shape}."
        )
    if grid.size < 3:
        raise ValueError("grid needs at least 3 points.")
    if np.any(np.diff(grid) <= 0):
        raise ValueError("grid must be strictly increasing.")

    finite = np.isfinite(log_density)
    if not finite.any():
        raise ValueError(
            "log density is not finite anywhere on the grid; check the grid "
            "covers the region where the posterior has mass."
        )

    shifted = np.where(finite, log_density - log_density[finite].max(), -np.inf)
    density = np.exp(shifted)
    total = np.trapezoid(density, grid)
    if not np.isfinite(total) or total <= 0:
        raise ValueError("density did not integrate to a positive finite value.")
    return density / total


def _cumulative(density, grid) -> np.ndarray:
    """Trapezoidal CDF on the grid, forced to run from 0 to 1."""
    widths = np.diff(grid)
    midpoints = 0.5 * (density[1:] + density[:-1])
    cdf = np.concatenate([[0.0], np.cumsum(widths * midpoints)])
    return cdf / cdf[-1]


def grid_mean(density, grid) -> float:
    """Posterior mean by numerical integration on a grid.

    Parameters
    ----------
    density : array_like
        Normalised density values.
    grid : array_like
        Grid points.

    Returns
    -------
    float
        The integral of ``x * p(x)``.
    """
    grid = np.asarray(grid, dtype=float)
    density = np.asarray(density, dtype=float)
    return float(np.trapezoid(grid * density, grid))


def grid_quantile(density, grid, q) -> np.ndarray:
    """Quantiles of a grid-approximated posterior.

    Parameters
    ----------
    density : array_like
        Normalised density values.
    grid : array_like
        Grid points.
    q : float or array_like
        Probabilities in (0, 1).

    Returns
    -------
    ndarray
        Interpolated quantiles, scalar-shaped if ``q`` was scalar.
    """
    grid = np.asarray(grid, dtype=float)
    density = np.asarray(density, dtype=float)
    cdf = _cumulative(density, grid)
    out = np.interp(np.asarray(q, dtype=float), cdf, grid)
    return out


def sample_from_grid(
    density,
    grid,
    size: int = 100_000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample from a grid-approximated posterior by inverting its CDF.

    Draws uniforms and reads them back through the interpolated cumulative
    distribution. The draws are independent, so the only error is the
    discretisation of the grid, which a fine grid makes negligible.

    Parameters
    ----------
    density : array_like
        Normalised density values.
    grid : array_like
        Grid points.
    size : int, default 100_000
        Number of draws.
    rng : numpy.random.Generator, optional
        Random generator.

    Returns
    -------
    ndarray
        Array of ``size`` draws.
    """
    rng = np.random.default_rng() if rng is None else rng
    grid = np.asarray(grid, dtype=float)
    density = np.asarray(density, dtype=float)
    cdf = _cumulative(density, grid)
    return np.interp(rng.random(size), cdf, grid)
