r"""Gunel-Dickey Bayes factors and Dirichlet posteriors for contingency tables.

Two things live here, both exact:

1. **Bayes factors against independence** for an ``R x C`` table, under each of
   the four sampling schemes of Gunel and Dickey (1974). All four are ratios of
   gamma functions -- no integration anywhere.
2. **Posterior draws of the cell probabilities**, from which any effect-size
   measure (Cramer's V, a risk ratio, a difference of proportions) inherits a
   full posterior distribution.

The second is the one worth teaching. A frequentist chi-square hands back a
test statistic and a p-value but no distribution over *how strong* the
association is. Here that distribution is free.

The four sampling schemes
-------------------------
Which Bayes factor is correct depends on what the study design fixed in
advance, and the four differ by enough to matter -- roughly a full Jeffreys
category between the extremes.

``poisson``
    Nothing fixed. Cell counts arrive as they arrive.
``joint_multinomial``
    The grand total was fixed ("we sampled 715 workers").
``independent_multinomial``
    One margin was fixed ("we sampled 220 leases in District A and 240 in
    District B"). This is the default here, because it matches how planning
    data is usually gathered, and for a 2x2 table it reduces to a test of the
    equality of two proportions.
``hypergeometric``
    Both margins fixed. Rare in practice; arises after a median split.

Evidence against independence is largest under Poisson and smallest under
hypergeometric, since each successive scheme conditions away more of the data.

Notation
--------
Following Jamil et al. (2017), with :math:`y_{rc}` the cell counts,
:math:`a_{rc}` the prior parameters (all equal to :math:`a` by default), and

.. math::

    \mathcal{D}(\vec{v}) = \frac{\prod_i \Gamma(v_i)}{\Gamma(\sum_i v_i)}

the Dirichlet normalising function. Dots denote summation over a dimension.
The marginal prior parameters are :math:`\xi_{r\cdot} = a_{r\cdot} - (C-1)a`,
:math:`\xi_{\cdot c} = a_{\cdot c} - (R-1)a`, and
:math:`\xi_{\cdot\cdot} = a_{\cdot\cdot} - (R-1)(C-1)a`, which for a symmetric
prior make :math:`\xi_{r\cdot}` and :math:`\xi_{\cdot c}` vectors of
:math:`a`.

The independent-multinomial form used here
------------------------------------------
Equation 10 of Jamil et al. gives, for fixed row margins,

.. math::

    \mathrm{BF}_{01}^{I} =
        \frac{\mathcal{D}(y_{\cdot *} + \xi_{\cdot *})}{\mathcal{D}(\xi_{\cdot *})}
        \frac{\mathcal{D}(y_{* \cdot} + a_{* \cdot})}{\mathcal{D}(a_{* \cdot})}
        \frac{\mathcal{D}(a_{**})}{\mathcal{D}(y_{**} + a_{**})}

This module instead evaluates the algebraically equivalent

.. math::

    \mathrm{BF}_{10}^{I} = \frac{\prod_r \mathcal{D}(\vec{a} + y_{r*})}
                                {\mathcal{D}(\vec{a})^{R-1}\,
                                 \mathcal{D}(\vec{a} + y_{\cdot *})}

which drops out of the direct construction -- under dependence each row draws
its own column distribution from :math:`\mathrm{Dirichlet}(\vec{a})`, under
independence they share one. Expanding both expressions in gamma functions,
every term involving :math:`\Gamma(RCa)` and :math:`\Gamma(y_{\cdot\cdot}+RCa)`
cancels and the two agree exactly; this is checked numerically in the test
suite against the published value of Jamil et al.'s doll-preference example.
The advantage of this form is that it accepts a *per-column* concentration
vector, so a ``Beta(a, b)`` prior on each group's rate can be carried into the
Bayes factor unchanged. The other three schemes are implemented directly from
the paper and require a scalar concentration.

References
----------
Gunel, E. and Dickey, J. (1974). Bayes factors for independence in
contingency tables. *Biometrika*, 61(3), 545-557.

Jamil, T., Ly, A., Morey, R. D., Love, J., Marsman, M., and Wagenmakers, E.-J.
(2017). Default "Gunel and Dickey" Bayes factors for contingency tables.
*Behavior Research Methods*, 49(2), 638-652.
"""

from __future__ import annotations

import numpy as np
from scipy import special

__all__ = [
    "SCHEMES",
    "log_multivariate_beta",
    "log_bayes_factor_independence",
    "posterior_cell_draws",
    "cramers_v",
    "log_odds_ratio",
    "validate_table",
]

#: The four Gunel-Dickey sampling schemes, in order of increasing restriction.
SCHEMES = (
    "poisson",
    "joint_multinomial",
    "independent_multinomial",
    "hypergeometric",
)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_table(table) -> np.ndarray:
    """Check that an array is a usable table of counts.

    Parameters
    ----------
    table : array_like
        Two-dimensional array of non-negative whole numbers.

    Returns
    -------
    ndarray
        The validated table as a float array of shape ``(R, C)``.

    Raises
    ------
    ValueError
        If the table is not 2-D, contains negative or non-integral values, is
        smaller than 2x2, or has an entirely empty row or column.
    """
    arr = np.asarray(table, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            f"table must be two-dimensional, got an array with {arr.ndim} "
            "dimension(s). Rows are the groups you sampled; columns are the "
            "categories you counted."
        )
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        raise ValueError(
            f"table must be at least 2x2, got {arr.shape[0]}x{arr.shape[1]}."
        )
    if np.any(arr < 0):
        raise ValueError("table cannot contain negative counts.")
    if not np.allclose(arr, np.round(arr)):
        raise ValueError("table must contain whole-number counts, not rates.")
    if np.any(arr.sum(axis=1) == 0):
        raise ValueError("every row of the table must contain at least one count.")
    if np.any(arr.sum(axis=0) == 0):
        raise ValueError(
            "every column of the table must contain at least one count; drop "
            "categories that were never observed."
        )
    return np.round(arr)


def _concentration_vector(concentration, n_cols: int) -> np.ndarray:
    """Broadcast a scalar or vector concentration to one value per column."""
    arr = np.asarray(concentration, dtype=float)
    if arr.ndim == 0:
        arr = np.full(n_cols, float(arr))
    elif arr.ndim == 1:
        if arr.size != n_cols:
            raise ValueError(
                f"concentration has {arr.size} entries but the table has "
                f"{n_cols} columns; pass one value per column or a single "
                "value for all of them."
            )
    else:
        raise ValueError("concentration must be a scalar or a 1-D array.")
    if np.any(arr <= 0):
        raise ValueError(f"concentration must be positive, got {concentration!r}.")
    return arr


def _scalar_concentration(concentration, scheme: str) -> float:
    """Require a scalar concentration for the schemes that only define one."""
    arr = np.asarray(concentration, dtype=float)
    if arr.ndim != 0:
        if arr.size == 1:
            arr = arr.reshape(())
        elif np.allclose(arr, arr.flat[0]):
            arr = np.asarray(float(arr.flat[0]))
        else:
            raise ValueError(
                f"the {scheme!r} scheme is defined for a single symmetric "
                "concentration only, but an unequal per-column vector was "
                "given. Use scheme='independent_multinomial', which accepts "
                "one, or pass a scalar."
            )
    value = float(arr)
    if value <= 0:
        raise ValueError(f"concentration must be positive, got {value}.")
    return value


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def log_multivariate_beta(alpha) -> float:
    r"""Log of the Dirichlet normalising function.

    .. math::

        \mathcal{D}(\alpha) = \frac{\prod_j \Gamma(\alpha_j)}
                                   {\Gamma(\sum_j \alpha_j)}

    Parameters
    ----------
    alpha : array_like
        Positive concentration parameters.

    Returns
    -------
    float
        ``log D(alpha)``.
    """
    alpha = np.asarray(alpha, dtype=float)
    if np.any(alpha <= 0):
        raise ValueError("Dirichlet concentration parameters must be positive.")
    return float(special.gammaln(alpha).sum() - special.gammaln(alpha.sum()))


def _log_multinomial_coefficient(counts, total: float) -> float:
    """Log of ``total! / prod(counts!)``."""
    counts = np.asarray(counts, dtype=float)
    return float(special.gammaln(total + 1.0) - special.gammaln(counts + 1.0).sum())


# ---------------------------------------------------------------------------
# The four Bayes factors
# ---------------------------------------------------------------------------


def _log_bf10_independent_multinomial(counts, concentration, fixed) -> float:
    """Equation 10, in the equivalent per-row Dirichlet form."""
    if fixed == "columns":
        counts = counts.T
    elif fixed != "rows":
        raise ValueError(f"fixed must be 'rows' or 'columns', got {fixed!r}.")

    n_rows, n_cols = counts.shape
    a_vec = _concentration_vector(concentration, n_cols)

    log_alt = sum(log_multivariate_beta(a_vec + counts[i]) for i in range(n_rows))
    log_null = log_multivariate_beta(a_vec + counts.sum(axis=0))
    return float(log_alt - (n_rows - 1) * log_multivariate_beta(a_vec) - log_null)


def _log_bf10_joint_multinomial(counts, a: float) -> float:
    """Equation 8: the grand total was fixed."""
    n_rows, n_cols = counts.shape
    xi_row, xi_col = np.full(n_rows, a), np.full(n_cols, a)
    a_mat = np.full((n_rows, n_cols), a)

    log_bf01 = (
        log_multivariate_beta(counts.sum(axis=1) + xi_row)
        - log_multivariate_beta(xi_row)
        + log_multivariate_beta(counts.sum(axis=0) + xi_col)
        - log_multivariate_beta(xi_col)
        + log_multivariate_beta(a_mat.ravel())
        - log_multivariate_beta((counts + a_mat).ravel())
    )
    return float(-log_bf01)


def _log_bf10_poisson(counts, a: float) -> float:
    """Equation 4.2 of GD74: nothing fixed in advance."""
    n_rows, n_cols = counts.shape
    total = counts.sum()
    a_mat = np.full((n_rows, n_cols), a)
    xi_row, xi_col = np.full(n_rows, a), np.full(n_cols, a)
    xi_total = a * n_rows * n_cols - (n_rows - 1) * (n_cols - 1) * a

    # GD74's default gamma scale parameter.
    scale = n_rows * n_cols * a / total

    log_bf01 = (
        (n_rows - 1) * (n_cols - 1) * np.log1p(1.0 / scale)
        + special.gammaln(total + xi_total)
        - special.gammaln(xi_total)
        + (special.gammaln(a_mat) - special.gammaln(counts + a_mat)).sum()
        + log_multivariate_beta(counts.sum(axis=1) + xi_row)
        - log_multivariate_beta(xi_row)
        + log_multivariate_beta(counts.sum(axis=0) + xi_col)
        - log_multivariate_beta(xi_col)
    )
    return float(-log_bf01)


def _log_bf10_hypergeometric(counts, a: float) -> float:
    """Equation 15: both margins fixed, summing over all tables with them."""
    if counts.shape != (2, 2):
        raise ValueError(
            "the hypergeometric scheme is implemented for 2x2 tables only, "
            "where fixing both margins leaves a single free cell. For a larger "
            "table the sum runs over every table sharing the margins, which is "
            "rarely what a study design actually justifies — use "
            "scheme='independent_multinomial' instead."
        )
    row_totals, col_totals = counts.sum(axis=1), counts.sum(axis=0)
    total = counts.sum()
    a_mat = np.full((2, 2), a)

    low = int(max(0.0, row_totals[0] - col_totals[1]))
    high = int(min(row_totals[0], col_totals[0]))
    terms = []
    for k in range(low, high + 1):
        candidate = np.array(
            [
                [k, row_totals[0] - k],
                [col_totals[0] - k, row_totals[1] - col_totals[0] + k],
            ],
            dtype=float,
        )
        if (candidate < 0).any():
            continue
        terms.append(
            _log_multinomial_coefficient(candidate.ravel(), total)
            + log_multivariate_beta((candidate + a_mat).ravel())
        )

    log_bf01 = (
        special.logsumexp(terms)
        - log_multivariate_beta((counts + a_mat).ravel())
        - _log_multinomial_coefficient(row_totals, total)
        - _log_multinomial_coefficient(col_totals, total)
    )
    return float(-log_bf01)


def log_bayes_factor_independence(
    table,
    concentration=1.0,
    scheme: str = "independent_multinomial",
    fixed: str = "rows",
) -> float:
    """Log Bayes factor for association against independence.

    Parameters
    ----------
    table : array_like
        Table of counts, shape ``(R, C)``.
    concentration : float or array_like, default 1.0
        Dirichlet concentration ``a``. ``a = 1`` makes every combination of
        parameter values equally likely a priori; larger values pull the
        alternative's predictions toward the null, so ``a = 10`` behaves like
        a flat prior already updated with nine hypothetical observations per
        cell. For ``scheme='independent_multinomial'`` a per-column vector is
        also accepted, which is how a ``Beta(a, b)`` prior on a rate is carried
        into the Bayes factor unchanged.
    scheme : str, default 'independent_multinomial'
        One of :data:`SCHEMES`. Pick the one matching what the study design
        fixed in advance; see the module docstring.
    fixed : {'rows', 'columns'}, default 'rows'
        Which margin was fixed, for the independent-multinomial scheme only.
        Rows are the groups you sampled.

    Returns
    -------
    float
        ``log(BF10)``. Positive values favour association between rows and
        columns; negative values favour independence.

    Notes
    -----
    Exact -- gamma functions only, with no integration. Validated against the
    published values in Jamil et al. (2017); see
    ``bayesplain/tests/test_dirichlet_multinomial.py``.
    """
    counts = validate_table(table)
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {list(SCHEMES)}, got {scheme!r}.")

    if scheme == "independent_multinomial":
        return _log_bf10_independent_multinomial(counts, concentration, fixed)

    a = _scalar_concentration(concentration, scheme)
    if scheme == "joint_multinomial":
        return _log_bf10_joint_multinomial(counts, a)
    if scheme == "poisson":
        return _log_bf10_poisson(counts, a)
    return _log_bf10_hypergeometric(counts, a)


# ---------------------------------------------------------------------------
# Posterior over the table
# ---------------------------------------------------------------------------


def posterior_cell_draws(
    table,
    concentration=1.0,
    size: int = 100_000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw whole tables of cell probabilities from the Dirichlet posterior.

    With a ``Dirichlet(a)`` prior over the full set of cells, the posterior is
    ``Dirichlet(a + counts)`` -- conjugate, so again just adding counts to
    concentration parameters. Each draw is a complete table of probabilities
    summing to one, so any quantity computed from a table can be computed once
    per draw to get its posterior distribution.

    Parameters
    ----------
    table : array_like
        Table of counts, shape ``(R, C)``.
    concentration : float or array_like, default 1.0
        Dirichlet concentration, scalar or one value per column.
    size : int, default 100_000
        Number of tables to draw.
    rng : numpy.random.Generator, optional
        Random generator.

    Returns
    -------
    ndarray
        Array of shape ``(size, R, C)``, each slice a table of probabilities
        summing to 1.
    """
    counts = validate_table(table)
    a_vec = _concentration_vector(concentration, counts.shape[1])
    rng = np.random.default_rng() if rng is None else rng

    alpha = (counts + a_vec[None, :]).ravel()
    flat = rng.dirichlet(alpha, size=size)
    return flat.reshape(size, *counts.shape)


# ---------------------------------------------------------------------------
# Effect sizes, which inherit a posterior from the draws above
# ---------------------------------------------------------------------------


def cramers_v(prob_tables) -> np.ndarray:
    r"""Cramer's V for one or many tables of probabilities.

    Computed from probabilities rather than counts, so it does not depend on
    sample size:

    .. math::

        \phi^2 = \sum_{ij} \frac{(p_{ij} - p_{i\cdot}p_{\cdot j})^2}
                                {p_{i\cdot}p_{\cdot j}}, \qquad
        V = \sqrt{\frac{\phi^2}{\min(R-1,\, C-1)}}

    Applied to draws from :func:`posterior_cell_draws`, this yields a posterior
    distribution over association strength -- the thing a chi-square test
    cannot give you.

    Parameters
    ----------
    prob_tables : array_like
        Either a single ``(R, C)`` table of probabilities summing to 1, or a
        stack of shape ``(size, R, C)``.

    Returns
    -------
    ndarray
        Cramer's V, scalar-shaped for a single table and shape ``(size,)``
        for a stack. Ranges from 0 (exact independence) to 1.
    """
    arr = np.asarray(prob_tables, dtype=float)
    single = arr.ndim == 2
    if single:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(
            f"expected a (R, C) table or a (size, R, C) stack, got shape "
            f"{np.shape(prob_tables)}."
        )

    row_marg = arr.sum(axis=2, keepdims=True)
    col_marg = arr.sum(axis=1, keepdims=True)
    expected = row_marg * col_marg

    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(expected > 0, (arr - expected) ** 2 / expected, 0.0)
    phi2 = terms.sum(axis=(1, 2))

    n_rows, n_cols = arr.shape[1], arr.shape[2]
    v = np.sqrt(phi2 / min(n_rows - 1, n_cols - 1))
    return v[0] if single else v


def log_odds_ratio(prob_tables) -> np.ndarray:
    """Log odds ratio of a 2x2 table of probabilities, or a stack of them.

    Independence corresponds to a log odds ratio of zero, which makes this the
    natural effect size when the table is 2x2 and the reason Jamil et al.
    report posteriors on this scale.

    Parameters
    ----------
    prob_tables : array_like
        A single ``(2, 2)`` table of probabilities, or a stack of shape
        ``(size, 2, 2)``.

    Returns
    -------
    ndarray
        ``log((p00 * p11) / (p01 * p10))``, scalar-shaped for a single table.
    """
    arr = np.asarray(prob_tables, dtype=float)
    single = arr.ndim == 2
    if single:
        arr = arr[None, ...]
    if arr.shape[1:] != (2, 2):
        raise ValueError(
            f"the odds ratio is only defined for a 2x2 table, got shape "
            f"{arr.shape[1:]}. Use cramers_v for larger tables."
        )
    with np.errstate(divide="ignore"):
        out = (
            np.log(arr[:, 0, 0])
            + np.log(arr[:, 1, 1])
            - np.log(arr[:, 0, 1])
            - np.log(arr[:, 1, 0])
        )
    return out[0] if single else out
