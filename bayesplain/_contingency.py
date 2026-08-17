"""Categorical breakdowns, and whether a table's rows and columns are related.

The frequentist counterpart is the chi-square test of independence (week 7),
and this is the analysis where the Bayesian version has the clearest advantage.
A chi-square test returns a statistic and a p-value; ask it *how strong* the
association is and it has nothing to say beyond a point estimate with no
uncertainty attached. Drawing from the Dirichlet posterior over cell
probabilities and computing an effect size once per draw gives that quantity a
full posterior, in closed form, for free.

The second advantage shows up in sparse tables. The chi-square approximation
needs roughly five expected cases per cell and warns below it; the posterior
does not degrade there at all, it simply gets wider, which is the honest
response to thin data.
"""

from __future__ import annotations

import numpy as np

from . import frequentist, priors
from ._config import get_draws, make_rng
from .core import dirichlet_multinomial as dm
from .result import Result

__all__ = ["contingency"]

#: Conventional bar for "at least a small association", used as the default
#: threshold for Cramer's V. Unlike a difference or a log odds ratio, V cannot
#: be negative and a posterior never puts mass at exactly zero, so P(V > 0) is
#: always 1 and tells you nothing. A conventional floor is the honest
#: replacement, and naming it out loud is better than implying a null exists.
SMALL_EFFECT_V = 0.1


def contingency(
    table,
    prior="uninformed",
    scheme: str = "independent_multinomial",
    fixed: str = "rows",
    effect: str = "auto",
    threshold: float | None = None,
    row_labels=None,
    col_labels=None,
    n_draws: int | None = None,
    seed="unset",
) -> Result:
    """Test whether the rows and columns of a table are related.

    Parameters
    ----------
    table : array_like
        Counts, shape ``(R, C)``. Rows are the groups you sampled; columns are
        the categories you counted within each group. A pandas crosstab works
        directly, and its labels are picked up automatically.
    prior : str, float, or ConcentrationPrior, default 'uninformed'
        Dirichlet concentration over the cells. See
        :func:`bayesplain.priors.describe`.
    scheme : str, default 'independent_multinomial'
        Which Gunel-Dickey sampling scheme the study design justifies; one of
        :data:`bayesplain.core.dirichlet_multinomial.SCHEMES`. The default
        matches "we sampled a fixed number of cases in each group", which is
        how planning data is usually gathered. The choice only affects the
        Bayes factor, never the posterior over the effect size.
    fixed : {'rows', 'columns'}, default 'rows'
        Which margin the design fixed, for the independent-multinomial scheme.
    effect : {'auto', 'cramers_v', 'log_odds_ratio'}, default 'auto'
        The effect size to put a posterior on. ``'auto'`` uses the log odds
        ratio for a 2x2 table, where zero means exactly no association, and
        Cramer's V otherwise.
    threshold : float, optional
        The value the reported probability is measured against. Defaults to 0
        for the log odds ratio and to :data:`SMALL_EFFECT_V` for Cramer's V.
    row_labels, col_labels : sequence of str, optional
        Names for the rows and columns. Read from a pandas object when not
        supplied.
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
    The doll-preference study of Hraba and Grant (1970), as analysed by Jamil
    et al. (2017): 62 of 89 African American children preferred a black doll,
    while 60 of 71 white children preferred a white doll.

    >>> import bayesplain as bp
    >>> res = bp.contingency(
    ...     [[62, 27], [11, 60]],
    ...     row_labels=["Black children", "white children"],
    ...     col_labels=["black doll", "white doll"],
    ... )
    >>> round(res.bayes_factor().log_bf10, 2)
    23.03
    >>> res.frequentist.pvalue < 0.001
    True

    Notes
    -----
    The Bayes factor is validated against the published values in Jamil et al.
    (2017) for all four sampling schemes.
    """
    row_labels, col_labels, counts = _read_table(table, row_labels, col_labels)
    resolved = priors.resolve_table(prior)
    n_draws = get_draws() if n_draws is None else int(n_draws)
    rng = make_rng(seed)

    n_rows, n_cols = counts.shape
    if effect == "auto":
        effect = "log_odds_ratio" if counts.shape == (2, 2) else "cramers_v"
    if effect not in {"cramers_v", "log_odds_ratio"}:
        raise ValueError(
            f"effect must be 'auto', 'cramers_v', or 'log_odds_ratio', got {effect!r}."
        )
    if effect == "log_odds_ratio" and counts.shape != (2, 2):
        raise ValueError(
            f"the log odds ratio is only defined for a 2x2 table, but this one "
            f"is {n_rows}x{n_cols}. Use effect='cramers_v'."
        )

    cells = dm.posterior_cell_draws(
        counts, concentration=resolved.a, size=n_draws, rng=rng
    )
    if effect == "log_odds_ratio":
        draws = dm.log_odds_ratio(cells)
        reference = 0.0 if threshold is None else float(threshold)
        quantity = (
            f"log odds ratio — {col_labels[0]} vs {col_labels[1]}, "
            f"across {row_labels[0]} and {row_labels[1]}"
        )
        subject = "log odds ratio"
        unit, scale, decimals = "", 1.0, 2
    else:
        draws = dm.cramers_v(cells)
        reference = SMALL_EFFECT_V if threshold is None else float(threshold)
        quantity = (
            f"strength of association (Cramér's V) between "
            f"{'rows' if n_rows > 2 else ' and '.join(row_labels)} and "
            f"{'columns' if n_cols > 2 else ' and '.join(col_labels)}"
        )
        subject = "association strength"
        unit, scale, decimals = "", 1.0, 3

    log_bf10 = dm.log_bayes_factor_independence(
        counts, concentration=resolved.a, scheme=scheme, fixed=fixed
    )
    twin = frequentist.chi_square_independence(counts)

    notes = _build_notes(counts, twin, effect, reference)

    def _refit(spec):
        return contingency(
            counts,
            prior=spec,
            scheme=scheme,
            fixed=fixed,
            effect=effect,
            threshold=reference,
            row_labels=row_labels,
            col_labels=col_labels,
            n_draws=n_draws,
            seed=seed,
        )

    return Result(
        quantity=quantity,
        draws=draws,
        posterior=None,  # no closed form for an effect size built from a table
        prior=resolved,
        subject=subject,
        analysis="contingency",
        frequentist=twin,
        log_bf10=log_bf10,
        bf_alternative="an association",
        bf_caveat=(
            f"Compares 'rows and columns are related' against 'they are "
            f"independent', under the {scheme.replace('_', ' ')} sampling "
            f"scheme. Which scheme is right depends on what your design fixed "
            f"in advance, and the four differ by roughly a full evidence "
            f"category. Prior-sensitive; run .sensitivity()."
        ),
        unit=unit,
        display_scale=scale,
        decimals=decimals,
        direction_reference=reference,
        components=_row_profiles(cells, row_labels),
        component_axis="share of each row falling in the first column (%)",
        component_scale=100.0,
        refit=_refit,
        prior_ladder=priors.SENSITIVITY_LADDER,
        n_draws=n_draws,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_table(table, row_labels, col_labels):
    """Pull counts and labels out of an array or a pandas crosstab."""
    if hasattr(table, "columns") and hasattr(table, "index"):
        if row_labels is None:
            row_labels = [str(v) for v in table.index]
        if col_labels is None:
            col_labels = [str(v) for v in table.columns]
        values = table.to_numpy()
    else:
        values = table

    counts = dm.validate_table(values)
    n_rows, n_cols = counts.shape

    row_labels = (
        [f"row {i + 1}" for i in range(n_rows)]
        if row_labels is None
        else [str(v) for v in row_labels]
    )
    col_labels = (
        [f"column {j + 1}" for j in range(n_cols)]
        if col_labels is None
        else [str(v) for v in col_labels]
    )
    if len(row_labels) != n_rows:
        raise ValueError(
            f"row_labels has {len(row_labels)} names but the table has {n_rows} rows."
        )
    if len(col_labels) != n_cols:
        raise ValueError(
            f"col_labels has {len(col_labels)} names but the table has "
            f"{n_cols} columns."
        )
    return row_labels, col_labels, counts


def _row_profiles(cells, row_labels):
    """Posterior draws of each row's share in the first column.

    Gives the components plot something concrete to show: the rows are only
    "related" to the columns insofar as their category profiles differ, so
    seeing those profiles side by side is what makes an association legible.
    """
    row_totals = cells.sum(axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        shares = np.where(row_totals > 0, cells[:, :, 0] / row_totals, np.nan)
    return {label: shares[:, i] for i, label in enumerate(row_labels)}


def _build_notes(counts, twin, effect, reference):
    """Warnings that matter for reading this particular table."""
    notes = []
    expected = _expected_counts(counts)
    sparse = int((expected < 5).sum())
    if sparse:
        notes.append(
            f"{sparse} of {counts.size} cells have expected counts below 5, "
            "where the chi-square approximation is unreliable. The posterior "
            "is unaffected — it just gets wider, which is the honest response "
            "to thin data."
        )
    if effect == "cramers_v":
        notes.append(
            f"Cramér's V cannot be negative, so P(above {reference:g}) is "
            "measured against a conventional 'small association' bar rather "
            "than a null. Set threshold= to a value that matters for your "
            "decision instead."
        )
    if counts.sum() < 40:
        notes.append(
            f"only {int(counts.sum())} observations in total, so the prior is "
            "doing visible work here — run .sensitivity()"
        )
    return notes


def _expected_counts(counts):
    """Compute the expected counts under independence."""
    total = counts.sum()
    return np.outer(counts.sum(axis=1), counts.sum(axis=0)) / total
