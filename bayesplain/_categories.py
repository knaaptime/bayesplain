"""One categorical variable, and whether its shares match what you expected.

The frequentist counterpart is the chi-square goodness-of-fit test: are
collisions spread evenly across the days of the week, do a program's
enrollees match the neighbourhood's demographics, did this year's complaints
fall into the same categories as last year's. It completes the chi-square half
of an introductory course alongside :func:`bayesplain.contingency`.

As with a table, the test hands back a statistic and a p-value and nothing
about *which* categories are off or by how much. The Dirichlet posterior over
the shares answers both: each category's share gets its own exact interval, and
an overall effect size -- Cohen's w -- gets a full posterior.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy import stats

from . import frequentist, priors
from ._config import get_draws, make_rng
from .core import dirichlet_multinomial as dm
from .result import Result, _Report, _wrap

__all__ = ["categories", "SMALL_EFFECT_W"]

#: Cohen's conventional bar for a small departure from the expected shares.
#: Like Cramer's V, w cannot be negative, so a posterior puts no mass at zero
#: and P(w > 0) says nothing; a named floor is the honest replacement.
SMALL_EFFECT_W = 0.1


def categories(
    data,
    expected=None,
    prior="uninformed",
    labels=None,
    threshold: float | None = None,
    n_draws: int | None = None,
    seed="unset",
) -> Result:
    """Compare how one categorical variable splits against expected shares.

    Parameters
    ----------
    data : array_like, mapping, or pandas Series
        Either the counts in each category -- a list of numbers, a
        ``{category: count}`` mapping, or the output of
        ``Series.value_counts()`` -- or the raw observations themselves, one
        label per case, which are tallied for you.
    expected : array_like or mapping, optional
        The shares each category should hold. Defaults to equal shares.
        Given as shares or as counts (it is normalised), in the same order as
        the categories, or as a ``{category: share}`` mapping.
    prior : str, float, or ConcentrationPrior, default 'uninformed'
        Dirichlet concentration over the category shares, centred on the
        expected shares: ``'uninformed'`` is worth one observation per
        category, spread in the expected proportions. See
        :func:`bayesplain.priors.describe`.
    labels : sequence of str, optional
        Category names, when they cannot be read from ``data``.
    threshold : float, optional
        The value of Cohen's w the reported probability is measured against.
        Defaults to the larger of Cohen's "small" bar, 0.1, and the *noise
        floor*: the w that data matching the expected shares exactly would
        still produce at this sample size. Without the floor, a modest sample
        spread over many categories reports a confident departure from pure
        noise, because w adds up the uncertainty in every share and nothing
        cancels.
    n_draws : int, optional
        Number of draws. Defaults to the package setting.
    seed : int or None, optional
        Seed for the draws.

    Returns
    -------
    Result
        The headline is Cohen's w, the overall size of the departure. Call
        ``.shares()`` for the table this analysis is really about: each
        category's share, with an interval, against what was expected.

    Examples
    --------
    Collisions by day of the week, against an even split:

    >>> import bayesplain as bp
    >>> days = {"Mon": 31, "Tue": 28, "Wed": 30, "Thu": 33,
    ...         "Fri": 45, "Sat": 38, "Sun": 25}
    >>> res = bp.categories(days)
    >>> res.frequentist.pvalue < 0.05
    False
    >>> round(res.shares_table["Fri"]["probability_above_expected"], 2)
    0.98

    The omnibus test sees nothing, but one category clearly stands out --
    which is what ``.shares()`` is for.
    """
    names, counts = _read_counts(data, labels)
    shares0 = _read_expected(expected, names)
    resolved = priors.resolve_table(prior)
    n_draws = get_draws() if n_draws is None else int(n_draws)
    rng = make_rng(seed)

    # Centred on the expected shares, with the same total weight as a
    # symmetric Dirichlet(a): for an even split the two are identical.
    prior_alpha = resolved.a * len(names) * shares0
    alpha = counts + prior_alpha
    draws_shares = rng.dirichlet(alpha, size=n_draws)
    w_draws = dm.cohens_w(draws_shares, shares0)

    log_bf10 = dm.log_bayes_factor_goodness_of_fit(counts, shares0, prior_alpha)
    twin = frequentist.goodness_of_fit(counts, shares0)
    total = float(counts.sum())
    floor = _noise_floor(total, shares0, prior_alpha)
    if threshold is None:
        threshold = max(SMALL_EFFECT_W, floor)

    marginals = {
        name: stats.beta(alpha[i], alpha.sum() - alpha[i])
        for i, name in enumerate(names)
    }
    table = _shares_table(names, counts, shares0, marginals)

    notes = [
        f"Cohen's w cannot be negative, so P(above {threshold:.3g}) is "
        "measured against a bar rather than a null. Call .shares() to see "
        "which categories are off, and by how much.",
        f"at this sample size, data matching the expected shares exactly "
        f"would still give w of about {floor:.3f} — every share is estimated "
        "with some noise, and w adds that noise up without letting any of it "
        "cancel. Read the interval against that floor, not against zero"
        + ("; the bar above is set to it." if threshold == floor else "."),
    ]
    sparse = int((total * shares0 < 5).sum())
    if sparse:
        notes.append(
            f"{sparse} of {len(names)} categories have expected counts below 5, "
            "where the chi-square approximation is unreliable. The posterior is "
            "unaffected — it just gets wider."
        )
    if total < 40:
        notes.append(
            f"only {int(total)} observations in total, so the prior is doing "
            "visible work here — run .sensitivity()"
        )

    def _refit(spec):
        return categories(
            dict(zip(names, counts)),
            expected=dict(zip(names, shares0)),
            prior=spec,
            threshold=threshold,
            n_draws=n_draws,
            seed=seed,
        )

    even = np.allclose(shares0, shares0[0])
    result = Result(
        quantity=(
            f"departure from {'an even split' if even else 'the expected shares'} "
            f"across {len(names)} categories (Cohen's w)"
        ),
        draws=w_draws,
        posterior=None,  # no closed form for w built from a Dirichlet
        prior=resolved,
        subject="departure from the expected shares",
        analysis="categories",
        frequentist=twin,
        log_bf10=log_bf10,
        bf_alternative="shares different from the expected ones",
        bf_caveat=(
            "Compares 'the shares are unknown' against 'they are exactly the "
            "expected ones'. Exact shares are rarely what anyone believes, and "
            "this number moves with the prior. Run .sensitivity()."
        ),
        unit="",
        display_scale=1.0,
        decimals=3,
        direction_reference=float(threshold),
        reference_is_null=False,
        components=marginals,
        component_axis="share of each category (%)",
        component_scale=100.0,
        custom_plots={"shares": _shares_plot(names, shares0, marginals)},
        refit=_refit,
        prior_ladder=priors.SENSITIVITY_LADDER,
        n_draws=n_draws,
        notes=notes,
        next_steps=".shares()  .plot(kind='shares')  .sensitivity()",
        no_predict_reason=(
            "categories() does not forecast; use .shares() for each "
            "category's estimated share."
        ),
    )
    result.shares_table = table
    result.shares = _make_shares_report(table)
    return result


def _noise_floor(total, shares0, prior_alpha) -> float:
    """Median posterior w when the data match the expected shares exactly.

    Uses its own fixed generator so that the floor, which is a property of
    the design rather than of the data, never depends on the seed.
    """
    rng = np.random.default_rng(0)
    draws = rng.dirichlet(total * shares0 + prior_alpha, size=20_000)
    return float(np.median(dm.cohens_w(draws, shares0)))


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


def _read_counts(data, labels):
    """Return category names and counts from any accepted input shape."""
    if isinstance(data, Mapping):
        names = [str(k) for k in data]
        values = np.asarray(list(data.values()), dtype=float)
    else:
        # Only a pandas object carries category names in its index; a plain
        # list has an .index *method*, which must not be mistaken for one.
        is_pandas = hasattr(data, "to_numpy")
        index = data.index if is_pandas else None
        raw = np.asarray(data.to_numpy() if is_pandas else data).ravel()
        if raw.dtype.kind in "biuf":
            values = raw.astype(float)
            if labels is not None:
                names = [str(v) for v in labels]
            elif index is not None and not _is_default_index(index, raw.size):
                names = [str(v) for v in index]
            else:
                names = [f"category {i + 1}" for i in range(raw.size)]
        else:
            # Raw observations: one label per case, to be tallied.
            present = np.array([v for v in raw if v is not None and v == v])
            found, values = np.unique(present.astype(str), return_counts=True)
            values = values.astype(float)
            names = [str(v) for v in found]
            if labels is not None:
                order = [str(v) for v in labels]
                missing = set(order) - set(names)
                lookup = dict(zip(names, values))
                values = np.array([lookup.get(k, 0.0) for k in order])
                names = order
                if set(found) - set(order):
                    extra = ", ".join(sorted(set(found) - set(order)))
                    raise ValueError(
                        f"the data contain categories not in labels: {extra}."
                    )
                del missing

    if len(names) != values.size:
        raise ValueError(
            f"labels has {len(names)} names but there are {values.size} counts."
        )
    if values.size < 2:
        raise ValueError(
            f"need at least two categories to compare shares, got {values.size}."
        )
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("every count must be a non-negative number.")
    if np.any(values != np.round(values)):
        raise ValueError(
            "counts must be whole numbers. If you have shares, multiply them by "
            "the number of observations first."
        )
    if values.sum() == 0:
        raise ValueError("every count is zero, so there is nothing to analyse.")
    return names, values


def _is_default_index(index, size: int) -> bool:
    """Whether a pandas index is just 0..n-1, carrying no category names."""
    try:
        return list(index) == list(range(size))
    except TypeError:  # pragma: no cover - exotic index types
        return False


def _read_expected(expected, names):
    """Return the expected shares in category order, normalised to sum to one."""
    if expected is None:
        return np.full(len(names), 1.0 / len(names))
    if isinstance(expected, Mapping):
        keys = {str(k): v for k, v in expected.items()}
        missing = [n for n in names if n not in keys]
        if missing:
            raise ValueError(
                f"expected has no share for: {', '.join(missing)}. Give one for "
                "every category."
            )
        shares = np.array([float(keys[n]) for n in names])
    else:
        shares = np.asarray(expected, dtype=float).ravel()
    if shares.size != len(names):
        raise ValueError(
            f"expected has {shares.size} shares but there are {len(names)} categories."
        )
    if np.any(~np.isfinite(shares)) or np.any(shares <= 0):
        raise ValueError(
            "every expected share must be positive: a category expected to be "
            "empty cannot be tested against, because a single case would rule "
            "the null out entirely."
        )
    return shares / shares.sum()


# ---------------------------------------------------------------------------
# The per-category view, which is the point of this analysis
# ---------------------------------------------------------------------------


def _shares_table(names, counts, shares0, marginals, level: float = 0.95):
    """Per-category numbers, as a nested dict for autograding and printing."""
    total = counts.sum()
    tail = (1.0 - level) / 2.0
    out = {}
    for i, name in enumerate(names):
        post = marginals[name]
        out[name] = {
            "observed": int(counts[i]),
            "observed_share": float(counts[i] / total),
            "expected_share": float(shares0[i]),
            "share_median": float(post.median()),
            "interval_low": float(post.ppf(tail)),
            "interval_high": float(post.ppf(1.0 - tail)),
            "probability_above_expected": float(post.sf(shares0[i])),
        }
    return out


def _make_shares_report(table):
    """Build the ``.shares()`` method bound to this result."""

    def shares() -> _Report:
        """Each category's share, with its interval, against the expected share.

        Returns
        -------
        _Report
            One row per category. Categories whose interval excludes the
            expected share are marked.
        """
        header = (
            f"{'category':<18}{'count':>7}{'expected':>10}{'estimate':>10}"
            f"{'95% interval':>18}{'P(above)':>10}"
        )
        lines = ["EACH CATEGORY AGAINST ITS EXPECTED SHARE", "", header, "-" * 74]
        for name, row in table.items():
            span = f"{row['interval_low']:.1%} to {row['interval_high']:.1%}"
            off = (
                not row["interval_low"] <= row["expected_share"] <= row["interval_high"]
            )
            lines.append(
                f"{name[:17]:<18}{row['observed']:>7,}{row['expected_share']:>10.1%}"
                f"{row['share_median']:>10.1%}{span:>18}"
                f"{row['probability_above_expected']:>10.3f}" + ("  *" if off else "")
            )
        lines += [""]
        lines += _wrap(
            "* the 95% interval excludes the expected share. P(above) is the "
            "probability the category's true share is higher than expected. "
            "Each row is an estimate, not a test, so no correction for looking "
            "at several categories is needed.",
            prefix="Note: ",
        )
        return _Report("\n".join(lines))

    return shares


def _shares_plot(names, shares0, marginals):
    """Each category's share interval, with its expected share marked."""

    def draw(ax):
        for row, name in enumerate(names):
            post = marginals[name]
            lo, hi = post.ppf([0.025, 0.975]) * 100
            inner = post.ppf([0.25, 0.75]) * 100
            ax.plot([lo, hi], [row, row], color="#4a4e69", lw=1.6, zorder=2)
            ax.plot(inner, [row, row], color="#22223b", lw=5, zorder=3)
            ax.plot(
                post.median() * 100,
                row,
                "o",
                color="white",
                markeredgecolor="#22223b",
                markersize=7,
                zorder=4,
            )
            ax.plot(
                shares0[row] * 100,
                row,
                "|",
                color="#c9184a",
                markersize=16,
                mew=2,
                zorder=5,
            )
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("share of observations (%)")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_title(
            "each category's share, 50% and 95% intervals; red marks the "
            "expected share",
            fontsize=10,
        )
        return ax

    return draw
