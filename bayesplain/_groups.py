"""Comparing many groups, without an omnibus test.

The frequentist counterpart is one-way ANOVA (week 9), and this is where the
package diverges from convention most deliberately.

ANOVA answers "is there a difference somewhere among these groups?" That is
almost never the question a planner has. "Is district A different from district
C, and by how much?" is a question you can act on; "there is a difference
somewhere among the six" is not. So the default output here is every pairwise
comparison with a credible interval, not an omnibus statistic.

This also sidesteps having to explain multiple-comparisons correction to a
non-mathematical audience, which is genuinely hard to do honestly. The reason
it can be sidestepped is worth saying out loud: estimating twenty quantities is
not the same activity as running twenty tests. A test is a decision procedure
whose error rate compounds with repetition; an interval is a description of
what the data support, and describing six districts does not make the
description of any one of them less accurate. What *does* protect against
reading noise as signal is partial pooling, which is available here as
``pool=True`` and is a better answer than a Tukey correction.

There is deliberately no omnibus Bayes factor. See :meth:`Result.bayes_factor`
on a result from this function for the reasoning, and use ``.pairwise()``,
which reports a validated JZS Bayes factor for each pair that matters.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from . import frequentist
from ._config import get_draws, make_rng
from .core import hierarchical, normal_t
from .result import Result, _Report, _wrap

__all__ = ["compare_groups"]


def compare_groups(
    data,
    by=None,
    labels=None,
    pool: bool = False,
    threshold: float = 0.0,
    unit: str = "",
    n_draws: int | None = None,
    seed="unset",
) -> Result:
    """Compare the averages of three or more groups.

    Parameters
    ----------
    data : mapping, sequence of array_like, or array_like
        The observations. Accepts a ``{name: values}`` mapping, a sequence of
        arrays (one per group), or a flat array of values paired with ``by``.
        A pandas Series or DataFrame column works with ``by`` as well.
    by : array_like, optional
        Group label for each observation, when ``data`` is a flat array.
    labels : sequence of str, optional
        Group names, when they cannot be read from ``data``.
    pool : bool, default False
        Partially pool the group estimates, pulling each toward the overall
        average in proportion to how noisy it is. Small groups get pulled the
        furthest, which stops the smallest site from winning the ranking on
        noise alone.
    threshold : float, default 0.0
        The spread the reported probability is measured against.
    unit : str, optional
        Display unit.
    n_draws : int, optional
        Number of draws. Defaults to the package setting.
    seed : int or None, optional
        Seed for the draws.

    Returns
    -------
    Result
        The headline quantity is the spread between the highest and lowest
        group average. Call ``.pairwise()`` for the comparison table that this
        analysis is really about, and ``.plot(kind='forest')`` to see the
        groups ranked with their uncertainty.

    Examples
    --------
    >>> import numpy as np, bayesplain as bp
    >>> rng = np.random.default_rng(0)
    >>> data = {
    ...     "North": rng.normal(12, 3, 40),
    ...     "South": rng.normal(15, 3, 38),
    ...     "East": rng.normal(12.5, 3, 45),
    ... }
    >>> res = bp.compare_groups(data, unit="days")
    >>> len(res.components)
    3
    >>> print(res.pairwise())          # doctest: +SKIP
    """
    names, samples = _read_groups(data, by, labels)
    n_groups = len(names)
    if n_groups < 3:
        raise ValueError(
            f"compare_groups is for three or more groups, got {n_groups}. For "
            "exactly two, use compare_means(), which gives an exact posterior "
            "for the difference."
        )
    n_draws = get_draws() if n_draws is None else int(n_draws)
    rng = make_rng(seed)

    stats_per_group = [normal_t.summarise(s, name) for s, name in zip(samples, names)]
    sizes = np.array([item[0] for item in stats_per_group], dtype=float)
    means = np.array([item[1] for item in stats_per_group], dtype=float)
    sds = np.array([item[2] for item in stats_per_group], dtype=float)
    errors = sds / np.sqrt(sizes)

    pooling = hierarchical.shrink(means, errors) if pool else None
    group_draws = _group_draws(
        names, sizes, means, sds, pooling, n_draws=n_draws, rng=rng
    )
    stacked = np.column_stack([group_draws[name] for name in names])

    # The honest replacement for an omnibus test: how far apart are the best
    # and worst groups, with uncertainty carried through.
    spread = stacked.max(axis=1) - stacked.min(axis=1)
    twin = frequentist.one_way_anova(samples)

    notes = _build_notes(names, sizes, pooling, threshold)

    result = Result(
        quantity="spread between the highest and lowest group average",
        draws=spread,
        posterior=None,
        prior=None,
        subject="spread between the highest and lowest group",
        analysis="compare_groups",
        frequentist=twin,
        log_bf10=None,
        unit=unit,
        display_scale=1.0,
        decimals=2,
        direction_reference=float(threshold),
        components={
            f"{name} (n={int(size)})": group_draws[name]
            for name, size in zip(names, sizes)
        },
        component_axis=f"group average ({unit})" if unit else "group average",
        n_draws=n_draws,
        notes=notes,
        no_sensitivity_reason=(
            "compare_groups has no prior to vary: the group posteriors use the "
            "standard reference prior and are fixed by the data alone. The "
            "choice that does change the answer here is pool=True versus "
            "pool=False, so run it both ways and compare the rankings — that "
            "is the sensitivity analysis this method calls for."
        ),
        no_bayes_factor_reason=(
            "compare_groups deliberately computes no omnibus Bayes factor. "
            "The omnibus question — 'is there a difference somewhere among "
            "these groups?' — is one a planner almost never needs answered, "
            "and grading the evidence for it would mean choosing a prior over "
            "every pattern of group differences at once. Use .pairwise(), "
            "which reports a validated Bayes factor for each pair you actually "
            "care about."
        ),
        next_steps=(
            ".pairwise()  .plot(kind='forest')  compare_groups(..., pool=True)"
        ),
    )

    # Attach the group-level machinery the pairwise views need.
    result.group_names = names
    result.group_draws = group_draws
    result.group_sizes = sizes
    result.group_means = means
    result.group_sds = sds
    result.pooling = pooling
    result.pairwise = _make_pairwise(result)
    result.custom_plots = {
        "forest": _forest_plot(result),
        "pairwise": _pairwise_plot(result),
    }
    return result


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


def _read_groups(data, by, labels):
    """Normalise the several accepted input shapes into names and arrays."""
    if by is not None:
        values = np.asarray(getattr(data, "to_numpy", lambda: data)()).ravel()
        keys = np.asarray(getattr(by, "to_numpy", lambda: by)()).ravel()
        if values.size != keys.size:
            raise ValueError(
                f"data has {values.size} values but by has {keys.size} labels; "
                "they must line up one to one."
            )
        names = [str(k) for k in dict.fromkeys(keys.tolist())]
        samples = [values[keys == name] for name in dict.fromkeys(keys.tolist())]
        return names, samples

    if hasattr(data, "items") and not hasattr(data, "ndim"):
        names = [str(k) for k in data]
        return names, [np.asarray(v, dtype=float).ravel() for v in data.values()]

    samples = [np.asarray(group, dtype=float).ravel() for group in data]
    if labels is None:
        names = [f"group {i + 1}" for i in range(len(samples))]
    else:
        names = [str(item) for item in labels]
        if len(names) != len(samples):
            raise ValueError(
                f"labels has {len(names)} names but data has {len(samples)} groups."
            )
    return names, samples


def _group_draws(names, sizes, means, sds, pooling, n_draws, rng):
    """Posterior draws for each group's average, pooled or not."""
    draws = {}
    for i, name in enumerate(names):
        if pooling is None:
            posterior = normal_t.mean_posterior(
                int(sizes[i]), float(means[i]), float(sds[i])
            )
        else:
            # Conditional on the estimated between-group spread, each shrunk
            # mean is normal. Treating tau-squared as known is the
            # empirical-Bayes shortcut this buys its closed form with.
            posterior = stats.norm(
                loc=pooling["means"][i], scale=pooling["standard_errors"][i]
            )
        draws[name] = posterior.rvs(size=n_draws, random_state=rng)
    return draws


def _build_notes(names, sizes, pooling, threshold):
    """Caveats specific to this comparison."""
    notes = [
        f"The spread reported above cannot be negative, so P(above "
        f"{threshold:g}) is not a test of anything. The comparison table from "
        ".pairwise() is what this analysis is for."
    ]
    smallest = int(sizes.min())
    if smallest < 15:
        notes.append(
            f"the smallest group has {smallest} observations; it will look "
            "more extreme than it is, which is exactly what pool=True corrects"
        )
    if pooling is not None:
        pulled = float(1.0 - pooling["weights"].min())
        notes.append(
            f"partial pooling is on: the least precise group was pulled "
            f"{pulled:.0%} of the way toward the overall average of "
            f"{pooling['grand_mean']:.2f}"
        )
        if pooling["tau2"] == 0.0:
            notes.append(
                "the estimated spread between groups is zero, so pooling has "
                "collapsed every group to the overall average — the data "
                "cannot distinguish these groups at all"
            )
    return notes


# ---------------------------------------------------------------------------
# The pairwise view, which is the point of this analysis
# ---------------------------------------------------------------------------


def _make_pairwise(result):
    """Build the ``.pairwise()`` method bound to this result."""

    def pairwise(level: float = 0.95, only=None, rope=None) -> _Report:
        """Compare every pair of groups, or only the pairs that matter.

        Parameters
        ----------
        level : float, default 0.95
            Credible level for each interval.
        only : sequence, optional
            Restrict to specific pairs, as ``[(name_a, name_b), ...]``. Run
            only the comparisons your decision actually turns on -- there is no
            statistical penalty for restraint here, but there is a reader's
            one.
        rope : tuple of float, optional
            A region of practical equivalence applied to every pair, which
            turns each row's verdict from "different" into "different enough
            to act on".

        Returns
        -------
        _Report
            A table with, for each pair, the estimated difference, its credible
            interval, the probability that the first group is higher, and a
            JZS Bayes factor for that pair.
        """
        from .core import normal_t as nt

        names = result.group_names
        pairs = (
            [(str(a), str(b)) for a, b in only]
            if only is not None
            else [
                (names[i], names[j])
                for i in range(len(names))
                for j in range(i + 1, len(names))
            ]
        )
        for a, b in pairs:
            for name in (a, b):
                if name not in result.group_draws:
                    raise ValueError(
                        f"unknown group {name!r}; available: {', '.join(names)}."
                    )

        pct = f"{level:.0%}"
        header = (
            f"{'comparison':<30}{'difference':<12}{pct + ' interval':<22}"
            f"{'P(1st>2nd)':>11}"
        )
        lines = [
            "PAIRWISE COMPARISONS"
            + (" (partially pooled)" if result.pooling is not None else ""),
            "",
            header,
            "-" * 76,
        ]

        index = {name: i for i, name in enumerate(names)}
        for a, b in pairs:
            diff = result.group_draws[a] - result.group_draws[b]
            lo, hi = np.quantile(diff, [(1 - level) / 2, 1 - (1 - level) / 2])
            prob = float((diff > 0).mean())
            label = f"{a} − {b}"
            span = f"{lo:.2f} to {hi:.2f}"
            lines.append(f"{label:<30}{np.median(diff):<12.2f}{span:<22}{prob:>11.3f}")

        # Per-pair Bayes factors, which are the validated two-sample JZS
        # integral rather than an omnibus number this package does not compute.
        lines += ["", f"{'comparison':<30}{'BF10 (JZS)':>12}   verdict"]
        lines += ["-" * 76]
        for a, b in pairs:
            i, j = index[a], index[b]
            n1, n2 = int(result.group_sizes[i]), int(result.group_sizes[j])
            twin = _two_sample_t(
                result.group_means[i],
                result.group_sds[i],
                n1,
                result.group_means[j],
                result.group_sds[j],
                n2,
            )
            bf = np.exp(
                nt.log_bayes_factor_ttest(
                    twin, n_effective=n1 * n2 / (n1 + n2), df=n1 + n2 - 2
                )
            )
            verdict = (
                "favours a difference"
                if bf > 3
                else ("favours no difference" if bf < 1 / 3 else "inconclusive")
            )
            lines.append(f"{a + ' − ' + b:<30}{bf:>12.2f}   {verdict}")

        if rope is not None:
            low, high = sorted(float(v) for v in rope)
            lines += ["", f"AGAINST A ROPE OF {low:g} TO {high:g}", "-" * 76]
            for a, b in pairs:
                diff = result.group_draws[a] - result.group_draws[b]
                lo, hi = np.quantile(diff, [(1 - level) / 2, 1 - (1 - level) / 2])
                if lo >= low and hi <= high:
                    verdict = "practically equivalent"
                elif hi < low or lo > high:
                    verdict = "practically different"
                else:
                    verdict = "too uncertain to call"
                lines.append(f"{a + ' − ' + b:<30}{verdict}")

        lines += [""]
        lines += _wrap(
            "Every row is an estimate, not a test, so no multiple-comparisons "
            "correction is applied or needed. Describing several groups does "
            "not make the description of any one of them less accurate. If you "
            "are worried that a small group looks extreme by chance, the fix "
            "is pool=True, not a correction.",
            prefix="Note: ",
        )
        return _Report("\n".join(lines))

    return pairwise


def _two_sample_t(mean1, sd1, n1, mean2, sd2, n2) -> float:
    """Welch t statistic from summary statistics."""
    se = np.sqrt(sd1**2 / n1 + sd2**2 / n2)
    return float((mean1 - mean2) / se)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _forest_plot(result):
    """Group averages with credible intervals, ranked."""

    def draw(ax):
        names = result.group_names
        order = sorted(names, key=lambda n: np.median(result.group_draws[n]))
        for row, name in enumerate(order):
            sample = result.group_draws[name]
            lo, hi = np.quantile(sample, [0.025, 0.975])
            inner = np.quantile(sample, [0.25, 0.75])
            ax.plot([lo, hi], [row, row], color="#4a4e69", lw=1.6, zorder=2)
            ax.plot(inner, [row, row], color="#22223b", lw=5, zorder=3)
            ax.plot(
                np.median(sample),
                row,
                "o",
                color="white",
                markeredgecolor="#22223b",
                markersize=7,
                zorder=4,
            )
        if result.pooling is not None:
            ax.axvline(
                result.pooling["grand_mean"],
                color="#c9184a",
                ls="--",
                lw=1.2,
                zorder=1,
                label="overall average",
            )
            ax.legend(frameon=False, loc="lower right")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order)
        ax.set_xlabel(result.component_axis)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_title(
            "each group's average, with 50% and 95% credible intervals",
            fontsize=10,
        )
        return ax

    return draw


def _pairwise_plot(result):
    """Every pairwise difference, with zero marked."""

    def draw(ax):
        names = result.group_names
        pairs = [
            (names[i], names[j])
            for i in range(len(names))
            for j in range(i + 1, len(names))
        ]
        for row, (a, b) in enumerate(pairs):
            diff = result.group_draws[a] - result.group_draws[b]
            lo, hi = np.quantile(diff, [0.025, 0.975])
            crosses = lo < 0 < hi
            colour = "#9a8c98" if crosses else "#22223b"
            ax.plot([lo, hi], [row, row], color=colour, lw=1.6, zorder=2)
            ax.plot(
                np.median(diff),
                row,
                "o",
                color="white",
                markeredgecolor=colour,
                markersize=7,
                zorder=4,
            )
        ax.axvline(0, color="#c9184a", lw=1.2, zorder=1)
        ax.set_yticks(range(len(pairs)))
        ax.set_yticklabels([f"{a} − {b}" for a, b in pairs])
        ax.set_xlabel(f"difference ({result.unit})" if result.unit else "difference")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_title("pairs whose interval crosses zero are shown faded", fontsize=10)
        return ax

    return draw
