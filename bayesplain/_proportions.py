"""Estimating rates, and comparing rates between two groups.

These two functions cover a large share of what an introductory course
actually needs, and they are the mathematically easiest part of the package:
everything here is closed form, and the only sampling is drawing from two
exactly-known distributions in order to subtract them.

The frequentist counterparts are the sample proportion with its standard error
(week 2) and the two-proportion z / chi-square test (week 4).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from . import frequentist, priors
from ._config import get_draws, make_rng
from .core import beta_binomial, dirichlet_multinomial, hierarchical
from .result import Prediction, Result, _Report, _wrap

__all__ = ["proportion", "compare_proportions"]

_ESTIMANDS = {
    "difference": (0.0, 100.0, "percentage points", 1),
    "risk_ratio": (1.0, 1.0, "times", 2),
    "odds_ratio": (1.0, 1.0, "times", 2),
}


# ---------------------------------------------------------------------------
# One rate
# ---------------------------------------------------------------------------


def proportion(
    successes: float,
    n: float,
    prior="uninformed",
    reference: float = 0.5,
    n_draws: int | None = None,
    seed="unset",
) -> Result:
    """Estimate one unknown rate from counts.

    The posterior is exact. With a ``Beta(a, b)`` prior and ``successes`` out of
    ``n``, it is ``Beta(a + successes, b + n - successes)`` -- conjugacy means
    observing data just adds counts to the two shape parameters, so there is
    nothing to approximate and nothing to converge.

    Parameters
    ----------
    successes : int
        Number of cases with the attribute of interest.
    n : int
        Number of cases examined.
    prior : str, tuple, or BetaPrior, default 'uninformed'
        A preset name, an ``(a, b)`` pair, or the result of
        ``bayesplain.priors.from_previous_study(...)``. See
        :func:`bayesplain.priors.describe`.
    reference : float, default 0.5
        The rate to compare against: the null value for the frequentist test,
        the threshold for the reported direction probability, and the point
        null for the Bayes factor. Set this to a policy-relevant number rather
        than leaving it at 0.5 whenever one exists.
    n_draws : int, optional
        Draws to take for derived quantities. Defaults to the package setting.
    seed : int or None, optional
        Seed for those draws. Defaults to the package seed, so that a whole
        class sees identical digits.

    Returns
    -------
    Result
        Call ``.summary()`` for the full report, and ``.predict(n=...)`` for a
        forecast of the successes among the next ``n`` cases.

    Examples
    --------
    >>> import bayesplain as bf
    >>> res = bf.proportion(successes=34, n=220, reference=0.10)
    >>> round(res.point(), 4)
    0.1566
    >>> round(res.probability(">", 0.10), 3)
    0.996

    Notes
    -----
    The 95% equal-tailed credible interval here will land very close to a
    Wilson score confidence interval on the same data. That is worth showing
    students explicitly: the two frameworks often produce nearly the same
    *numbers* while licensing very different *sentences*, so the choice between
    them is rarely about arithmetic.
    """
    successes, n = beta_binomial.validate_counts(successes, n)
    resolved = priors.resolve_proportion(prior)
    n_draws = get_draws() if n_draws is None else int(n_draws)
    rng = make_rng(seed)

    post = beta_binomial.posterior(successes, n, resolved.a, resolved.b)
    draws = post.rvs(size=n_draws, random_state=rng)

    if not 0.0 < reference < 1.0:
        raise ValueError(
            f"reference must be strictly between 0 and 1, got {reference}. It "
            "is a rate to compare against, e.g. 0.10 for 10%."
        )

    log_bf10 = beta_binomial.log_bayes_factor_point_null(
        successes, n, p0=reference, a=resolved.a, b=resolved.b
    )
    twin = frequentist.one_proportion(successes, n, p0=reference)

    total = n
    notes = []
    if n < 30:
        notes.append(
            f"only {n} observations, so the prior is doing visible work here — "
            "run .sensitivity()"
        )
    if successes in (0, n):
        notes.append(
            f"all {n} observations fell on one side; the posterior still gives "
            "a usable interval where a Wald confidence interval would collapse "
            "to zero width"
        )

    def _predictor(n: int = 100) -> Prediction:
        n_new = int(n)
        if n_new < 1 or n_new != n:
            raise ValueError(f"n must be a positive whole number of cases, got {n}.")
        forecast = stats.betabinom(
            n_new, resolved.a + successes, resolved.b + total - successes
        )
        return Prediction(
            what=f"number of successes in the next {n_new:,} cases",
            dist=forecast,
            plug_in=stats.binom(n_new, successes / total),
            discrete=True,
        )

    def _refit(spec):
        return proportion(
            successes,
            n,
            prior=spec,
            reference=reference,
            n_draws=n_draws,
            seed=seed,
        )

    return Result(
        quantity=f"rate ({successes} of {n})",
        draws=draws,
        posterior=post,
        prior=resolved,
        subject="rate",
        analysis="proportion",
        frequentist=twin,
        log_bf10=log_bf10,
        bf_caveat=(
            f"Compares 'the rate is unknown' against 'the rate is exactly "
            f"{reference:.4g}'. A point null like that is rarely what anyone "
            f"believes, and the number moves with the prior. Run .sensitivity()."
        ),
        unit="%",
        display_scale=100.0,
        decimals=1,
        direction_reference=reference,
        components={f"rate ({successes}/{n})": post},
        component_axis="rate (%)",
        component_scale=100.0,
        predictor=_predictor,
        refit=_refit,
        prior_ladder=priors.SENSITIVITY_LADDER,
        n_draws=n_draws,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Two rates
# ---------------------------------------------------------------------------


def compare_proportions(
    successes,
    n,
    prior="uninformed",
    labels=None,
    estimand: str = "difference",
    pool: bool = False,
    threshold: float = 0.0,
    n_draws: int | None = None,
    seed="unset",
) -> Result:
    """Compare a rate between two groups, or across several.

    Each group gets its own exact Beta posterior. The quantity of interest --
    the difference, the risk ratio, or the odds ratio -- is then obtained by
    drawing from both and combining draw by draw.

    Those draws are independent samples from distributions known in closed
    form, the same operation as rolling dice. They are not MCMC: there is no
    chain, no burn-in, nothing to check for convergence. The distinction is
    worth making explicitly, because "simulation" and "MCMC" get used
    interchangeably and are not the same idea.

    With three or more groups the headline becomes the spread between the
    highest and lowest rate, and the analysis is really about ``.pairwise()``
    -- the same shape as :func:`bayesplain.compare_groups`, for rates.

    Parameters
    ----------
    successes : array_like
        Success counts, one per group: ``[x1, x2]``, or more.
    n : array_like
        Trial counts, one per group.
    prior : str, tuple, or BetaPrior, default 'uninformed'
        Prior applied to each group's rate independently.
    labels : sequence of str, optional
        Group names, used in output and in the plain-English sentence.
        Defaults to ``('group 1', 'group 2', ...)``.
    estimand : {'difference', 'risk_ratio', 'odds_ratio'}, default 'difference'
        Which comparison to report, for two groups. Always oriented as group 2
        relative to group 1.
    pool : bool, default False
        With three or more groups, partially pool the rates: estimate how
        much groups like these typically differ, and pull each rate toward the
        typical one in proportion to how little data it rests on. Stops the
        smallest district from topping the ranking on noise alone.
    threshold : float, default 0.0
        With three or more groups, the spread between the highest and lowest
        rate that the reported probability is measured against, as a
        proportion (0.05 for five points).
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
    Two districts' eviction filing rates -- the case where the two frameworks
    give the same numbers and very different advice:

    >>> import bayesplain as bf
    >>> res = bf.compare_proportions(
    ...     successes=[34, 51],
    ...     n=[220, 240],
    ...     labels=["District A", "District B"],
    ... )
    >>> round(res.probability(">", 0), 2)
    0.94
    >>> res.frequentist.pvalue > 0.05
    True
    """
    successes = np.asarray(successes)
    n = np.asarray(n)
    if successes.ndim != 1 or successes.shape != n.shape or successes.shape[0] < 2:
        raise ValueError(
            f"compare_proportions needs one success count and one trial count "
            f"per group, for two or more groups; got {successes.size} success "
            f"counts and {n.size} trial counts. For a table of categories use "
            "contingency()."
        )
    if successes.shape[0] > 2:
        if estimand != "difference":
            raise ValueError(
                "with three or more groups every comparison is a difference in "
                "rates; estimand= applies to two groups only."
            )
        return _compare_several(
            successes, n, prior, labels, pool, threshold, n_draws, seed
        )
    if pool:
        raise ValueError(
            "pool=True needs at least three groups: with two, the spread "
            "between groups would be estimated from a single difference."
        )
    pairs = [beta_binomial.validate_counts(s, t) for s, t in zip(successes, n)]
    (x1, n1), (x2, n2) = pairs

    if estimand not in _ESTIMANDS:
        raise ValueError(
            f"estimand must be one of {sorted(_ESTIMANDS)}, got {estimand!r}."
        )
    reference, scale, unit, decimals = _ESTIMANDS[estimand]

    if labels is None:
        labels = ("group 1", "group 2")
    labels = tuple(str(item) for item in labels)
    if len(labels) != 2:
        raise ValueError(f"labels needs exactly two names, got {len(labels)}.")

    resolved = priors.resolve_proportion(prior)
    n_draws = get_draws() if n_draws is None else int(n_draws)
    rng = make_rng(seed)

    post1 = beta_binomial.posterior(x1, n1, resolved.a, resolved.b)
    post2 = beta_binomial.posterior(x2, n2, resolved.a, resolved.b)
    draws1 = post1.rvs(size=n_draws, random_state=rng)
    draws2 = post2.rvs(size=n_draws, random_state=rng)

    if estimand == "difference":
        draws = draws2 - draws1
        quantity = f"difference in rate ({labels[1]} − {labels[0]})"
    elif estimand == "risk_ratio":
        draws = draws2 / draws1
        quantity = f"risk ratio ({labels[1]} ÷ {labels[0]})"
    else:
        odds1 = draws1 / (1.0 - draws1)
        odds2 = draws2 / (1.0 - draws2)
        draws = odds2 / odds1
        quantity = f"odds ratio ({labels[1]} ÷ {labels[0]})"

    # Columns are [successes, failures], so a [a, b] concentration on the
    # table is exactly the Beta(a, b) prior used for each group's rate above.
    # Tying them means .sensitivity() moves the interval and the Bayes factor
    # with the same assumption rather than varying one and holding the other.
    table = np.array([[x1, n1 - x1], [x2, n2 - x2]], dtype=float)
    log_bf10 = dirichlet_multinomial.log_bayes_factor_independence(
        table, concentration=[resolved.a, resolved.b]
    )
    twin = frequentist.two_proportions([x1, x2], [n1, n2], estimand=estimand)

    notes = []
    if min(x1, x2, n1 - x1, n2 - x2) < 5:
        notes.append(
            "at least one cell holds fewer than 5 cases, where the chi-square "
            "approximation is unreliable but the exact posterior is not"
        )

    def _refit(spec):
        return compare_proportions(
            [x1, x2],
            [n1, n2],
            prior=spec,
            labels=labels,
            estimand=estimand,
            n_draws=n_draws,
            seed=seed,
        )

    return Result(
        quantity=quantity,
        draws=draws,
        posterior=None,  # difference of two Betas has no tidy closed form
        prior=resolved,
        analysis="compare_proportions",
        frequentist=twin,
        log_bf10=log_bf10,
        bf_caveat=(
            "Compares 'the two groups have different rates' against 'they have "
            "the same rate', under a flat Dirichlet prior over the 2x2 table. "
            "Prior-sensitive; run .sensitivity() before quoting it."
        ),
        unit=unit,
        display_scale=scale,
        decimals=decimals,
        direction_reference=reference,
        higher_label=labels[1],
        lower_label=labels[0],
        components={
            f"{labels[0]} ({x1}/{n1})": post1,
            f"{labels[1]} ({x2}/{n2})": post2,
        },
        # The components are each group's rate, always a percentage, whether
        # the headline estimand is their difference or their ratio.
        component_axis="rate for each group (%)",
        component_scale=100.0,
        refit=_refit,
        prior_ladder=priors.SENSITIVITY_LADDER,
        n_draws=n_draws,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Three or more rates
# ---------------------------------------------------------------------------


def _compare_several(successes, n, prior, labels, pool, threshold, n_draws, seed):
    """Compare three or more rates, optionally with partial pooling."""
    from ._groups import _forest_plot, _pairwise_plot

    counts = [beta_binomial.validate_counts(s, t) for s, t in zip(successes, n)]
    x = np.array([c[0] for c in counts], dtype=float)
    t = np.array([c[1] for c in counts], dtype=float)
    k = x.size
    if labels is None:
        labels = [f"group {i + 1}" for i in range(k)]
    names = [str(item) for item in labels]
    if len(names) != k:
        raise ValueError(f"labels has {len(names)} names but there are {k} groups.")
    if len(set(names)) != k:
        raise ValueError("group labels must be distinct.")

    resolved = priors.resolve_proportion(prior)
    n_draws = get_draws() if n_draws is None else int(n_draws)
    rng = make_rng(seed)

    if pool:
        pooling = hierarchical.shrink_proportions(x, t)
        alpha, beta = pooling["alpha"], pooling["beta"]
    else:
        pooling = None
        alpha, beta = resolved.a + x, resolved.b + t - x
    posteriors = {name: stats.beta(alpha[i], beta[i]) for i, name in enumerate(names)}
    group_draws = {
        name: post.rvs(size=n_draws, random_state=rng)
        for name, post in posteriors.items()
    }
    stacked = np.column_stack([group_draws[name] for name in names])
    spread = stacked.max(axis=1) - stacked.min(axis=1)
    twin = frequentist.several_proportions(x, t)

    notes = [
        f"The spread reported above cannot be negative, so P(above "
        f"{threshold * 100:g} points) is not a test of anything. The "
        "comparison table from .pairwise() is what this analysis is for."
    ]
    smallest = int(t.min())
    if smallest < 30 and not pool:
        notes.append(
            f"the smallest group has {smallest} cases; its rate will look more "
            "extreme than it is, which is exactly what pool=True corrects"
        )
    if pooling is not None:
        if pooling["full"]:
            notes.append(
                "partial pooling is on, and the groups are indistinguishable "
                "from one shared rate: every group has been pulled all the way "
                f"to {pooling['grand_mean']:.1%}. The data cannot tell these "
                "groups apart"
            )
        else:
            pulled = float(1.0 - pooling["weights"].min())
            notes.append(
                f"partial pooling is on: the group resting on the least data "
                f"was pulled {pulled:.0%} of the way toward the typical rate of "
                f"{pooling['grand_mean']:.1%}. The pull is estimated from these "
                "same groups (empirical Bayes), which understates uncertainty "
                "a little when there are only a handful of them"
            )

    def _refit(spec):
        return _compare_several(x, t, spec, names, False, threshold, n_draws, seed)

    result = Result(
        quantity="spread between the highest and lowest rate",
        draws=spread,
        posterior=None,
        prior=None if pool else resolved,
        subject="spread between the highest and lowest rate",
        analysis="compare_proportions",
        frequentist=twin,
        log_bf10=None,
        unit="percentage points",
        display_scale=100.0,
        decimals=1,
        direction_reference=float(threshold),
        components={
            f"{name} ({int(x[i])}/{int(t[i])})": posteriors[name]
            for i, name in enumerate(names)
        },
        component_axis="rate for each group (%)",
        component_scale=100.0,
        refit=None if pool else _refit,
        prior_ladder=None if pool else priors.SENSITIVITY_LADDER,
        n_draws=n_draws,
        notes=notes,
        no_sensitivity_reason=(
            "with pool=True the prior on each rate is estimated from the "
            "groups themselves, so there is no fixed prior to vary. The choice "
            "that changes the answer is pool=True versus pool=False; run both "
            "and compare the rankings."
        ),
        no_bayes_factor_reason=(
            "no omnibus Bayes factor is computed for three or more rates, for "
            "the same reason compare_groups() computes none: 'is there a "
            "difference somewhere?' is rarely the question, and grading it "
            "needs a prior over every pattern of differences at once. "
            ".pairwise() reports a Bayes factor for each pair."
        ),
        no_predict_reason=(
            "compare_proportions has no single next observation to forecast. "
            "Run proportion() on the group you care about and call .predict()."
        ),
        next_steps=(
            ".pairwise()  .plot(kind='forest')  compare_proportions(..., pool=True)"
        ),
    )
    result.group_names = names
    result.group_draws = group_draws
    result.group_scale = 100.0
    result.pooling = pooling
    result.pairwise = _make_rate_pairwise(result, x, t, resolved)
    result.custom_plots = {
        "forest": _forest_plot(result),
        "pairwise": _pairwise_plot(result),
    }
    return result


def _make_rate_pairwise(result, x, t, resolved):
    """Build the ``.pairwise()`` method for a several-rate comparison."""
    index = {name: i for i, name in enumerate(result.group_names)}

    def pairwise(level: float = 0.95, only=None, rope=None) -> _Report:
        """Compare every pair of groups' rates, or only the pairs that matter.

        Parameters
        ----------
        level : float, default 0.95
            Credible level for each interval.
        only : sequence, optional
            Restrict to specific pairs, as ``[(name_a, name_b), ...]``.
        rope : tuple of float, optional
            A region of practical equivalence on the difference in rates, as
            proportions: ``(-0.02, 0.02)`` for two points either way.

        Returns
        -------
        _Report
            For each pair: the difference in percentage points, its credible
            interval, the probability the first rate is higher, and a
            Gunel-Dickey Bayes factor for that pair's 2x2 table.
        """
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
                if name not in index:
                    raise ValueError(
                        f"unknown group {name!r}; available: {', '.join(names)}."
                    )
        tail = (1.0 - level) / 2.0
        pct = f"{level:.0%}"
        lines = [
            "PAIRWISE COMPARISONS"
            + (" (partially pooled)" if result.pooling is not None else ""),
            "",
            f"{'comparison':<30}{'diff (pts)':>11}{pct + ' interval':>22}"
            f"{'P(1st>2nd)':>12}{'BF10':>10}",
            "-" * 85,
        ]
        verdicts = []
        for a, b in pairs:
            diff = result.group_draws[a] - result.group_draws[b]
            lo, hi = np.quantile(diff, [tail, 1.0 - tail])
            prob = float((diff > 0).mean())
            i, j = index[a], index[b]
            table = np.array([[x[i], t[i] - x[i]], [x[j], t[j] - x[j]]])
            with np.errstate(over="ignore"):
                bf = float(
                    np.exp(
                        dirichlet_multinomial.log_bayes_factor_independence(
                            table, concentration=[resolved.a, resolved.b]
                        )
                    )
                )
            span = f"{lo * 100:.1f} to {hi * 100:.1f}".replace("-", "−")
            middle = f"{np.median(diff) * 100:.1f}".replace("-", "−")
            lines.append(
                f"{(a + ' − ' + b)[:29]:<30}"
                f"{middle:>11}{span:>22}{prob:>12.3f}{bf:>10.3g}"
            )
            if rope is not None:
                low, high = sorted(float(v) for v in rope)
                if lo >= low and hi <= high:
                    verdicts.append((a, b, "practically equivalent"))
                elif hi < low or lo > high:
                    verdicts.append((a, b, "practically different"))
                else:
                    verdicts.append((a, b, "too uncertain to call"))
        if rope is not None:
            low, high = sorted(float(v) for v in rope)
            lines += [
                "",
                f"AGAINST A ROPE OF {low * 100:g} TO {high * 100:g} POINTS",
                "-" * 85,
            ]
            lines += [f"{a + ' − ' + b:<30}{verdict}" for a, b, verdict in verdicts]
        lines += [""]
        bf_note = (
            " Bayes factors use each pair's own counts under the stated prior, "
            "not the pooled estimates."
            if result.pooling is not None
            else ""
        )
        lines += _wrap(
            "Every row is an estimate, not a test, so no multiple-comparisons "
            "correction is applied or needed. If a small group looks extreme by "
            "chance, the fix is pool=True, not a correction." + bf_note,
            prefix="Note: ",
        )
        return _Report("\n".join(lines))

    return pairwise
