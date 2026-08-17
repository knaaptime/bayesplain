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

from . import frequentist, priors
from ._config import get_draws, make_rng
from .core import beta_binomial, dirichlet_multinomial
from .result import Result

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
        Call ``.summary()`` for the full report.

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
    n_draws: int | None = None,
    seed="unset",
) -> Result:
    """Compare a rate between two groups.

    Each group gets its own exact Beta posterior. The quantity of interest --
    the difference, the risk ratio, or the odds ratio -- is then obtained by
    drawing from both and combining draw by draw.

    Those draws are independent samples from distributions known in closed
    form, the same operation as rolling dice. They are not MCMC: there is no
    chain, no burn-in, nothing to check for convergence. The distinction is
    worth making explicitly, because "simulation" and "MCMC" get used
    interchangeably and are not the same idea.

    Parameters
    ----------
    successes : array_like
        Two success counts, ``[x1, x2]``.
    n : array_like
        Two trial counts, ``[n1, n2]``.
    prior : str, tuple, or BetaPrior, default 'uninformed'
        Prior applied to each group's rate independently.
    labels : sequence of str, optional
        Group names, used in output and in the plain-English sentence.
        Defaults to ``('group 1', 'group 2')``.
    estimand : {'difference', 'risk_ratio', 'odds_ratio'}, default 'difference'
        Which comparison to report. Always oriented as group 2 relative to
        group 1.
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
    if successes.shape != (2,) or n.shape != (2,):
        raise ValueError(
            f"compare_proportions needs exactly two groups, got "
            f"{successes.shape[0] if successes.ndim else 0} success counts and "
            f"{n.shape[0] if n.ndim else 0} trial counts. For three or more "
            "groups use compare_groups(); for a table of categories use "
            "contingency()."
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
    twin = frequentist.two_proportions([x1, x2], [n1, n2])

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
