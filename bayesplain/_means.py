"""Estimating an average, and comparing averages between two groups.

The frequentist counterparts are the one-sample t-test (week 5) and the
two-sample t-test with its confidence interval, which is where the
confidence-interval misreading gets its lecture.

Something specific to these two analyses is worth flagging in class. Under the
standard reference prior the posterior is fixed by the data alone -- the
``prior=`` argument here affects **only the Bayes factor**. So a student can
vary the prior across its entire documented range, watch the credible interval
not move by a single digit, and watch the Bayes factor move a lot. The two
numbers are answering different questions, and this is the cleanest place in
the course to see that without argument.
"""

from __future__ import annotations

from . import frequentist, priors
from ._config import get_draws, make_rng
from .core import normal_t
from .result import Result

__all__ = ["mean", "compare_means"]


def mean(
    x,
    prior="conventional",
    reference: float = 0.0,
    label: str = "",
    unit: str = "",
    n_draws: int | None = None,
    seed="unset",
) -> Result:
    """Estimate one unknown average from a sample.

    The posterior is exact: with the reference prior, the mean of a normal
    sample has a scaled Student-t posterior with ``n - 1`` degrees of freedom.
    No sampling, nothing to converge.

    Parameters
    ----------
    x : array_like
        Observations. Non-finite values are dropped.
    prior : str, float, or EffectSizePrior, default 'conventional'
        Cauchy prior on standardised effect size. Affects the Bayes factor
        only; the posterior and credible interval do not depend on it.
    reference : float, default 0.0
        The value to compare against: the null for the t-test, the threshold
        for the reported probability, and the point null for the Bayes factor.
    label : str, optional
        What the quantity is, for output, e.g. ``"commute time"``.
    unit : str, optional
        Display unit, e.g. ``"minutes"``.
    n_draws : int, optional
        Number of draws for derived quantities. Defaults to the package
        setting.
    seed : int or None, optional
        Seed for those draws.

    Returns
    -------
    Result
        Call ``.summary()`` for the full report.

    Examples
    --------
    >>> import numpy as np, bayesplain as bp
    >>> commutes = np.array([28, 35, 42, 31, 25, 38, 45, 33, 29, 40.0])
    >>> res = bp.mean(commutes, reference=30, label="commute time", unit="minutes")
    >>> round(res.point("mean"), 2)
    34.6
    >>> res.exact
    True
    """
    n, sample_mean, sd = normal_t.summarise(x, "x")
    resolved = priors.resolve_effect_size(prior)
    n_draws = get_draws() if n_draws is None else int(n_draws)
    rng = make_rng(seed)

    post = normal_t.mean_posterior(n, sample_mean, sd)
    draws = post.rvs(size=n_draws, random_state=rng)

    twin = frequentist.one_mean(x, mu0=reference)
    log_bf10 = normal_t.log_bayes_factor_ttest(
        twin.statistic, n_effective=n, df=n - 1, scale=resolved.scale
    )

    subject = label or "average"
    notes = []
    if n < 15:
        notes.append(
            f"only {n} observations, so this leans on the assumption that the "
            "values are roughly normally distributed — check a histogram "
            "before trusting the tails"
        )

    def _refit(spec):
        return mean(
            x,
            prior=spec,
            reference=reference,
            label=label,
            unit=unit,
            n_draws=n_draws,
            seed=seed,
        )

    return Result(
        quantity=f"average {label}".strip() + f" (from {n} observations)",
        draws=draws,
        posterior=post,
        prior=resolved,
        subject=subject,
        analysis="mean",
        frequentist=twin,
        log_bf10=log_bf10,
        bf_alternative="a real difference from the reference",
        bf_caveat=(
            f"Compares 'the average differs from {reference:g}' against 'it is "
            f"exactly {reference:g}'. Unlike the interval above, this number "
            "does move with the prior — run .sensitivity() to see by how much."
        ),
        unit=unit,
        display_scale=1.0,
        decimals=2,
        direction_reference=reference,
        components={subject: post},
        component_axis=f"{subject} ({unit})" if unit else subject,
        refit=_refit,
        prior_ladder=priors.EFFECT_SENSITIVITY_LADDER,
        n_draws=n_draws,
        notes=notes,
    )


def compare_means(
    x,
    y,
    prior="conventional",
    labels=None,
    equal_var: bool = False,
    unit: str = "",
    n_draws: int | None = None,
    seed="unset",
) -> Result:
    """Compare the average of two groups.

    With ``equal_var=False`` (the default) this is the Behrens-Fisher problem:
    each group's mean has an exact Student-t posterior, and their difference is
    obtained by drawing from both and subtracting. Welch's test already makes
    the same unequal-variance assumption on the frequentist side, so the two
    halves of the output are answering questions about the same model.

    With ``equal_var=True`` the difference itself has a closed-form Student-t
    posterior and no sampling is needed at all.

    Parameters
    ----------
    x, y : array_like
        The two samples. The difference is oriented as ``y - x``.
    prior : str, float, or EffectSizePrior, default 'conventional'
        Cauchy prior on standardised effect size. Affects the Bayes factor
        only.
    labels : sequence of str, optional
        Group names. Defaults to ``('group 1', 'group 2')``.
    equal_var : bool, default False
        Pool the variances. Off by default, because assuming two groups have
        identical spread is an assumption people make out of habit rather than
        belief.
    unit : str, optional
        Display unit for the difference.
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
    >>> import numpy as np, bayesplain as bp
    >>> rng = np.random.default_rng(0)
    >>> a, b = rng.normal(30, 8, 60), rng.normal(34, 11, 55)
    >>> res = bp.compare_means(a, b, labels=["Route A", "Route B"], unit="minutes")
    >>> res.probability(">", 0) > 0.9
    True
    """
    n1, mean1, sd1 = normal_t.summarise(x, "x")
    n2, mean2, sd2 = normal_t.summarise(y, "y")
    resolved = priors.resolve_effect_size(prior)
    n_draws = get_draws() if n_draws is None else int(n_draws)
    rng = make_rng(seed)

    if labels is None:
        labels = ("group 1", "group 2")
    labels = tuple(str(item) for item in labels)
    if len(labels) != 2:
        raise ValueError(f"labels needs exactly two names, got {len(labels)}.")

    post1 = normal_t.mean_posterior(n1, mean1, sd1)
    post2 = normal_t.mean_posterior(n2, mean2, sd2)

    if equal_var:
        posterior = normal_t.pooled_difference_posterior(n1, mean1, sd1, n2, mean2, sd2)
        draws = posterior.rvs(size=n_draws, random_state=rng)
    else:
        posterior = None  # difference of two Student-t has no closed form
        draws = normal_t.difference_draws(
            n1, mean1, sd1, n2, mean2, sd2, size=n_draws, rng=rng
        )

    twin = frequentist.two_means(x, y, equal_var=equal_var)
    n_effective = n1 * n2 / (n1 + n2)
    log_bf10 = normal_t.log_bayes_factor_ttest(
        twin.statistic,
        n_effective=n_effective,
        df=n1 + n2 - 2,
        scale=resolved.scale,
    )

    notes = []
    if min(n1, n2) < 15:
        notes.append(
            f"the smaller group has {min(n1, n2)} observations, so this leans "
            "on an assumption of roughly normal data — check a histogram"
        )
    spread_ratio = max(sd1, sd2) / min(sd1, sd2)
    if equal_var and spread_ratio > 2:
        notes.append(
            f"the two groups' spreads differ by a factor of {spread_ratio:.1f}, "
            "which makes equal_var=True hard to justify here — rerun with the "
            "default equal_var=False"
        )

    def _refit(spec):
        return compare_means(
            x,
            y,
            prior=spec,
            labels=labels,
            equal_var=equal_var,
            unit=unit,
            n_draws=n_draws,
            seed=seed,
        )

    return Result(
        quantity=f"difference in average ({labels[1]} − {labels[0]})",
        draws=draws,
        posterior=posterior,
        prior=resolved,
        subject="difference in averages",
        analysis="compare_means",
        frequentist=twin,
        log_bf10=log_bf10,
        bf_alternative="a difference",
        bf_caveat=(
            "Compares 'the two averages differ' against 'they are identical', "
            "with a Cauchy prior on standardised effect size. Note that the "
            "credible interval above does not depend on this prior at all, "
            "while this number does — run .sensitivity()."
        ),
        unit=unit,
        display_scale=1.0,
        decimals=2,
        direction_reference=0.0,
        higher_label=labels[1],
        lower_label=labels[0],
        components={
            f"{labels[0]} (n={n1})": post1,
            f"{labels[1]} (n={n2})": post2,
        },
        component_axis=f"average for each group ({unit})" if unit else "group average",
        refit=_refit,
        prior_ladder=priors.EFFECT_SENSITIVITY_LADDER,
        n_draws=n_draws,
        notes=notes,
    )
