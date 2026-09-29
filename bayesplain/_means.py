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

Two options change what is being estimated rather than how. ``paired=True``
compares two measurements on the same units -- before and after, this year and
last -- by analysing the within-pair differences, which is the paired t-test's
model. ``log=True`` works on the log scale and reports the answer as a ratio of
typical (geometric-mean) values, which is the honest way to handle skewed data
like prices and incomes.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy import stats

from . import frequentist, priors
from ._config import get_draws, make_rng
from .core import normal_t
from .result import Prediction, Result

__all__ = ["mean", "compare_means"]


def _positive(arr: np.ndarray, name: str) -> np.ndarray:
    """Log-transform a sample, refusing values a logarithm cannot take."""
    if (arr <= 0).any():
        bad = int((arr <= 0).sum())
        raise ValueError(
            f"log=True needs every value of {name} to be positive, but {bad} "
            "are zero or negative. A log scale suits quantities like prices, "
            "incomes, and travel times; for data that can be zero, analyse the "
            "original scale instead."
        )
    return np.log(arr)


def _finite(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def _safe_log(value: float) -> float:
    return float(np.log(value)) if value > 0 else -np.inf


def _exp_twin(twin, test_suffix: str, null_statement: str):
    """Carry a log-scale t-test back to the original scale for printing."""
    return replace(
        twin,
        test=twin.test + test_suffix,
        estimate=float(np.exp(twin.estimate)),
        interval=(
            None
            if twin.interval is None
            else (float(np.exp(twin.interval[0])), float(np.exp(twin.interval[1])))
        ),
        null_statement=null_statement,
    )


def mean(
    x,
    prior="conventional",
    reference: float = 0.0,
    label: str = "",
    unit: str = "",
    log: bool = False,
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
    log : bool, default False
        Work on the log scale and report the typical (geometric-mean) value
        on the original scale. For skewed, strictly positive data -- prices,
        incomes -- where the arithmetic mean is dragged around by a few large
        values. ``reference`` stays on the original scale.
    n_draws : int, optional
        Number of draws for derived quantities. Defaults to the package
        setting.
    seed : int or None, optional
        Seed for those draws.

    Returns
    -------
    Result
        Call ``.summary()`` for the full report, and ``.predict()`` for a
        forecast of the next observation.

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
    values = _finite(x)
    if log:
        if not reference > 0:
            raise ValueError(
                f"with log=True the reference must be positive, got {reference}."
            )
        values = _positive(values, "x")
    n, sample_mean, sd = normal_t.summarise(values, "x")
    resolved = priors.resolve_effect_size(prior)
    n_draws = get_draws() if n_draws is None else int(n_draws)
    rng = make_rng(seed)

    post = normal_t.mean_posterior(n, sample_mean, sd)
    draws = post.rvs(size=n_draws, random_state=rng)

    if log:
        twin = _exp_twin(
            frequentist.one_mean(values, mu0=np.log(reference)),
            " on the log scale",
            f"the true typical value were exactly {reference:g}",
        )
        # Back on the original scale the posterior is exp of a Student-t,
        # which has no finite mean; the draws carry everything that is
        # reported.
        draws, post = np.exp(draws), None
    else:
        twin = frequentist.one_mean(values, mu0=reference)
    log_bf10 = normal_t.log_bayes_factor_ttest(
        twin.statistic, n_effective=n, df=n - 1, scale=resolved.scale
    )

    if log:
        subject = f"typical {label}".strip() if label else "typical value"
        quantity = f"{subject} (geometric mean, from {n} observations)"
    else:
        subject = label or "average"
        quantity = f"average {label}".strip() + f" (from {n} observations)"
    notes = []
    if n < 15:
        notes.append(
            f"only {n} observations, so this leans on the assumption that the "
            f"{'logged ' if log else ''}values are roughly normally distributed "
            "— check a histogram before trusting the tails"
        )

    def _predictor() -> Prediction:
        scale = sd * np.sqrt(1.0 + 1.0 / n)
        forecast = stats.t(df=n - 1, loc=sample_mean, scale=scale)
        crit = stats.t.ppf(0.975, n - 1)
        textbook = (sample_mean - crit * scale, sample_mean + crit * scale)
        if log:
            textbook = (float(np.exp(textbook[0])), float(np.exp(textbook[1])))
        return Prediction(
            what=f"next {label or 'observation'}",
            dist=forecast,
            plug_in=stats.norm(loc=sample_mean, scale=sd),
            frequentist_interval=textbook,
            frequentist_method="t prediction interval, 95%",
            unit=unit,
            decimals=2,
            transform=np.exp if log else None,
            inverse=_safe_log if log else None,
        )

    def _refit(spec):
        return mean(
            x,
            prior=spec,
            reference=reference,
            label=label,
            unit=unit,
            log=log,
            n_draws=n_draws,
            seed=seed,
        )

    return Result(
        quantity=quantity,
        draws=draws,
        posterior=post,
        prior=resolved,
        subject=subject,
        analysis="mean",
        frequentist=twin,
        log_bf10=log_bf10,
        bf_alternative="a real difference from the reference",
        bf_caveat=(
            f"Compares 'the {'typical value' if log else 'average'} differs "
            f"from {reference:g}' against 'it is exactly {reference:g}'. Unlike "
            "the interval above, this number "
            "does move with the prior — run .sensitivity() to see by how much."
        ),
        unit=unit,
        display_scale=1.0,
        decimals=2,
        direction_reference=reference,
        components={subject: post if post is not None else draws},
        component_axis=f"{subject} ({unit})" if unit else subject,
        predictor=_predictor,
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
    paired: bool = False,
    log: bool = False,
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

    With ``paired=True`` the two samples are two measurements on the same
    units, and the analysis is of the within-pair differences: an exact
    Student-t posterior for their average, paired with the paired t-test.

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
        belief. Has no meaning for paired data.
    paired : bool, default False
        Treat ``x[i]`` and ``y[i]`` as two measurements on the same unit --
        the same intersection before and after, the same tract in two years.
        Pairs with either value missing are dropped. Pairing removes the
        variation between units that both measurements share, and the summary
        says how much narrower that makes the interval.
    log : bool, default False
        Compare on the log scale and report the ratio of typical
        (geometric-mean) values, ``y`` over ``x``. For skewed, strictly
        positive data such as prices or incomes, where "B is 1.3 times A" is
        both the better-behaved analysis and the better sentence.
    unit : str, optional
        Display unit for the difference. Ignored with ``log=True``, where the
        answer is a ratio.
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

    The same twelve intersections, measured before and after a signal
    retiming. Every one improved a little, but the intersections differ from
    each other far more than that, so only the paired analysis can see it:

    >>> before = np.array([41, 55, 38, 62, 47, 51, 36, 58, 44, 49, 53, 40.0])
    >>> after = before - np.array([3, 2, 4, 1, 3, 2, 5, 2, 3, 4, 2, 3.0])
    >>> paired = bp.compare_means(before, after, paired=True)
    >>> round(paired.probability("<", 0), 3)
    1.0
    >>> round(bp.compare_means(before, after).probability("<", 0), 2)
    0.77
    """
    if paired and equal_var:
        raise ValueError(
            "equal_var has no meaning for paired data: the analysis is of the "
            "within-pair differences, which form a single sample. Drop one of "
            "the two options."
        )
    if labels is None:
        labels = ("group 1", "group 2")
    labels = tuple(str(item) for item in labels)
    if len(labels) != 2:
        raise ValueError(f"labels needs exactly two names, got {len(labels)}.")

    if paired:
        a = np.asarray(x, dtype=float).ravel()
        b = np.asarray(y, dtype=float).ravel()
        if a.size != b.size:
            raise ValueError(
                f"paired=True needs one y for every x, but x has {a.size} values "
                f"and y has {b.size}."
            )
        keep = np.isfinite(a) & np.isfinite(b)
        a, b = a[keep], b[keep]
    else:
        a, b = _finite(x), _finite(y)
    if log:
        a, b = _positive(a, "x"), _positive(b, "y")

    resolved = priors.resolve_effect_size(prior)
    n_draws = get_draws() if n_draws is None else int(n_draws)
    rng = make_rng(seed)

    if paired:
        return _compare_paired(a, b, resolved, labels, log, unit, n_draws, seed, rng)

    n1, mean1, sd1 = normal_t.summarise(a, "x")
    n2, mean2, sd2 = normal_t.summarise(b, "y")
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

    twin = frequentist.two_means(a, b, equal_var=equal_var)
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
            f"on an assumption of roughly normal {'logged ' if log else ''}data "
            "— check a histogram"
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
            log=log,
            unit=unit,
            n_draws=n_draws,
            seed=seed,
        )

    components = {
        f"{labels[0]} (n={n1})": post1,
        f"{labels[1]} (n={n2})": post2,
    }
    if log:
        return _ratio_result(
            draws=np.exp(draws),
            twin=_exp_twin(
                twin,
                " on the log scale",
                "the two groups' typical values were identical",
            ),
            log_bf10=log_bf10,
            resolved=resolved,
            labels=labels,
            unit=unit,
            components={
                name: np.exp(post.rvs(size=n_draws, random_state=rng))
                for name, post in components.items()
            },
            refit=_refit,
            n_draws=n_draws,
            notes=notes,
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
        components=components,
        component_axis=f"average for each group ({unit})" if unit else "group average",
        refit=_refit,
        prior_ladder=priors.EFFECT_SENSITIVITY_LADDER,
        n_draws=n_draws,
        notes=notes,
    )


def _compare_paired(a, b, resolved, labels, log, unit, n_draws, seed, rng):
    """Paired comparison: one exact posterior for the average difference."""
    differences = b - a
    n, mean_d, sd_d = normal_t.summarise(
        differences, "the within-pair differences (y − x)"
    )
    posterior = normal_t.mean_posterior(n, mean_d, sd_d)
    draws = posterior.rvs(size=n_draws, random_state=rng)
    twin = frequentist.paired_means(a, b)
    log_bf10 = normal_t.log_bayes_factor_ttest(
        twin.statistic, n_effective=n, df=n - 1, scale=resolved.scale
    )

    notes = [_pairing_note(a, b, twin, log)]
    if n < 15:
        notes.append(
            f"only {n} pairs, so this leans on the within-pair differences "
            "being roughly normal — check a histogram of them"
        )

    # Each group's own average, for the components plot. These are wide and
    # overlapping whenever units differ a lot from each other, which is the
    # picture that shows why pairing matters.
    components = {}
    for name, sample in zip(labels, (a, b)):
        if sample.size >= 2 and np.ptp(sample) > 0:
            m, sd = float(sample.mean()), float(sample.std(ddof=1))
            components[f"{name} (n={n})"] = normal_t.mean_posterior(n, m, sd)

    def _refit(spec):
        return _compare_paired(
            a,
            b,
            priors.resolve_effect_size(spec),
            labels,
            log,
            unit,
            n_draws,
            seed,
            make_rng(seed),
        )

    if log:
        return _ratio_result(
            draws=np.exp(draws),
            twin=_exp_twin(
                twin,
                " on the log scale",
                "the typical within-pair ratio were exactly 1",
            ),
            log_bf10=log_bf10,
            resolved=resolved,
            labels=labels,
            unit=unit,
            components={
                name: np.exp(post.rvs(size=n_draws, random_state=rng))
                for name, post in components.items()
            },
            refit=_refit,
            n_draws=n_draws,
            notes=notes,
            paired=True,
        )

    return Result(
        quantity=f"average within-pair difference ({labels[1]} − {labels[0]})",
        draws=draws,
        posterior=posterior,
        prior=resolved,
        subject="average difference",
        analysis="compare_means",
        frequentist=twin,
        log_bf10=log_bf10,
        bf_alternative="a difference",
        bf_caveat=(
            "Compares 'the average within-pair difference is not zero' against "
            "'it is exactly zero', with a Cauchy prior on standardised effect "
            "size. The credible interval above does not depend on this prior; "
            "this number does — run .sensitivity()."
        ),
        unit=unit,
        display_scale=1.0,
        decimals=2,
        direction_reference=0.0,
        higher_label=labels[1],
        lower_label=labels[0],
        components=components,
        component_axis=f"average for each group ({unit})" if unit else "group average",
        refit=_refit,
        prior_ladder=priors.EFFECT_SENSITIVITY_LADDER,
        n_draws=n_draws,
        notes=notes,
    )


def _pairing_note(a, b, paired_twin, log) -> str:
    """Say how much the pairing bought, against an unpaired analysis."""
    unpaired = frequentist.two_means(a, b)
    width_paired = paired_twin.interval[1] - paired_twin.interval[0]
    width_unpaired = unpaired.interval[1] - unpaired.interval[0]
    scale = "log-scale " if log else ""
    if width_paired < width_unpaired:
        return (
            f"pairing matters here: analysed as two independent groups, the "
            f"{scale}interval would be {width_unpaired / width_paired:.1f} times "
            "as wide, because each unit's own level — shared by both of its "
            "measurements — would count as noise"
        )
    return (
        "pairing did not narrow the interval here: the two measurements on each "
        "unit are not positively related, so there was no shared variation "
        "for the pairing to remove"
    )


def _ratio_result(
    *,
    draws,
    twin,
    log_bf10,
    resolved,
    labels,
    unit,
    components,
    refit,
    n_draws,
    notes,
    paired=False,
):
    """Assemble a log-scale comparison, reported as a ratio of typical values."""
    what = "typical within-pair ratio" if paired else "ratio of typical values"
    return Result(
        quantity=f"{what} ({labels[1]} ÷ {labels[0]}, geometric means)",
        draws=draws,
        posterior=None,
        prior=resolved,
        subject=what,
        analysis="compare_means",
        frequentist=twin,
        log_bf10=log_bf10,
        bf_alternative="a difference",
        bf_caveat=(
            "Computed on the log scale: compares 'the typical values differ' "
            "against 'they are identical', with a Cauchy prior on standardised "
            "effect size. The interval above does not depend on this prior; "
            "this number does — run .sensitivity()."
        ),
        unit="times",
        display_scale=1.0,
        decimals=2,
        direction_reference=1.0,
        higher_label=labels[1],
        lower_label=labels[0],
        components=components,
        component_axis=(
            f"typical value for each group ({unit})" if unit else "typical value"
        ),
        refit=refit,
        prior_ladder=priors.EFFECT_SENSITIVITY_LADDER,
        n_draws=n_draws,
        notes=notes,
    )
