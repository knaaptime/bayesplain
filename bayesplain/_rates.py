"""Rates of events over an exposure, and comparing two of them.

A proportion is a share of cases; a rate is a count of events per unit of
watching -- collisions per year, evictions per thousand renter households,
crashes per million vehicle-miles. Planners ask the second kind of question
at least as often as the first, and treating a count over time as if it were a
share of cases is one of the most common mistakes in applied work.

The posterior is exact: a Gamma prior on the rate updates to a Gamma
posterior by adding the events to one parameter and the exposure to the
other. The ratio of two rates has an exact posterior too, so the headline
comparison involves no sampling at all.

The frequentist counterparts are the exact Poisson test with its Garwood
interval, and the exact conditional test for two rates -- which, conditional on
the total count, is just a binomial test on one group's share of the events.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from . import frequentist, priors
from ._config import get_draws, make_rng
from .core import gamma_poisson as gp
from .result import Prediction, Result

__all__ = ["rate", "compare_rates"]

_POISSON_NOTE = (
    "this assumes events arrive independently at a steady rate. Events that "
    "cluster — several injuries in one crash, a run of filings at one "
    "building — vary more than that, and the interval will be too narrow"
)


def _decimals_for(value: float) -> int:
    """Enough decimal places to show a rate's leading digits."""
    if not np.isfinite(value) or value <= 0:
        return 2
    return int(min(6, max(2, 2 - np.floor(np.log10(value)))))


def _read_periods(events, exposure):
    """Collapse per-period counts to a total, keeping the periods for checks."""
    y = np.atleast_1d(np.asarray(events, dtype=float)).ravel()
    t = np.atleast_1d(np.asarray(exposure, dtype=float)).ravel()
    if t.size == 1 and y.size > 1:
        t = np.full(y.size, t[0])
    if y.size != t.size:
        raise ValueError(
            f"events has {y.size} periods but exposure has {t.size}; pass one "
            "exposure per period, or a single exposure shared by every period."
        )
    checked = [gp.validate_events(yi, ti) for yi, ti in zip(y, t)]
    y = np.array([c[0] for c in checked], dtype=float)
    t = np.array([c[1] for c in checked], dtype=float)
    return y, t


def _dispersion_note(y, t):
    """Flag counts that vary between periods more than a Poisson rate allows."""
    if y.size < 3 or y.sum() == 0:
        return None
    fitted = t * y.sum() / t.sum()
    statistic = float(((y - fitted) ** 2 / fitted).sum())
    df = y.size - 1
    ratio = statistic / df
    pvalue = float(stats.chi2.sf(statistic, df))
    if ratio > 1.5 and pvalue < 0.05:
        return (
            f"the counts vary between periods about {ratio:.1f} times as much "
            "as a steady rate would produce (dispersion test p = "
            f"{pvalue:.2g}). The rate is probably not steady, and this "
            "interval understates the uncertainty — report it as a lower "
            "bound on how uncertain you are"
        )
    return None


# ---------------------------------------------------------------------------
# One rate
# ---------------------------------------------------------------------------


def rate(
    events,
    exposure=1.0,
    reference: float | None = None,
    prior="jeffreys",
    label: str = "",
    unit: str = "",
    n_draws: int | None = None,
    seed="unset",
) -> Result:
    """Estimate one rate of events per unit of exposure.

    With a ``Gamma(a, b)`` prior and ``events`` over ``exposure``, the posterior
    is ``Gamma(a + events, b + exposure)`` -- exact, with nothing to converge.

    Parameters
    ----------
    events : int or array_like
        Events counted. Pass one count per period (with ``exposure`` per
        period, or a single shared one) and the counts are summed -- and
        checked for whether they vary more than a steady rate allows.
    exposure : float or array_like, default 1.0
        How much was watched, in whatever units you want the rate reported
        in. To report crashes per million vehicle-miles, give the exposure in
        millions of vehicle-miles.
    reference : float
        The rate to compare against, per unit of exposure: a statewide rate,
        last year's figure, a target. Required, because unlike a proportion
        there is no neutral default -- a rate of 0.5 means nothing until you
        say per what.
    prior : str, tuple, or RatePrior, default 'jeffreys'
        A preset name, a ``(shape, exposure)`` pair, or the result of
        :func:`bayesplain.priors.from_previous_period`.
    label : str, optional
        What is being counted, e.g. ``"pedestrian collisions"``.
    unit : str, optional
        How to print the rate, e.g. ``"per year"`` or ``"per 1,000
        households"``.
    n_draws : int, optional
        Draws to take. Defaults to the package setting.
    seed : int or None, optional
        Seed for those draws.

    Returns
    -------
    Result
        Call ``.summary()`` for the full report, and ``.predict(exposure=...)``
        for a forecast of the next period's count.

    Examples
    --------
    An intersection saw 14 pedestrian collisions over 4 years; the citywide
    average for comparable intersections is 2 a year.

    >>> import bayesplain as bp
    >>> res = bp.rate(14, exposure=4, reference=2.0, unit="per year")
    >>> round(res.point(), 2)
    3.54
    >>> round(res.probability(">", 2.0), 3)
    0.976
    >>> res.predict(exposure=1).interval()
    (0.0, 8.0)
    """
    if reference is None:
        raise ValueError(
            "rate() needs a reference rate to compare against — a citywide "
            "average, last year's figure, a target. It is not optional because "
            "a rate has no neutral default the way a proportion has 50%."
        )
    y_periods, t_periods = _read_periods(events, exposure)
    events_total = int(y_periods.sum())
    exposure_total = float(t_periods.sum())
    if not reference > 0:
        raise ValueError(f"reference must be a positive rate, got {reference}.")

    resolved = priors.resolve_rate(prior)
    a, b = resolved.shape, resolved.exposure
    n_draws = get_draws() if n_draws is None else int(n_draws)
    rng = make_rng(seed)

    post = gp.posterior(events_total, exposure_total, a, b)
    draws = post.rvs(size=n_draws, random_state=rng)
    log_bf10 = gp.log_bayes_factor_point_null(
        events_total, exposure_total, reference, a, b
    )
    twin = frequentist.one_rate(events_total, exposure_total, rate0=reference)

    notes = []
    if events_total < 10:
        notes.append(
            f"only {events_total} events, so this rate rests on very few "
            "occurrences and chance alone moves it a lot — the wide interval "
            "is the honest summary"
        )
    dispersion = _dispersion_note(y_periods, t_periods)
    if dispersion:
        notes.append(dispersion)
    notes.append(_POISSON_NOTE)

    subject = f"rate of {label}" if label else "rate"
    decimals = _decimals_for(float(post.median()))

    def _predictor(exposure: float = 1.0) -> Prediction:
        forecast = gp.predictive(events_total, exposure_total, exposure, a, b)
        plug_in = stats.poisson(events_total / exposure_total * exposure)
        return Prediction(
            what=(
                f"number of {label or 'events'} over a further exposure of {exposure:g}"
            ),
            dist=forecast,
            plug_in=plug_in,
            discrete=True,
        )

    def _refit(spec):
        return rate(
            y_periods,
            t_periods,
            reference=reference,
            prior=spec,
            label=label,
            unit=unit,
            n_draws=n_draws,
            seed=seed,
        )

    return Result(
        quantity=(
            f"{subject} ({events_total:,} events over an exposure of "
            f"{exposure_total:,g})"
        ),
        draws=draws,
        posterior=post,
        prior=resolved,
        subject=subject,
        analysis="rate",
        frequentist=twin,
        log_bf10=log_bf10,
        bf_alternative="a rate different from the reference",
        bf_caveat=(
            f"Compares 'the rate is unknown' against 'it is exactly "
            f"{reference:g}'. With a preset prior the alternative spreads the "
            "ratio of the rate to the reference evenly on a log scale around "
            "one; the number moves with that choice. Run .sensitivity()."
        ),
        unit=unit,
        display_scale=1.0,
        decimals=decimals,
        direction_reference=float(reference),
        components={subject: post},
        component_axis=f"{subject} ({unit})" if unit else subject,
        predictor=_predictor,
        refit=_refit,
        prior_ladder=priors.RATE_SENSITIVITY_LADDER,
        n_draws=n_draws,
        notes=notes,
        next_steps=(".probability('>', x)  .predict(exposure=1)  .sensitivity()"),
    )


# ---------------------------------------------------------------------------
# Two rates
# ---------------------------------------------------------------------------


def compare_rates(
    events,
    exposure=(1.0, 1.0),
    prior="jeffreys",
    labels=None,
    estimand: str = "ratio",
    unit: str = "",
    n_draws: int | None = None,
    seed="unset",
) -> Result:
    """Compare the rate of events between two groups.

    The rate ratio has an exact posterior -- a scaled beta-prime distribution
    -- so the headline answer involves no sampling. The difference in rates is
    drawn from the two exact Gamma posteriors and subtracted.

    Parameters
    ----------
    events : array_like
        Two event counts, ``[y1, y2]``.
    exposure : array_like, default (1.0, 1.0)
        Two exposures, ``[t1, t2]``, in the same units. Unequal exposure is
        the whole reason to compare rates rather than counts: 40 crashes in
        ten years is a lower rate than 12 in two.
    prior : str, tuple, or RatePrior, default 'jeffreys'
        Prior applied to each group's rate.
    labels : sequence of str, optional
        Group names. Defaults to ``('group 1', 'group 2')``.
    estimand : {'ratio', 'difference'}, default 'ratio'
        Which comparison to report, oriented as group 2 relative to group 1.
        The ratio is the conventional one for rates, and is exact.
    unit : str, optional
        How to print a rate, e.g. ``"per year"``. Used for the difference and
        for each group's own rate.
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
    Pedestrian collisions before and after a road diet: 31 in the four years
    before, 9 in the two years since.

    >>> import bayesplain as bp
    >>> res = bp.compare_rates(
    ...     events=[31, 9], exposure=[4, 2], labels=["before", "after"],
    ...     unit="per year",
    ... )
    >>> round(res.point(), 2)
    0.59
    >>> round(res.probability("<", 1.0), 3)
    0.931
    """
    events = np.asarray(events)
    exposure = np.asarray(exposure)
    if events.shape != (2,) or exposure.shape != (2,):
        raise ValueError(
            "compare_rates needs exactly two groups: two event counts and two "
            "exposures."
        )
    (y1, t1), (y2, t2) = (gp.validate_events(y, t) for y, t in zip(events, exposure))
    if y1 + y2 == 0:
        raise ValueError(
            "no events in either group, so there is nothing to compare. The "
            "data are consistent with any ratio at all."
        )
    if estimand not in {"ratio", "difference"}:
        raise ValueError(f"estimand must be 'ratio' or 'difference', got {estimand!r}.")
    if labels is None:
        labels = ("group 1", "group 2")
    labels = tuple(str(item) for item in labels)
    if len(labels) != 2:
        raise ValueError(f"labels needs exactly two names, got {len(labels)}.")

    resolved = priors.resolve_rate(prior)
    a, b = resolved.shape, resolved.exposure
    n_draws = get_draws() if n_draws is None else int(n_draws)
    rng = make_rng(seed)

    post1 = gp.posterior(y1, t1, a, b)
    post2 = gp.posterior(y2, t2, a, b)
    if estimand == "ratio":
        posterior = gp.ratio_posterior(y1, t1, y2, t2, a, b)
        draws = posterior.rvs(size=n_draws, random_state=rng)
        quantity = f"rate ratio ({labels[1]} ÷ {labels[0]})"
        reference, shown_unit, decimals = 1.0, "times", 2
    else:
        posterior = None  # difference of two Gammas has no tidy closed form
        draws = post2.rvs(size=n_draws, random_state=rng) - post1.rvs(
            size=n_draws, random_state=rng
        )
        quantity = f"difference in rate ({labels[1]} − {labels[0]})"
        reference, shown_unit = 0.0, unit
        decimals = _decimals_for(max(y1 / t1, y2 / t2))

    log_bf10 = gp.log_bayes_factor_equal_rates(y1, t1, y2, t2, a, b)
    twin = frequentist.two_rates([y1, y2], [t1, t2], estimand=estimand)

    notes = []
    if min(y1, y2) < 5:
        notes.append(
            f"one group has only {min(y1, y2)} events; the exact posterior "
            "handles that, but expect a wide interval"
        )
    notes.append(_POISSON_NOTE)

    def _refit(spec):
        return compare_rates(
            [y1, y2],
            [t1, t2],
            prior=spec,
            labels=labels,
            estimand=estimand,
            unit=unit,
            n_draws=n_draws,
            seed=seed,
        )

    return Result(
        quantity=quantity,
        draws=draws,
        posterior=posterior,
        prior=resolved,
        analysis="compare_rates",
        frequentist=twin,
        log_bf10=log_bf10,
        bf_caveat=(
            "Compares 'the two rates differ' against 'they are the same', "
            "conditioning on the total number of events. The alternative "
            "spreads the rate ratio evenly on a log scale around one; the "
            "number moves with that choice. Run .sensitivity()."
        ),
        unit=shown_unit,
        display_scale=1.0,
        decimals=decimals,
        direction_reference=reference,
        higher_label=labels[1],
        lower_label=labels[0],
        components={
            f"{labels[0]} ({y1} over {t1:g})": post1,
            f"{labels[1]} ({y2} over {t2:g})": post2,
        },
        component_axis=f"rate for each group ({unit})" if unit else "rate",
        component_scale=1.0,
        refit=_refit,
        prior_ladder=priors.RATE_SENSITIVITY_LADDER,
        n_draws=n_draws,
        notes=notes,
        no_predict_reason=(
            "compare_rates has no single next observation to forecast. Run "
            "rate() on each group and call .predict() on each."
        ),
    )
