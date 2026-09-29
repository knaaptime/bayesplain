"""How much one thing changes with another: a line with one predictor.

The frequentist counterpart is the t-test on a least-squares slope, usually met
right after correlation. The two analyses answer different questions about the
same scatter plot: a correlation says how tightly the points cluster around a
line, in unitless terms; a slope says how much the outcome changes per unit of
the predictor, in the units a decision is made in -- dollars per square foot,
minutes per mile.

Under the reference prior the slope's posterior is an exact Student-t, and its
interval coincides with the confidence interval. So does the forecast for a
new observation, which coincides with the textbook prediction interval.

This stops at one predictor on purpose. See :doc:`/scope`.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy import stats

from . import frequentist, priors
from ._config import get_draws, make_rng
from ._frame import column
from .core import normal_t
from .core import regression as core_reg
from .result import Prediction, Result, _missing_note

__all__ = ["regression"]


def regression(
    x,
    y,
    prior="conventional",
    labels=None,
    unit: str = "",
    reference: float = 0.0,
    n_draws: int | None = None,
    seed="unset",
    data=None,
) -> Result:
    """Estimate how much ``y`` changes per unit of ``x``, with one predictor.

    Parameters
    ----------
    x, y : array_like or str
        Predictor and outcome, or the names of two columns in ``data``. Rows
        where either is missing are dropped, and the summary says how many.
    prior : str, float, or EffectSizePrior, default 'conventional'
        Cauchy prior on the standardised slope. Affects the Bayes factor
        only; the posterior and credible interval do not depend on it.
    labels : sequence of str, optional
        Names of the predictor and the outcome, used in output.
    unit : str, optional
        Unit of the slope, e.g. ``"dollars per square foot"``.
    reference : float, default 0.0
        The slope to compare against. Zero means "no relationship"; set a
        policy-relevant value when one exists.
    n_draws : int, optional
        Number of draws. Defaults to the package setting.
    seed : int or None, optional
        Seed for the draws.
    data : data frame, optional
        Where to look ``x`` and ``y`` up when they are column names. The column
        names become the labels unless ``labels`` is given.

    Returns
    -------
    Result
        Call ``.summary()``, ``.predict(x=...)`` for a forecast at a new value
        of the predictor, and ``.plot(kind='line')`` for the fitted line with
        its credible band.

    Examples
    --------
    >>> import numpy as np, bayesplain as bp
    >>> rng = np.random.default_rng(0)
    >>> sqft = rng.uniform(800, 3000, 60)
    >>> value = 150 * sqft + rng.normal(0, 60_000, 60)
    >>> res = bp.regression(sqft, value, labels=["floor area", "assessed value"],
    ...                     unit="dollars per sq ft")
    >>> res.exact
    True
    >>> lo, hi = res.interval(kind="eti")
    >>> bool(np.allclose((lo, hi), res.frequentist.interval))
    True
    """
    x, x_name = column(x, data, "x")
    y, y_name = column(y, data, "y")
    if labels is None and x_name and y_name:
        labels = (x_name, y_name)
    fit = core_reg.fit_line(x, y)
    resolved = priors.resolve_effect_size(prior)
    n_draws = get_draws() if n_draws is None else int(n_draws)
    rng = make_rng(seed)

    if labels is None:
        labels = ("x", "y")
    labels = tuple(str(item) for item in labels)
    if len(labels) != 2:
        raise ValueError(f"labels needs exactly two names, got {len(labels)}.")

    post = core_reg.slope_posterior(fit)
    draws = post.rvs(size=n_draws, random_state=rng)
    twin = frequentist.simple_regression(x, y)
    if reference != 0.0:
        # The twin's p-value is for a zero slope; rebuild it for the reference.
        t_ref = (fit.slope - reference) / fit.slope_se
        twin = _retarget(twin, t_ref, fit.n - 2, reference)
    log_bf10 = _log_bf(fit, reference, resolved.scale)

    a = np.asarray(x, dtype=float).ravel()
    b = np.asarray(y, dtype=float).ravel()
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    missing = _missing_note(int((~keep).sum()), fit.n, "incomplete rows")

    notes = [
        f"one predictor only. The slope describes how {labels[1]} moves with "
        f"{labels[0]} in this data, not what would happen if you changed "
        f"{labels[0]}: anything else that moves with {labels[0]} is folded "
        "into this number. For more predictors, see the scope page (Bambi).",
    ]
    if missing:
        notes.append(missing)
    if fit.n < 15:
        notes.append(
            f"only {fit.n} points, so this leans on the scatter around the line "
            "being roughly normal — look at the residuals before trusting the "
            "tails"
        )

    def _predictor(x: float) -> Prediction:
        x0 = float(x)
        forecast = core_reg.predictive(fit, x0)
        # The textbook prediction interval, built the textbook way.
        crit = stats.t.ppf(0.975, fit.n - 2)
        half = (
            crit
            * fit.residual_sd
            * np.sqrt(1.0 + 1.0 / fit.n + (x0 - fit.x_mean) ** 2 / fit.sxx)
        )
        centre = float(fit.at(x0))
        textbook = (centre - half, centre + half)
        outside = x0 < a.min() or x0 > a.max()
        what = f"{labels[1]} at {labels[0]} = {x0:g}"
        if outside:
            what += " (outside the observed range — an extrapolation)"
        return Prediction(
            what=what,
            dist=forecast,
            plug_in=stats.norm(loc=float(fit.at(x0)), scale=fit.residual_sd),
            frequentist_interval=textbook,
            frequentist_method="t prediction interval, 95%",
            decimals=2,
        )

    def _refit(spec):
        return regression(
            a,
            b,
            prior=spec,
            labels=labels,
            unit=unit,
            reference=reference,
            n_draws=n_draws,
            seed=seed,
        )

    return Result(
        quantity=f"change in {labels[1]} per unit of {labels[0]} (slope)",
        draws=draws,
        posterior=post,
        prior=resolved,
        subject="slope",
        analysis="regression",
        frequentist=twin,
        log_bf10=log_bf10,
        bf_alternative="a slope different from the reference",
        bf_caveat=(
            f"Compares 'the slope differs from {reference:g}' against 'it is "
            f"exactly {reference:g}', with a Cauchy prior on the standardised "
            "slope (Zellner-Siow). R's regressionBF uses a narrower default, "
            "0.354; pass prior=0.354 to match it. The interval above does not "
            "depend on this prior; this number does — run .sensitivity()."
        ),
        unit=unit,
        display_scale=1.0,
        decimals=_decimals_for(fit.slope_se),
        direction_reference=float(reference),
        components=None,
        custom_plots={"line": _line_plot(a, b, fit, labels)},
        predictor=_predictor,
        refit=_refit,
        prior_ladder=priors.EFFECT_SENSITIVITY_LADDER,
        n_draws=n_draws,
        notes=notes,
        next_steps=".predict(x=...)  .plot(kind='line')  .sensitivity()",
    )


def _log_bf(fit, reference: float, scale: float) -> float:
    """Zellner-Siow Bayes factor against a slope of ``reference``.

    Subtracting ``reference * x`` from ``y`` turns "the slope is reference"
    into "the slope is zero" without changing the scatter around the line, so
    the zero-slope integral applies to the shifted t statistic.
    """
    if reference == 0.0:
        return core_reg.log_bayes_factor_slope(fit, scale=scale)
    t_ref = (fit.slope - reference) / fit.slope_se
    return normal_t.log_bayes_factor_ttest(
        t_ref, n_effective=fit.n, df=fit.n - 2, scale=scale
    )


def _retarget(twin, t_ref: float, df: int, reference: float):
    """Re-aim the slope t-test at a nonzero reference slope."""
    return replace(
        twin,
        statistic=float(t_ref),
        pvalue=float(2.0 * stats.t.sf(abs(t_ref), df)),
        null_statement=f"the true slope were exactly {reference:g}",
    )


def _decimals_for(se: float) -> int:
    """Show the slope to roughly the precision its standard error supports."""
    if not np.isfinite(se) or se <= 0:
        return 3
    return int(min(6, max(0, 2 - np.floor(np.log10(se)))))


def _line_plot(a, b, fit, labels):
    """Scatter, fitted line, and a 95% credible band for the average outcome."""

    def draw(ax):
        ax.scatter(a, b, s=22, alpha=0.55, color="#4a4e69", edgecolor="none")
        grid = np.linspace(a.min(), a.max(), 200)
        crit = stats.t.ppf(0.975, fit.n - 2)
        leverage = 1.0 / fit.n + (grid - fit.x_mean) ** 2 / fit.sxx
        band = fit.residual_sd * np.sqrt(leverage)
        line = fit.at(grid)
        ax.fill_between(
            grid, line - crit * band, line + crit * band, color="#c9184a", alpha=0.18
        )
        ax.plot(grid, line, color="#c9184a", lw=1.8)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(
            "fitted line with a 95% credible band for the average outcome",
            fontsize=10,
        )
        return ax

    return draw
