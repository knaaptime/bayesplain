r"""Simple linear regression -- one predictor -- under the reference prior.

With :math:`y_i = \alpha + \beta x_i + \varepsilon_i`, normal errors, and the
reference prior :math:`p(\alpha, \beta, \sigma^2) \propto 1/\sigma^2`, the
posterior for the slope is exactly a scaled Student-t:

.. math::

    \beta \mid \text{data} \sim t_{n-2}\!\left(\hat\beta,\;
        s^2 / S_{xx}\right)

where :math:`s^2` is the residual variance and :math:`S_{xx}` the sum of
squared deviations of ``x``. The same is true of the average outcome at any
value of ``x``, and of a new observation there. All closed form; and, as with
a single mean, the equal-tailed interval coincides with the textbook
confidence interval digit for digit.

The Bayes factor
----------------
The Zellner-Siow Bayes factor for a slope of zero (Liang et al., 2008; Rouder
and Morey, 2012) places a Cauchy prior of scale ``r`` on the standardised
slope, :math:`\beta\, s_x / \sigma`. With one predictor it depends on the data
only through the slope's t statistic, and equals the JZS t-test integral with
:math:`N = n` and :math:`\nu = n - 2` -- so it is computed by
:func:`bayesplain.core.normal_t.log_bayes_factor_ttest`, which the
validation tests already cross-check, rather than by a second implementation
of the same integral.

The line stops at one predictor on purpose. Two or more require integrating
over several hyperparameters at once, which is where a sampler becomes
necessary and this package hands over to Bambi.

References
----------
Liang, F., Paulo, R., Molina, G., Clyde, M. A., and Berger, J. O. (2008).
Mixtures of g priors for Bayesian variable selection. *Journal of the American
Statistical Association*, 103(481), 410-423.

Rouder, J. N., and Morey, R. D. (2012). Default Bayes factors for model
selection in regression. *Multivariate Behavioral Research*, 47(6), 877-903.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from . import normal_t

__all__ = [
    "LineFit",
    "fit_line",
    "slope_posterior",
    "mean_response_posterior",
    "predictive",
    "log_bayes_factor_slope",
]


@dataclass(frozen=True)
class LineFit:
    """The sufficient statistics of a least-squares line.

    Attributes
    ----------
    n : int
        Number of complete pairs.
    slope, intercept : float
        Least-squares estimates.
    residual_sd : float
        :math:`s`, with ``n - 2`` degrees of freedom.
    x_mean : float
        Mean of the predictor.
    sxx : float
        Sum of squared deviations of the predictor.
    r_squared : float
        Share of the variance in ``y`` the line accounts for.
    """

    n: int
    slope: float
    intercept: float
    residual_sd: float
    x_mean: float
    sxx: float
    r_squared: float

    @property
    def slope_se(self) -> float:
        """Standard error of the slope."""
        return self.residual_sd / np.sqrt(self.sxx)

    @property
    def t(self) -> float:
        """The slope's t statistic."""
        return self.slope / self.slope_se

    def at(self, x0) -> np.ndarray:
        """Fitted value of the line at ``x0``."""
        return self.intercept + self.slope * np.asarray(x0, dtype=float)


def fit_line(x, y) -> LineFit:
    """Fit a least-squares line, dropping rows where either value is missing.

    Parameters
    ----------
    x, y : array_like
        Predictor and outcome.

    Returns
    -------
    LineFit
        The quantities every posterior below is built from.

    Raises
    ------
    ValueError
        With fewer than three complete pairs, a constant predictor, or a
        perfect fit, none of which leave a residual variance to estimate.
    """
    a = np.asarray(x, dtype=float)
    if a.ndim > 1 and a.shape[1] != 1:
        raise ValueError(
            f"regression() takes one predictor, but x has {a.shape[1]} columns. "
            "With more than one predictor the posterior is no longer closed "
            "form; use Bambi (bmb.Model('y ~ x1 + x2', data))."
        )
    a = a.ravel()
    b = np.asarray(y, dtype=float).ravel()
    if a.size != b.size:
        raise ValueError(
            f"x has {a.size} values but y has {b.size}; they must line up."
        )
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    n = int(a.size)
    if n < 3:
        raise ValueError(
            f"need at least 3 complete (x, y) pairs to fit a line and "
            f"estimate the scatter around it, got {n}."
        )
    if np.ptp(a) == 0:
        raise ValueError(
            "every value of x is identical, so there is no slope to estimate."
        )
    x_mean = float(a.mean())
    sxx = float(((a - x_mean) ** 2).sum())
    slope = float(((a - x_mean) * (b - b.mean())).sum() / sxx)
    intercept = float(b.mean() - slope * x_mean)
    residuals = b - (intercept + slope * a)
    sse = float((residuals**2).sum())
    if sse <= 0:
        raise ValueError(
            "the points lie exactly on a line, so there is no scatter to "
            "estimate and the posterior is undefined."
        )
    syy = float(((b - b.mean()) ** 2).sum())
    return LineFit(
        n=n,
        slope=slope,
        intercept=intercept,
        residual_sd=float(np.sqrt(sse / (n - 2))),
        x_mean=x_mean,
        sxx=sxx,
        r_squared=float(1.0 - sse / syy) if syy > 0 else 0.0,
    )


def slope_posterior(fit: LineFit):
    """Exact Student-t posterior for the slope.

    Parameters
    ----------
    fit : LineFit
        From :func:`fit_line`.

    Returns
    -------
    scipy.stats.rv_continuous_frozen
        ``t(n - 2, slope, se)``.
    """
    return stats.t(df=fit.n - 2, loc=fit.slope, scale=fit.slope_se)


def mean_response_posterior(fit: LineFit, x0: float):
    """Exact posterior for the average outcome at ``x0``.

    Parameters
    ----------
    fit : LineFit
        From :func:`fit_line`.
    x0 : float
        Value of the predictor.

    Returns
    -------
    scipy.stats.rv_continuous_frozen
        Scaled Student-t with ``n - 2`` degrees of freedom. It widens away
        from the middle of the data, which is why extrapolation is costly.
    """
    scale = fit.residual_sd * np.sqrt(1.0 / fit.n + (x0 - fit.x_mean) ** 2 / fit.sxx)
    return stats.t(df=fit.n - 2, loc=float(fit.at(x0)), scale=float(scale))


def predictive(fit: LineFit, x0: float):
    """Posterior predictive distribution of one new outcome at ``x0``.

    Parameters
    ----------
    fit : LineFit
        From :func:`fit_line`.
    x0 : float
        Value of the predictor.

    Returns
    -------
    scipy.stats.rv_continuous_frozen
        Scaled Student-t with ``n - 2`` degrees of freedom, identical to the
        textbook prediction interval's distribution.
    """
    scale = fit.residual_sd * np.sqrt(
        1.0 + 1.0 / fit.n + (x0 - fit.x_mean) ** 2 / fit.sxx
    )
    return stats.t(df=fit.n - 2, loc=float(fit.at(x0)), scale=float(scale))


def log_bayes_factor_slope(
    fit: LineFit, scale: float = normal_t.DEFAULT_CAUCHY_SCALE
) -> float:
    """Log Zellner-Siow Bayes factor for "the slope is not zero".

    Parameters
    ----------
    fit : LineFit
        From :func:`fit_line`.
    scale : float, default 0.707
        Width of the Cauchy prior on the standardised slope. R's
        ``BayesFactor::regressionBF`` defaults to ``sqrt(2) / 4`` (about
        0.354); pass that to reproduce it.

    Returns
    -------
    float
        ``log(BF10)``.
    """
    return normal_t.log_bayes_factor_ttest(
        fit.t, n_effective=fit.n, df=fit.n - 2, scale=scale
    )
