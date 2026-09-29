r"""Gamma-Poisson conjugate analysis for rates of events over an exposure.

A rate is a count divided by how long, or how much, you watched: collisions per
year, filings per thousand households, crashes per million vehicle-miles. With
``y`` events over exposure ``t`` and a ``Gamma(a, b)`` prior on the rate, the
posterior is exact:

.. math::

    \lambda \mid y \sim \mathrm{Gamma}(a + y,\; b + t)

Observing data adds the events to the shape and the exposure to the rate
parameter -- the same "add the counts" update as a Beta prior on a proportion.
With ``b = 0`` the prior is improper but the posterior is not, as long as some
exposure was observed. ``a = 0.5, b = 0`` is Jeffreys' prior, whose
equal-tailed interval sits between the two halves of the exact (Garwood)
confidence interval.

Comparing two rates
-------------------
Under the same shape-``a`` prior on each rate, the ratio of the two rates has
an exact posterior: each ``\lambda_i (b + t_i)`` is ``Gamma(a + y_i, 1)``, and the
ratio of two independent Gammas is a beta-prime variable. So

.. math::

    \frac{\lambda_2}{\lambda_1} \,\Big|\, y \;\sim\;
        \frac{b + t_1}{b + t_2}\,\mathrm{BetaPrime}(a + y_2,\; a + y_1)

in closed form, with nothing to sample.

Bayes factors
-------------
For a proper prior (``b > 0``) the marginal likelihood is a negative binomial
and every Bayes factor is a ratio of Gamma functions. An improper prior cannot
grade evidence -- the arbitrary constant in front of it does not cancel -- so
for ``b = 0`` the Bayes factors here are built on the *ratio* of the rate to
its comparison instead, which is scale-free:

* one rate against a reference ``\lambda_0``: under the alternative,
  ``\lambda / \lambda_0 \sim \mathrm{BetaPrime}(a, a)``, a prior centred (in
  median) on the reference and symmetric on the log scale;
* two rates: conditional on the total number of events, the second group's
  share is binomial, and under the alternative the rate ratio is again
  ``\mathrm{BetaPrime}(a, a)``. This is the Bayesian counterpart of the exact
  conditional test, and like it, it never needs the overall level of the
  rates.

Each of these is a one-dimensional integral, evaluated by quadrature.

Pure functions only. Nothing here knows it is being used to teach.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, special, stats

__all__ = [
    "validate_events",
    "posterior",
    "ratio_posterior",
    "log_marginal_likelihood",
    "log_bayes_factor_point_null",
    "log_bayes_factor_equal_rates",
    "predictive",
]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_events(events: float, exposure: float) -> tuple[int, float]:
    """Check that an events/exposure pair describes something observable.

    Parameters
    ----------
    events : int
        Number of events counted. Must be a non-negative whole number.
    exposure : float
        How much was watched: time, population, distance. Must be positive.

    Returns
    -------
    tuple
        ``(events, exposure)`` as ``(int, float)``.

    Raises
    ------
    ValueError
        With a message that says what is wrong in plain terms.
    """
    try:
        events_f = float(events)
        exposure_f = float(exposure)
    except (TypeError, ValueError) as err:
        raise ValueError(
            f"events and exposure must be numbers, got {events!r} and {exposure!r}."
        ) from err
    if not np.isfinite(events_f) or events_f < 0:
        raise ValueError(f"events must be a non-negative count, got {events}.")
    if events_f != int(events_f):
        raise ValueError(
            f"events must be a whole number of events, got {events}. If you have "
            "a rate rather than a count, multiply it by the exposure first."
        )
    if not np.isfinite(exposure_f) or exposure_f <= 0:
        raise ValueError(
            f"exposure must be positive, got {exposure}. It is how much you "
            "watched: years, households, vehicle-miles."
        )
    return int(events_f), exposure_f


def _validate_prior(a: float, b: float) -> None:
    if not a > 0:
        raise ValueError(f"the prior shape must be positive, got {a}.")
    if not b >= 0:
        raise ValueError(f"the prior exposure must be zero or positive, got {b}.")


# ---------------------------------------------------------------------------
# Posteriors
# ---------------------------------------------------------------------------


def posterior(events: float, exposure: float, a: float = 0.5, b: float = 0.0):
    """Exact Gamma posterior for one rate.

    Parameters
    ----------
    events : int
        Events counted.
    exposure : float
        Exposure over which they were counted.
    a : float, default 0.5
        Prior shape: worth ``a`` events.
    b : float, default 0.0
        Prior rate parameter: worth ``b`` units of exposure. Zero gives an
        improper prior that is still fine for estimation.

    Returns
    -------
    scipy.stats.rv_continuous_frozen
        ``Gamma(a + events, b + exposure)``, as a frozen scipy distribution.

    Examples
    --------
    >>> post = posterior(12, 4.0)
    >>> round(float(post.mean()), 3)
    3.125
    """
    events, exposure = validate_events(events, exposure)
    _validate_prior(a, b)
    return stats.gamma(a=a + events, scale=1.0 / (b + exposure))


def ratio_posterior(
    events1: float,
    exposure1: float,
    events2: float,
    exposure2: float,
    a: float = 0.5,
    b: float = 0.0,
):
    """Exact posterior for the ratio of two rates, group 2 over group 1.

    Parameters
    ----------
    events1, events2 : int
        Events in each group.
    exposure1, exposure2 : float
        Exposure in each group.
    a : float, default 0.5
        Prior shape on each rate.
    b : float, default 0.0
        Prior rate parameter on each rate.

    Returns
    -------
    scipy.stats.rv_continuous_frozen
        ``((b + t1) / (b + t2)) * BetaPrime(a + y2, a + y1)``.

    Examples
    --------
    >>> post = ratio_posterior(10, 2.0, 20, 2.0)
    >>> round(float(post.median()), 2)
    1.98
    """
    y1, t1 = validate_events(events1, exposure1)
    y2, t2 = validate_events(events2, exposure2)
    _validate_prior(a, b)
    return stats.betaprime(a + y2, a + y1, scale=(b + t1) / (b + t2))


def predictive(
    events: float,
    exposure: float,
    new_exposure: float = 1.0,
    a: float = 0.5,
    b: float = 0.0,
):
    """Posterior predictive distribution of the count over a new exposure.

    A Gamma mixture of Poissons is a negative binomial, so the forecast is
    exact.

    Parameters
    ----------
    events : int
        Events counted so far.
    exposure : float
        Exposure they were counted over.
    new_exposure : float, default 1.0
        Exposure to forecast over, in the same units.
    a, b : float
        Prior shape and rate parameter.

    Returns
    -------
    scipy.stats.rv_discrete_frozen
        Negative binomial over the count of future events.
    """
    events, exposure = validate_events(events, exposure)
    _validate_prior(a, b)
    if not new_exposure > 0:
        raise ValueError(f"new_exposure must be positive, got {new_exposure}.")
    shape = a + events
    rate = b + exposure
    return stats.nbinom(n=shape, p=rate / (rate + new_exposure))


# ---------------------------------------------------------------------------
# Marginal likelihoods and Bayes factors
# ---------------------------------------------------------------------------


def _log_gamma_integral(events: float, exposure: float, a: float, b: float) -> float:
    """Log of the integral of lambda^y exp(-lambda t) against Gamma(a, b)."""
    return float(
        special.gammaln(a + events)
        - special.gammaln(a)
        + a * np.log(b)
        - (a + events) * np.log(b + exposure)
    )


def log_marginal_likelihood(
    events: float, exposure: float, a: float = 1.0, b: float = 1.0
) -> float:
    """Log marginal likelihood of a count under a proper Gamma prior.

    Parameters
    ----------
    events : int
        Events counted.
    exposure : float
        Exposure.
    a, b : float
        Prior shape and rate parameter. ``b`` must be positive: an improper
        prior has no marginal likelihood.

    Returns
    -------
    float
        Log of the negative-binomial probability of the observed count.
    """
    events, exposure = validate_events(events, exposure)
    _validate_prior(a, b)
    if b == 0:
        raise ValueError(
            "an improper prior (b = 0) has no marginal likelihood; give the "
            "prior some exposure."
        )
    return (
        _log_gamma_integral(events, exposure, a, b)
        + events * np.log(exposure)
        - float(special.gammaln(events + 1))
    )


def _log_integral_over_ratio(log_likelihood, a: float) -> float:
    """Log of the integral of exp(log_likelihood(R)) against BetaPrime(a, a).

    Integrated over ``u = log R`` and shifted by the peak so that the
    integrand stays near one, which keeps quadrature stable at large counts.
    """
    prior = stats.betaprime(a, a)

    def log_integrand(u):
        ratio = np.exp(u)
        return log_likelihood(u) + prior.logpdf(ratio) + u

    probe = np.linspace(-60.0, 60.0, 4001)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        values = log_integrand(probe)
    finite = np.isfinite(values)
    peak_index = int(np.argmax(np.where(finite, values, -np.inf)))
    peak = float(values[peak_index])
    centre = float(probe[peak_index])

    def shifted(u):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            value = np.exp(log_integrand(u) - peak)
        return float(value) if np.isfinite(value) else 0.0

    # Split at the peak so quad cannot step over a narrow spike, which is
    # what the likelihood becomes once the counts are large.
    left, _ = integrate.quad(shifted, -np.inf, centre, limit=200)
    right, _ = integrate.quad(shifted, centre, np.inf, limit=200)
    total = left + right
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("the rate Bayes factor integral did not converge.")
    return peak + float(np.log(total))


def log_bayes_factor_point_null(
    events: float,
    exposure: float,
    rate0: float,
    a: float = 0.5,
    b: float = 0.0,
) -> float:
    """Log Bayes factor for "the rate is unknown" against "it equals rate0".

    Parameters
    ----------
    events : int
        Events counted.
    exposure : float
        Exposure.
    rate0 : float
        The reference rate, per unit of exposure.
    a : float, default 0.5
        Prior shape.
    b : float, default 0.0
        Prior rate parameter. When positive, the alternative uses the proper
        ``Gamma(a, b)`` prior directly and the result is closed form. When
        zero, the alternative puts ``BetaPrime(a, a)`` on the ratio of the
        rate to ``rate0``, so the prior is centred on the reference and
        carries no units.

    Returns
    -------
    float
        ``log(BF10)``.
    """
    events, exposure = validate_events(events, exposure)
    _validate_prior(a, b)
    if not rate0 > 0:
        raise ValueError(f"the reference rate must be positive, got {rate0}.")
    expected0 = rate0 * exposure
    # The Poisson log likelihood at the reference, without the y! term, which
    # appears identically on both sides and cancels.
    log_null = events * np.log(expected0) - expected0
    if b > 0:
        return (
            _log_gamma_integral(events, exposure, a, b)
            + events * np.log(exposure)
            - log_null
        )

    def log_likelihood(u):
        mu = expected0 * np.exp(u)
        return events * np.log(mu) - mu - log_null

    return _log_integral_over_ratio(log_likelihood, a)


def log_bayes_factor_equal_rates(
    events1: float,
    exposure1: float,
    events2: float,
    exposure2: float,
    a: float = 0.5,
    b: float = 0.0,
) -> float:
    """Log Bayes factor for "the two rates differ" against "they are equal".

    Parameters
    ----------
    events1, events2 : int
        Events in each group.
    exposure1, exposure2 : float
        Exposure in each group.
    a : float, default 0.5
        Prior shape on each rate.
    b : float, default 0.0
        Prior rate parameter. When positive, both models use proper
        ``Gamma(a, b)`` priors and the result is a ratio of Gamma functions.
        When zero, the comparison conditions on the total count -- the second
        group's share of events is binomial -- and the alternative puts
        ``BetaPrime(a, a)`` on the rate ratio.

    Returns
    -------
    float
        ``log(BF10)``.

    Examples
    --------
    With equal exposures and ``b = 0``, the conditional share has a
    ``Beta(a, a)`` prior, so this reproduces the Beta-binomial point-null
    Bayes factor at one half:

    >>> from bayesplain.core import beta_binomial
    >>> a = log_bayes_factor_equal_rates(18, 3.0, 35, 3.0, a=1.0)
    >>> b = beta_binomial.log_bayes_factor_point_null(35, 53, 0.5, 1.0, 1.0)
    >>> bool(abs(a - b) < 1e-6)
    True
    """
    y1, t1 = validate_events(events1, exposure1)
    y2, t2 = validate_events(events2, exposure2)
    _validate_prior(a, b)
    if b > 0:
        return (
            _log_gamma_integral(y1, t1, a, b)
            + _log_gamma_integral(y2, t2, a, b)
            - _log_gamma_integral(y1 + y2, t1 + t2, a, b)
        )
    if y1 + y2 == 0:
        raise ValueError(
            "no events in either group, so there is nothing to compare; the "
            "Bayes factor is exactly 1."
        )
    # Share of events expected in group 2 if the rates were equal.
    log_c = np.log(t2 / t1)
    log_p0 = log_c - np.logaddexp(0.0, log_c)
    log_q0 = -np.logaddexp(0.0, log_c)
    log_null = y2 * log_p0 + y1 * log_q0

    def log_likelihood(u):
        z = u + log_c
        log_p = -np.logaddexp(0.0, -z)
        log_q = -np.logaddexp(0.0, z)
        return y2 * log_p + y1 * log_q - log_null

    return _log_integral_over_ratio(log_likelihood, a)
