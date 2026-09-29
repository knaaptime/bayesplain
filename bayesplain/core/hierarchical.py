r"""Closed-form partial pooling for a set of group means.

When several groups are compared at once, the smallest one usually looks the
most extreme -- not because it is, but because a small sample has room to
wander. Partial pooling corrects for that by pulling each group's estimate
toward the overall average, and pulling harder when the group's own estimate is
noisier.

The pull is not a fudge. It is what follows from taking seriously the idea that
the groups are all drawn from some common population: a district's observed
average is evidence both about that district and about what districts in
general look like, and the second piece of evidence should inform the first.

This module estimates the between-group spread :math:`\tau^2` with the
DerSimonian-Laird estimator, which is closed form, and then shrinks:

.. math::

    w_i = \frac{\tau^2}{\tau^2 + \mathrm{se}_i^2}, \qquad
    \tilde{\mu}_i = w_i \bar{x}_i + (1 - w_i)\hat{\mu}

A group measured precisely (:math:`\mathrm{se}_i` small) keeps almost all of
its own estimate; a group measured poorly is pulled most of the way to the
overall average. When the groups genuinely differ a lot, :math:`\tau^2` is
large, every :math:`w_i` approaches 1, and pooling does nothing -- which is the
correct behaviour, not a failure to shrink.

No sampler, no hierarchy of priors to specify. An empirical-Bayes shortcut,
and honest about being one: it treats the estimated :math:`\tau^2` as known,
which understates uncertainty slightly when the number of groups is small.

Proportions
-----------
For rates the same idea runs through a Beta population distribution instead
of a normal one. The group rates are treated as draws from
:math:`\mathrm{Beta}(\mu\kappa, (1-\mu)\kappa)`, where :math:`\mu` is the
typical rate and :math:`\kappa` says how tightly the groups cluster around
it. Both are estimated by maximising the beta-binomial marginal likelihood --
an optimisation over two numbers, not a sampler -- and each group's posterior
is then the exact conjugate update
:math:`\mathrm{Beta}(\mu\kappa + x_i,\; (1-\mu)\kappa + n_i - x_i)`. Its mean
is :math:`w_i \hat p_i + (1 - w_i)\mu` with :math:`w_i = n_i/(n_i + \kappa)`:
a small group keeps less of its own rate.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, special

__all__ = ["between_group_variance", "shrink", "shrink_proportions"]

#: Past this concentration the groups are indistinguishable from one shared
#: rate, and pooling is reported as complete.
_FULL_POOLING_KAPPA = 1e7


def between_group_variance(means, standard_errors) -> float:
    r"""Estimate the spread between true group means, by DerSimonian-Laird.

    Parameters
    ----------
    means : array_like
        Observed group means.
    standard_errors : array_like
        Standard error of each group mean.

    Returns
    -------
    float
        The estimate of :math:`\tau^2`, floored at zero. A value of exactly
        zero means the observed spread between groups is no larger than their
        individual noise can account for, in which case full pooling is the
        implied answer.
    """
    means = np.asarray(means, dtype=float)
    se = np.asarray(standard_errors, dtype=float)
    if means.shape != se.shape:
        raise ValueError("means and standard_errors must have the same shape.")
    k = means.size
    if k < 2:
        raise ValueError(f"need at least 2 groups, got {k}.")
    if np.any(se <= 0):
        raise ValueError("every standard error must be positive.")

    weights = 1.0 / se**2
    pooled_mean = float((weights * means).sum() / weights.sum())
    q = float((weights * (means - pooled_mean) ** 2).sum())
    denominator = weights.sum() - (weights**2).sum() / weights.sum()
    if denominator <= 0:
        return 0.0
    return float(max(0.0, (q - (k - 1)) / denominator))


def shrink(means, standard_errors):
    r"""Pull group means toward the overall average, in proportion to noise.

    Parameters
    ----------
    means : array_like
        Observed group means.
    standard_errors : array_like
        Standard error of each group mean.

    Returns
    -------
    dict
        With keys ``'means'`` and ``'standard_errors'`` for the shrunk
        estimates, ``'weights'`` for each group's :math:`w_i` (1 means no
        shrinkage, 0 means full pooling), ``'tau2'`` for the estimated
        between-group variance, and ``'grand_mean'``.

    Examples
    --------
    A tiny district with an extreme average gets pulled the furthest:

    >>> out = shrink([10.0, 12.0, 25.0], [1.0, 1.0, 8.0])
    >>> bool(out["weights"][2] < out["weights"][0])
    True
    >>> bool(out["means"][2] < 25.0)
    True
    """
    means = np.asarray(means, dtype=float)
    se = np.asarray(standard_errors, dtype=float)
    tau2 = between_group_variance(means, se)

    precision = 1.0 / (tau2 + se**2)
    grand_mean = float((precision * means).sum() / precision.sum())

    weights = tau2 / (tau2 + se**2)
    shrunk_means = weights * means + (1.0 - weights) * grand_mean
    # Conditional on tau2, the posterior variance of each group mean.
    shrunk_se = np.sqrt(weights * se**2)

    return {
        "means": shrunk_means,
        "standard_errors": shrunk_se,
        "weights": weights,
        "tau2": tau2,
        "grand_mean": grand_mean,
    }


def shrink_proportions(successes, n):
    r"""Pull group rates toward the typical rate, by empirical-Bayes beta-binomial.

    Parameters
    ----------
    successes : array_like
        Success count in each group.
    n : array_like
        Trial count in each group.

    Returns
    -------
    dict
        With keys ``'alpha'`` and ``'beta'`` for each group's posterior Beta
        parameters, ``'weights'`` for each group's :math:`w_i` (1 means no
        shrinkage, 0 means full pooling), ``'kappa'`` for the estimated
        concentration, ``'grand_mean'`` for the typical rate :math:`\mu`, and
        ``'full'`` for whether the groups collapsed onto one shared rate.

    Examples
    --------
    A small group with an extreme rate is pulled the furthest:

    >>> out = shrink_proportions([10, 45, 80, 9], [100, 100, 100, 12])
    >>> bool(out["weights"][3] < out["weights"][0])
    True
    >>> bool(out["alpha"][3] / (out["alpha"][3] + out["beta"][3]) < 9 / 12)
    True
    """
    x = np.asarray(successes, dtype=float)
    t = np.asarray(n, dtype=float)
    if x.shape != t.shape or x.size < 2:
        raise ValueError("need matching success and trial counts for 2+ groups.")

    def negative_log_likelihood(params):
        mu = special.expit(params[0])
        kappa = np.exp(params[1])
        a, b = mu * kappa, (1.0 - mu) * kappa
        return -float((special.betaln(a + x, b + t - x) - special.betaln(a, b)).sum())

    pooled = (x.sum() + 0.5) / (t.sum() + 1.0)
    upper = np.log(_FULL_POOLING_KAPPA)
    best = None
    # A few starting concentrations, because the likelihood in kappa is
    # often flat on one side and a single start can stall there.
    for start in (1.0, 10.0, 100.0, 1e4):
        fit = optimize.minimize(
            negative_log_likelihood,
            x0=[special.logit(pooled), np.log(start)],
            method="L-BFGS-B",
            bounds=[(-30.0, 30.0), (np.log(1e-3), upper)],
        )
        if best is None or fit.fun < best.fun:
            best = fit
    mu = float(special.expit(best.x[0]))
    kappa = float(np.exp(best.x[1]))
    full = best.x[1] >= upper - 1e-6
    return {
        "alpha": mu * kappa + x,
        "beta": (1.0 - mu) * kappa + t - x,
        "weights": t / (t + kappa),
        "kappa": kappa,
        "grand_mean": mu,
        "full": bool(full),
    }
