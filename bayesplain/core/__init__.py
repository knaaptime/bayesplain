"""Layer 0: the mathematics, as pure functions.

Nothing in this sub-package knows it is being used to teach. No printing, no
objects that carry explanations, no opinions about what should be reported
first. Functions in, arrays and floats out, fully unit-testable in isolation.

Modules
-------
beta_binomial
    Exact Beta posteriors and closed-form marginal likelihoods for
    proportions.
dirichlet_multinomial
    Closed-form Bayes factors against independence for contingency tables,
    plus Dirichlet posterior draws over cell probabilities.
intervals
    Credible intervals, threshold probabilities, Monte Carlo error.
grid
    Grid approximation and inverse-CDF sampling, the general fallback for
    one-parameter posteriors without a conjugate form.
normal_t
    Student-t posteriors for means, and the JZS Bayes factor for a t statistic.
hierarchical
    Closed-form partial pooling for a set of group means.
correlation
    Exact sampling density of a correlation coefficient, with the posterior and
    Bayes factor that follow from it by one-dimensional integration.
"""

from __future__ import annotations

from . import (
    beta_binomial,
    correlation,
    dirichlet_multinomial,
    grid,
    hierarchical,
    intervals,
    normal_t,
)

__all__ = [
    "beta_binomial",
    "correlation",
    "dirichlet_multinomial",
    "grid",
    "hierarchical",
    "intervals",
    "normal_t",
]
