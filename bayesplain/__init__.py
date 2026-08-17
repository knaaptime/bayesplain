"""Bayesian answers to the questions frequentist tests are usually asked.

``bayesplain`` gives you the Bayesian version of the handful of tests an
introductory statistics course is built around -- proportions, means,
contingency tables, correlation, group comparisons -- and prints the
conventional test alongside it every time.

Three design commitments
------------------------
**Estimation first, Bayes factors second.** The headline output is a posterior,
a credible interval, and the probability of clearing a threshold you name.
``BF = 4.2`` is not a sentence anyone puts in a memo, so the Bayes factor is a
method call rather than a printed default.

**Every result carries its frequentist twin.** The chi-square statistic, the t,
the p-value, the confidence interval -- computed and printed next to the
posterior, with a plain-English note on what each does and does not claim. You
cannot use this package without seeing both numbers for every analysis you run.

**No sampler in the dependency chain.** numpy and scipy, full stop, with
matplotlib optional. Everything is a conjugate posterior, a closed-form Bayes
factor, or a one-dimensional integral. It installs in a Colab cell in seconds,
with no compiler and no convergence warnings, ever.

Getting started
---------------
>>> import bayesplain as bf
>>> res = bf.compare_proportions(
...     successes=[34, 51],
...     n=[220, 240],
...     labels=["District A", "District B"],
... )
>>> print(res.summary())          # doctest: +SKIP
>>> round(res.probability(">", 0), 2)
0.94

See Also
--------
bayesplain.priors : Named priors and what each one assumes.
bayesplain.frequentist : The conventional tests, computed on their own.
bayesplain.core : The mathematics, as pure functions.
"""

from __future__ import annotations

from . import core, datasets, frequentist, priors
from ._config import (
    DEFAULT_DRAWS,
    DEFAULT_SEED,
    get_draws,
    get_seed,
    set_draws,
    set_seed,
)
from ._contingency import SMALL_EFFECT_V, contingency
from ._correlation import correlation
from ._groups import compare_groups
from ._means import compare_means, mean
from ._proportions import compare_proportions, proportion
from .result import BayesFactor, Decision, Result

try:  # pragma: no cover - depends on install method
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("bayesplain")
except (ImportError, PackageNotFoundError):  # pragma: no cover
    __version__ = "0.0.0.dev0"

__all__ = [
    # analyses
    "proportion",
    "compare_proportions",
    "contingency",
    "SMALL_EFFECT_V",
    "mean",
    "compare_means",
    "correlation",
    "compare_groups",
    # result types
    "Result",
    "Decision",
    "BayesFactor",
    # sub-packages
    "priors",
    "frequentist",
    "core",
    "datasets",
    # configuration
    "get_seed",
    "set_seed",
    "get_draws",
    "set_draws",
    "DEFAULT_SEED",
    "DEFAULT_DRAWS",
    "__version__",
]
