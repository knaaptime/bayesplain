"""The frequentist twin of every analysis, computed and printed alongside it.

Design pillar two: no result in this package appears without the conventional
test that would ordinarily have been run on the same data, so that nobody
finishes the course able to produce a posterior but unable to read a p-value in
someone else's report.

Each twin carries not just its numbers but a statement of what it claims and
what it does not, generated from the numbers themselves. The claim text is the
teaching payload, and putting it next to the Bayesian answer is what turns the
"which one is right" question into a "what did you actually ask" question.

Requires scipy and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = [
    "FrequentistTwin",
    "one_proportion",
    "two_proportions",
    "chi_square_independence",
    "one_mean",
    "two_means",
    "correlation",
    "one_way_anova",
]


# ---------------------------------------------------------------------------
# The twin container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrequentistTwin:
    """A conventional test result, with its interpretation attached.

    Attributes
    ----------
    test : str
        Name of the test, as a student would meet it in a methods section.
    statistic_name : str
        Symbol for the test statistic, e.g. ``"chi-square"`` or ``"z"``.
    statistic : float
        The test statistic.
    pvalue : float
        The p-value.
    dof : float or None
        Degrees of freedom, where the test has them.
    estimate : float or None
        The point estimate the test is built around.
    interval : tuple of float or None
        Confidence interval on ``estimate``.
    interval_level : float
        Coverage of ``interval``.
    interval_method : str
        How the interval was constructed, since the choice is not unique.
    null_statement : str
        The null hypothesis in words.
    """

    test: str
    statistic_name: str
    statistic: float
    pvalue: float
    dof: float | None = None
    estimate: float | None = None
    interval: tuple[float, float] | None = None
    interval_level: float = 0.95
    interval_method: str = ""
    null_statement: str = ""

    @property
    def significant(self) -> bool:
        """Whether the p-value clears the conventional 0.05 threshold."""
        return bool(self.pvalue < 0.05)

    def claims(self) -> str:
        """State, in words, what this p-value does claim.

        Returns
        -------
        str
            A sentence built from this result's own numbers.
        """
        null = self.null_statement or "the null hypothesis were true"
        return (
            f"If {null}, data at least this extreme would turn up "
            f"{self.rate_in_words()}."
        )

    def rate_in_words(self) -> str:
        """Express the p-value as a frequency, without rounding it to zero.

        Returns
        -------
        str
            A phrase such as ``'11.0% of the time'``, or ``'fewer than 1 time
            in 1,000'`` when a percentage would round away the whole number.
        """
        if self.pvalue >= 0.001:
            return f"{self.pvalue:.1%} of the time"
        if self.pvalue < 1e-6:
            return "less than once in a million studies"
        return f"fewer than 1 time in {round(1.0 / self.pvalue):,}"

    def disclaims(self) -> str:
        """State what this p-value is routinely but wrongly taken to claim.

        Returns
        -------
        str
            A sentence naming the misreading.
        """
        shown = f"{self.pvalue:.3f}" if self.pvalue >= 0.001 else f"{self.pvalue:.2g}"
        return (
            f"It is not the probability that the null is true, not the "
            f"probability this finding is wrong, and it says nothing about how "
            f"large the effect is. p = {shown} is a statement about "
            f"data given a hypothesis, not about a hypothesis given data."
        )

    def interval_claims(self, formatter=None) -> str:
        """State what the confidence interval does and does not claim.

        Parameters
        ----------
        formatter : callable, optional
            ``formatter(low, high) -> str``, used to render the bounds in
            whatever display units the caller reports in. Defaults to raw
            values in the quantity's own units.

        Returns
        -------
        str
            Two sentences: the correct reading, then the tempting one.
        """
        if self.interval is None:
            return ""
        lo, hi = self.interval
        pct = f"{self.interval_level:.0%}"
        span = formatter(lo, hi) if formatter else f"{lo:.4g} to {hi:.4g}"
        return (
            f"Across many repetitions of this study, {pct} of intervals built "
            f"this way would contain the true value. It does not say there is "
            f"a {pct} chance the true value lies in the range {span} — that "
            f"sentence is only available for a credible interval."
        )


# ---------------------------------------------------------------------------
# Proportions
# ---------------------------------------------------------------------------


def one_proportion(
    successes: float, n: float, p0: float = 0.5, level: float = 0.95
) -> FrequentistTwin:
    """Exact binomial test and Wilson score interval for one proportion.

    Uses the exact binomial test rather than the normal approximation, and the
    Wilson score interval rather than the Wald interval, because both behave
    sensibly at the small samples and extreme rates that real planning data
    routinely produce.

    Parameters
    ----------
    successes : int
        Number of successes observed.
    n : int
        Number of trials.
    p0 : float, default 0.5
        Null value for the rate.
    level : float, default 0.95
        Confidence level for the interval.

    Returns
    -------
    FrequentistTwin
        The conventional result for this data.
    """
    successes, n = int(successes), int(n)
    test = stats.binomtest(successes, n, p0)
    ci = test.proportion_ci(confidence_level=level, method="wilson")
    phat = successes / n
    # z is reported for orientation; the p-value above is exact, not from z.
    se_null = np.sqrt(p0 * (1.0 - p0) / n)
    z = (phat - p0) / se_null if se_null > 0 else np.nan

    return FrequentistTwin(
        test="exact binomial test",
        statistic_name="z (for reference)",
        statistic=float(z),
        pvalue=float(test.pvalue),
        dof=None,
        estimate=float(phat),
        interval=(float(ci.low), float(ci.high)),
        interval_level=level,
        interval_method="Wilson score",
        null_statement=f"the true rate were exactly {p0:.1%}",
    )


def two_proportions(successes, n, level: float = 0.95) -> FrequentistTwin:
    """Two-proportion chi-square test and Wald interval on the difference.

    The chi-square statistic here is computed without a continuity correction,
    so it equals the square of the two-proportion z statistic and the p-values
    agree exactly -- worth saying out loud, since students meet the same test
    under both names.

    The interval is the plain Wald interval on the difference, because that is
    the one reproduced in textbooks and in the reports students will read. It
    is known to under-cover for small samples or rates near 0 and 1, which
    makes for a useful comparison against the credible interval in exactly
    those cases.

    Parameters
    ----------
    successes : array_like
        Two success counts, ``[x1, x2]``.
    n : array_like
        Two trial counts, ``[n1, n2]``.
    level : float, default 0.95
        Confidence level for the interval.

    Returns
    -------
    FrequentistTwin
        The conventional result, oriented as group 2 minus group 1.
    """
    x1, x2 = (int(v) for v in successes)
    n1, n2 = (int(v) for v in n)
    p1, p2 = x1 / n1, x2 / n2
    diff = p2 - p1

    pooled = (x1 + x2) / (n1 + n2)
    se_null = np.sqrt(pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2))
    if se_null > 0:
        z = diff / se_null
        pvalue = 2.0 * stats.norm.sf(abs(z))
    else:
        z, pvalue = np.nan, np.nan

    se_diff = np.sqrt(p1 * (1.0 - p1) / n1 + p2 * (1.0 - p2) / n2)
    crit = stats.norm.ppf(0.5 + level / 2.0)
    interval = (float(diff - crit * se_diff), float(diff + crit * se_diff))

    return FrequentistTwin(
        test="two-proportion z-test (equivalently chi-square, 1 df)",
        statistic_name="chi-square",
        statistic=float(z**2),
        pvalue=float(pvalue),
        dof=1.0,
        estimate=float(diff),
        interval=interval,
        interval_level=level,
        interval_method="Wald",
        null_statement="the two rates were identical",
    )


def one_mean(x, mu0: float = 0.0, level: float = 0.95) -> FrequentistTwin:
    """One-sample t-test and its confidence interval.

    Parameters
    ----------
    x : array_like
        Observations.
    mu0 : float, default 0.0
        Null value for the mean.
    level : float, default 0.95
        Confidence level.

    Returns
    -------
    FrequentistTwin
        The conventional result.

    Notes
    -----
    This interval will coincide, digit for digit, with the equal-tailed
    credible interval from the reference-prior posterior. That is not a
    coincidence to explain away: the arithmetic is the same, and only the
    licensed interpretation differs.
    """
    arr = np.asarray(x, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    n = arr.size
    mean, sd = arr.mean(), arr.std(ddof=1)
    se = sd / np.sqrt(n)
    df = n - 1
    t = (mean - mu0) / se
    pvalue = 2.0 * stats.t.sf(abs(t), df)
    crit = stats.t.ppf(0.5 + level / 2.0, df)

    return FrequentistTwin(
        test="one-sample t-test",
        statistic_name="t",
        statistic=float(t),
        pvalue=float(pvalue),
        dof=float(df),
        estimate=float(mean),
        interval=(float(mean - crit * se), float(mean + crit * se)),
        interval_level=level,
        interval_method="Student t",
        null_statement=f"the true mean were exactly {mu0:g}",
    )


def two_means(x, y, equal_var: bool = False, level: float = 0.95) -> FrequentistTwin:
    """Two-sample t-test and confidence interval on the difference.

    Defaults to Welch's unequal-variance test, which is the honest default and
    the one that matches the Behrens-Fisher posterior on the Bayesian side.

    Parameters
    ----------
    x, y : array_like
        The two samples. The difference is oriented as ``y - x``.
    equal_var : bool, default False
        Assume equal variances and pool them (Student's t) rather than using
        Welch's correction.
    level : float, default 0.95
        Confidence level.

    Returns
    -------
    FrequentistTwin
        The conventional result, oriented as group 2 minus group 1.
    """
    a = np.asarray(x, dtype=float).ravel()
    b = np.asarray(y, dtype=float).ravel()
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    n1, n2 = a.size, b.size
    m1, m2 = a.mean(), b.mean()
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    diff = m2 - m1

    if equal_var:
        df = n1 + n2 - 2
        pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
        se = np.sqrt(pooled * (1.0 / n1 + 1.0 / n2))
        name = "two-sample t-test (equal variances pooled)"
    else:
        term1, term2 = v1 / n1, v2 / n2
        se = np.sqrt(term1 + term2)
        df = (term1 + term2) ** 2 / (term1**2 / (n1 - 1) + term2**2 / (n2 - 1))
        name = "Welch's two-sample t-test"

    t = diff / se
    pvalue = 2.0 * stats.t.sf(abs(t), df)
    crit = stats.t.ppf(0.5 + level / 2.0, df)

    return FrequentistTwin(
        test=name,
        statistic_name="t",
        statistic=float(t),
        pvalue=float(pvalue),
        dof=float(df),
        estimate=float(diff),
        interval=(float(diff - crit * se), float(diff + crit * se)),
        interval_level=level,
        interval_method="Welch" if not equal_var else "pooled Student t",
        null_statement="the two group means were identical",
    )


def correlation(x, y, level: float = 0.95) -> FrequentistTwin:
    """Pearson correlation with its p-value and Fisher-z interval.

    Parameters
    ----------
    x, y : array_like
        Paired observations.
    level : float, default 0.95
        Confidence level.

    Returns
    -------
    FrequentistTwin
        The conventional result.
    """
    a = np.asarray(x, dtype=float).ravel()
    b = np.asarray(y, dtype=float).ravel()
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    n = a.size
    r, pvalue = stats.pearsonr(a, b)

    # Fisher's z transform, the standard route to an interval on r.
    if n > 3 and abs(r) < 1.0:
        z = np.arctanh(r)
        se = 1.0 / np.sqrt(n - 3)
        crit = stats.norm.ppf(0.5 + level / 2.0)
        interval = (
            float(np.tanh(z - crit * se)),
            float(np.tanh(z + crit * se)),
        )
    else:
        interval = None

    return FrequentistTwin(
        test="Pearson correlation test",
        statistic_name="r",
        statistic=float(r),
        pvalue=float(pvalue),
        dof=float(n - 2),
        estimate=float(r),
        interval=interval,
        interval_level=level,
        interval_method="Fisher z",
        null_statement="the two variables were truly unrelated",
    )


def one_way_anova(groups) -> FrequentistTwin:
    """One-way ANOVA F test across three or more groups.

    Parameters
    ----------
    groups : sequence of array_like
        The samples, one per group.

    Returns
    -------
    FrequentistTwin
        The omnibus result. ``estimate`` carries eta-squared, the share of
        variance explained, as a point estimate with no uncertainty attached.

    Notes
    -----
    The omnibus question -- "is there a difference somewhere?" -- is rarely the
    one a planner needs answered, which is why the Bayesian side of this
    package leads with pairwise comparisons instead.
    """
    samples = [np.asarray(g, dtype=float).ravel() for g in groups]
    samples = [g[np.isfinite(g)] for g in samples]
    f_stat, pvalue = stats.f_oneway(*samples)

    grand = np.concatenate(samples)
    grand_mean = grand.mean()
    ss_between = sum(g.size * (g.mean() - grand_mean) ** 2 for g in samples)
    ss_total = ((grand - grand_mean) ** 2).sum()
    eta_squared = float(ss_between / ss_total) if ss_total > 0 else np.nan

    k = len(samples)
    return FrequentistTwin(
        test="one-way ANOVA",
        statistic_name="F",
        statistic=float(f_stat),
        pvalue=float(pvalue),
        dof=float(k - 1),
        estimate=eta_squared,
        interval=None,
        interval_level=0.95,
        interval_method="none reported for eta-squared",
        null_statement="every group had the same mean",
    )


def chi_square_independence(table, correction: bool = False) -> FrequentistTwin:
    """Pearson chi-square test of independence for an r x c table.

    Parameters
    ----------
    table : array_like
        Table of counts.
    correction : bool, default False
        Apply Yates' continuity correction. Off by default so that the 2x2
        statistic matches the square of the two-proportion z.

    Returns
    -------
    FrequentistTwin
        The conventional result. ``estimate`` carries the sample Cramer's V,
        as a point estimate with no uncertainty attached -- which is precisely
        the gap the Bayesian version fills.
    """
    counts = np.asarray(table, dtype=float)
    chi2, pvalue, dof, expected = stats.chi2_contingency(counts, correction=correction)
    total = counts.sum()
    min_dim = min(counts.shape) - 1
    v = float(np.sqrt(chi2 / (total * min_dim))) if min_dim > 0 else np.nan

    small = int((expected < 5).sum())
    note = ""
    if small:
        note = (
            f" Note: {small} cell(s) have expected counts below 5, where this "
            "test's approximation is unreliable."
        )

    return FrequentistTwin(
        test="Pearson chi-square test of independence",
        statistic_name="chi-square",
        statistic=float(chi2),
        pvalue=float(pvalue),
        dof=float(dof),
        estimate=v,
        interval=None,
        interval_method="none available for Cramer's V" + note,
        null_statement="rows and columns were unrelated",
    )
