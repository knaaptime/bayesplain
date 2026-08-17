"""The object every analysis returns.

One class for all seven analyses, so that the methods a student learns in week
2 are the same methods they use in week 9. The muscle memory transfers; only
the question changes.

The ordering of the printed output encodes the package's main opinion. A
posterior, a credible interval, and a probability of clearing a threshold come
first, because those are the things a memo can be written from. The Bayes
factor is a method call rather than a printed default, because "BF = 4.2" is
not a sentence anyone puts in a memo, and because it is prior-sensitive in a
way that is genuinely hard to explain honestly to a non-technical audience.

Nothing here is specific to proportions -- the seven analysis functions all
assemble one of these.
"""

from __future__ import annotations

import textwrap
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .core import intervals
from .frequentist import FrequentistTwin

__all__ = ["Result", "Decision", "BayesFactor"]

_RULE = "=" * 74
_THIN = "-" * 74

#: Units that attach directly to the number rather than after a space.
_TIGHT_UNITS = frozenset({"%", "×", "°"})


def _join_unit(text: str, unit: str) -> str:
    """Attach a unit to a formatted number with the right spacing."""
    if not unit:
        return text
    return f"{text}{unit}" if unit in _TIGHT_UNITS else f"{text} {unit}"


def _fmt_dof(dof: float) -> str:
    """Format degrees of freedom, which Welch's correction makes fractional."""
    return f"{dof:g}" if float(dof).is_integer() else f"{dof:.1f}"


# ---------------------------------------------------------------------------
# Small printable helpers
# ---------------------------------------------------------------------------


class _Report:
    """Multi-line text that prints as itself in a terminal and in Jupyter."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return self.text


def _wrap(text: str, prefix: str = "", width: int = 74) -> list[str]:
    """Wrap prose to a fixed width, indenting continuations under ``prefix``.

    Parameters
    ----------
    text : str
        Prose to wrap.
    prefix : str
        Prepended to the first line; continuations are indented to match its
        length.
    width : int, default 74
        Total line width including the prefix.

    Returns
    -------
    list of str
        The wrapped lines.
    """
    body = textwrap.wrap(text, width=max(20, width - len(prefix)))
    if not body:
        return [prefix.rstrip()]
    pad = " " * len(prefix)
    return [prefix + body[0]] + [pad + line for line in body[1:]]


@dataclass(frozen=True)
class Decision:
    """The outcome of comparing a credible interval against a ROPE.

    "ROPE" is a region of practical equivalence: a range of values you have
    decided in advance are too small to act on. Comparing the credible
    interval against it replaces "reject the null" with a decision rule that
    refers to consequences rather than to sampling distributions.

    Attributes
    ----------
    verdict : str
        One of ``'practically different'``, ``'practically equivalent'``, or
        ``'too uncertain to call'``.
    rope : tuple of float
        The region of practical equivalence that was supplied.
    interval : tuple of float
        The credible interval it was compared against.
    probability_inside : float
        Posterior probability that the quantity falls inside the ROPE.
    explanation : str
        Plain-English account of the verdict.
    """

    verdict: str
    rope: tuple[float, float]
    interval: tuple[float, float]
    probability_inside: float
    explanation: str

    def __repr__(self) -> str:
        return "\n".join(_wrap(self.explanation, prefix=f"{self.verdict.upper()} — "))


@dataclass(frozen=True)
class BayesFactor:
    """A Bayes factor, with the caveat it should never be quoted without.

    Attributes
    ----------
    bf10 : float
        Evidence for the alternative relative to the null.
    bf01 : float
        Its reciprocal.
    log_bf10 : float
        Log of ``bf10``, which is what is actually computed.
    interpretation : str
        Conventional verbal grade for the magnitude.
    caveat : str
        What the number depends on besides the data.
    """

    bf10: float
    bf01: float
    log_bf10: float
    interpretation: str
    caveat: str

    def __repr__(self) -> str:
        return (
            f"BF10 = {self.bf10:.3g}  ({self.interpretation})\n"
            f"BF01 = {self.bf01:.3g}\n{self.caveat}"
        )


def _grade_bayes_factor(bf10: float, alternative: str = "a difference") -> str:
    """Label a Bayes factor's magnitude on Jeffreys' conventional scale.

    The scale is a descriptive simplification of a continuous quantity, and
    the boundaries are conventions rather than results. Its main use is to
    stop anyone getting excited about a Bayes factor between 1/3 and 3.

    Parameters
    ----------
    bf10 : float
        Evidence for the alternative relative to the null.
    alternative : str, default 'a difference'
        What the alternative hypothesis asserts, so the label reads correctly
        for the analysis at hand -- 'an association' for a table, 'a
        relationship' for a correlation.

    Returns
    -------
    str
        A phrase such as ``'moderate evidence against a difference'``.
    """
    bf = bf10 if bf10 >= 1.0 else 1.0 / bf10
    direction = "for" if bf10 >= 1.0 else "against"
    if bf < 3:
        strength = "not worth more than a bare mention"
    elif bf < 10:
        strength = "moderate"
    elif bf < 30:
        strength = "strong"
    elif bf < 100:
        strength = "very strong"
    else:
        strength = "extreme"
    return f"{strength} evidence {direction} {alternative}"


# ---------------------------------------------------------------------------
# The Result
# ---------------------------------------------------------------------------


class Result:
    """What every ``bayesplain`` analysis hands back.

    Parameters
    ----------
    quantity : str
        Name of the estimated quantity, as it should appear in output, e.g.
        ``"difference in filing rate (District B - District A)"``.
    draws : ndarray
        Posterior draws of ``quantity``. Always present, so that a single
        plotting routine and a single set of summary methods work for every
        analysis in the package.
    posterior : scipy.stats.rv_continuous_frozen, optional
        The analytic posterior, when one exists in closed form. When present,
        intervals and probabilities are read from it rather than estimated,
        and carry no Monte Carlo error.
    prior : object, optional
        The resolved prior, expected to expose ``.label`` and ``.dist()``.
    subject : str, optional
        How to refer to the quantity inside a running sentence, when the
        header form does not read well there. ``quantity`` might be
        ``"rate (34 of 220)"`` while ``subject`` is ``"filing rate"``.
        Defaults to ``quantity``.
    analysis : str
        Name of the analysis that produced this, for the summary header.
    frequentist : FrequentistTwin, optional
        The conventional test on the same data.
    log_bf10 : float, optional
        Log Bayes factor, if the analysis defines one.
    bf_caveat : str
        What the Bayes factor depends on besides the data.
    bf_alternative : str, default 'a difference'
        What the alternative hypothesis asserts, so the Bayes factor's verbal
        grade reads correctly -- 'an association' for a table, 'a
        relationship' for a correlation.
    unit : str
        Display unit, e.g. ``"percentage points"``.
    display_scale : float, default 1.0
        Multiplier applied for display only. Use 100 to show a difference in
        proportions as percentage points. Thresholds passed to
        :meth:`probability` and :meth:`decide` are always in the *unscaled*
        units of the quantity itself.
    decimals : int, optional
        Decimal places for display. Defaults to 1 when ``display_scale`` is
        100, otherwise 3.
    direction_reference : float, default 0.0
        The value that :meth:`probability` compares against by default, and
        the one the "probability of direction" line in the summary reports.
    higher_label, lower_label : str, optional
        Group names used to build a plain-English sentence for comparisons.
    components : mapping, optional
        Per-group posteriors or draws, for plotting and for the pairwise
        views used by ``compare_groups``.
    component_axis : str, optional
        Axis label for the components plot. Components live on a different
        scale from the headline quantity -- each group's own rate rather than
        the difference between them -- so they need their own label.
    component_scale : float, optional
        Display multiplier for the components plot. Defaults to
        ``display_scale``, which is wrong whenever the estimand is a ratio of
        quantities rather than the quantities themselves.
    custom_plots : mapping, optional
        Extra plot kinds this analysis offers, as ``{name: draw(ax)}``. Lets a
        correlation expose a scatter plot without every other analysis
        carrying the concept.
    no_sensitivity_reason : str, optional
        Why :meth:`sensitivity` is unavailable, when it is. Saying why is more
        useful than a bare "not implemented", because usually the reason is
        that there is no prior to vary rather than that nobody wrote the code.
    no_bayes_factor_reason : str, optional
        Why :meth:`bayes_factor` is unavailable, when it is.
    next_steps : str, optional
        The suggestions printed at the foot of a summary. Defaults to the
        methods every result supports.
    refit : callable, optional
        ``refit(prior) -> Result``, used by :meth:`sensitivity` to recompute
        the same analysis under a different assumption.
    prior_ladder : sequence of str, optional
        Prior names to walk through in :meth:`sensitivity`.
    n_draws : int, optional
        Number of draws taken; inferred from ``draws`` when omitted.
    notes : sequence of str, optional
        Warnings or caveats specific to this data, e.g. small cell counts.
    """

    def __init__(
        self,
        *,
        quantity: str,
        draws: np.ndarray,
        posterior: Any = None,
        prior: Any = None,
        subject: str = "",
        analysis: str = "",
        frequentist: FrequentistTwin | None = None,
        log_bf10: float | None = None,
        bf_caveat: str = "",
        bf_alternative: str = "a difference",
        unit: str = "",
        display_scale: float = 1.0,
        decimals: int | None = None,
        direction_reference: float = 0.0,
        higher_label: str = "",
        lower_label: str = "",
        components: Mapping[str, Any] | None = None,
        component_axis: str = "",
        component_scale: float | None = None,
        custom_plots: Mapping[str, Callable[[Any], Any]] | None = None,
        no_sensitivity_reason: str = "",
        no_bayes_factor_reason: str = "",
        next_steps: str = "",
        refit: Callable[[Any], Result] | None = None,
        prior_ladder: Sequence[str] | None = None,
        n_draws: int | None = None,
        notes: Sequence[str] | None = None,
    ) -> None:
        self.quantity = quantity
        self.draws = np.asarray(draws, dtype=float).ravel()
        self.posterior = posterior
        self.prior = prior
        self.subject = subject or quantity
        self.analysis = analysis
        self.frequentist = frequentist
        self.log_bf10 = log_bf10
        self.bf_caveat = bf_caveat
        self.bf_alternative = bf_alternative
        self.unit = unit
        self.display_scale = float(display_scale)
        self.decimals = (
            decimals if decimals is not None else (1 if display_scale == 100 else 3)
        )
        self.direction_reference = float(direction_reference)
        self.higher_label = higher_label
        self.lower_label = lower_label
        self.components = dict(components) if components else {}
        self.component_axis = component_axis
        self.component_scale = component_scale
        self.custom_plots = dict(custom_plots) if custom_plots else {}
        self.no_sensitivity_reason = no_sensitivity_reason
        self.no_bayes_factor_reason = no_bayes_factor_reason
        self.next_steps = next_steps
        self._refit = refit
        self.prior_ladder = tuple(prior_ladder) if prior_ladder else ()
        self.n_draws = int(n_draws) if n_draws is not None else int(self.draws.size)
        self.notes = tuple(notes) if notes else ()

    # -- basic summaries ----------------------------------------------------

    @property
    def is_comparison(self) -> bool:
        """Whether this result estimates a difference between groups."""
        return bool(self.higher_label and self.lower_label)

    @property
    def exact(self) -> bool:
        """Whether an analytic posterior is available for this quantity."""
        return self.posterior is not None

    def point(self, kind: str = "median") -> float:
        """Return a single representative value from the posterior.

        Parameters
        ----------
        kind : {'median', 'mean'}, default 'median'
            The median is the default because it is stable under the skew that
            rate posteriors routinely have.

        Returns
        -------
        float
            The requested summary, taken analytically when possible.
        """
        if kind not in {"median", "mean"}:
            raise ValueError(f"kind must be 'median' or 'mean', got {kind!r}.")
        if self.exact:
            return float(
                self.posterior.median() if kind == "median" else self.posterior.mean()
            )
        return float(np.median(self.draws) if kind == "median" else np.mean(self.draws))

    def interval(self, level: float = 0.95, kind: str = "hdi") -> tuple[float, float]:
        """Credible interval for the quantity.

        Unlike a confidence interval, this one supports the sentence people
        already want to say: given the data and the stated prior, there is a
        ``level`` probability the quantity lies in this range.

        Parameters
        ----------
        level : float, default 0.95
            Probability the interval should contain.
        kind : {'hdi', 'eti'}, default 'hdi'
            Highest-density (shortest) or equal-tailed. Use ``'eti'`` when
            lining the interval up against a confidence interval.

        Returns
        -------
        tuple of float
            Lower and upper bounds, in the quantity's own units.
        """
        return intervals.interval(
            draws=self.draws, level=level, kind=kind, dist=self.posterior
        )

    def probability(self, op: str = ">", value: float | None = None) -> float:
        """Posterior probability that the quantity satisfies a comparison.

        This is the method that answers the question a planner actually has:
        not "is there an effect" but "how likely is it that this program cuts
        filings by at least ten percent".

        Parameters
        ----------
        op : {'>', '>=', '<', '<=', 'between', 'outside'}, default '>'
            Comparison to evaluate.
        value : float or tuple of float, optional
            Threshold in the quantity's own units -- not the display units. A
            difference in proportions displayed as "5.7 percentage points" is
            compared against ``0.057``. Defaults to
            ``direction_reference`` (0 for comparisons).

        Returns
        -------
        float
            Posterior probability, computed analytically when an exact
            posterior is available.

        Examples
        --------
        >>> import bayesplain as bf
        >>> res = bf.compare_proportions([34, 51], [220, 240])
        >>> round(res.probability(">", 0), 2)
        0.94
        """
        if value is None:
            value = self.direction_reference

        if self.exact and op in {">", ">=", "<", "<="}:
            thresh = float(value)
            if op in {">", ">="}:
                return float(self.posterior.sf(thresh))
            return float(self.posterior.cdf(thresh))
        if self.exact and op in {"between", "outside"}:
            low, high = sorted(float(v) for v in value)
            inside = float(self.posterior.cdf(high) - self.posterior.cdf(low))
            return inside if op == "between" else 1.0 - inside
        return intervals.probability_from_draws(self.draws, op, value)

    def probability_of_direction(self) -> float:
        """Probability the quantity is on the side of zero the data suggest.

        Returns
        -------
        float
            A value between 0.5 and 1. Read as confidence in the *sign* of the
            effect, separately from its size.
        """
        above = self.probability(">", self.direction_reference)
        return float(max(above, 1.0 - above))

    def monte_carlo_error(self) -> float:
        """Monte Carlo standard error of the posterior mean.

        Reported rather than hidden. Because the package fixes a default seed
        so that a whole classroom sees identical digits, the noise that seed
        conceals should be stated explicitly somewhere -- this is where.

        Returns
        -------
        float
            Standard error of the mean, or 0.0 when the answer is analytic.
        """
        if self.exact:
            return 0.0
        return intervals.monte_carlo_se(self.draws)

    # -- decisions ---------------------------------------------------------

    def decide(
        self, rope: tuple[float, float], level: float = 0.95, kind: str = "hdi"
    ) -> Decision:
        """Compare the credible interval against a region of practical equivalence.

        The replacement for "reject the null". You name a range of values too
        small to matter for the decision at hand -- on policy grounds, not
        statistical ones -- and the verdict follows from whether the credible
        interval sits inside it, outside it, or straddles its edge.

        Parameters
        ----------
        rope : tuple of float
            ``(low, high)`` bounds of the region of practical equivalence, in
            the quantity's own units.
        level : float, default 0.95
            Credible level for the comparison.
        kind : {'hdi', 'eti'}, default 'hdi'
            Interval type.

        Returns
        -------
        Decision
            The verdict and its explanation.
        """
        low, high = sorted(float(v) for v in rope)
        lo, hi = self.interval(level=level, kind=kind)
        inside = self.probability("between", (low, high))
        pct = f"{level:.0%}"
        span = self._fmt_range(low, high)

        if lo >= low and hi <= high:
            verdict = "practically equivalent"
            explanation = (
                f"the whole {pct} credible interval sits inside the range you "
                f"called too small to act on ({span}), so the data support "
                f"treating this as no practical difference"
            )
        elif hi < low or lo > high:
            verdict = "practically different"
            explanation = (
                f"the whole {pct} credible interval sits outside the range you "
                f"called too small to act on ({span}), so the difference is "
                f"both real and big enough to matter"
            )
        else:
            verdict = "too uncertain to call"
            explanation = (
                f"the {pct} credible interval straddles the edge of the range "
                f"you called too small to act on ({span}); the honest answer is "
                f"that this data cannot settle the decision, and the useful "
                f"next question is how much more data would"
            )
        return Decision(verdict, (low, high), (lo, hi), inside, explanation)

    # -- language ----------------------------------------------------------

    def sentence(self, level: float = 0.95) -> str:
        """One plain-English line, ready to paste into a memo.

        Parameters
        ----------
        level : float, default 0.95
            Credible level to quote.

        Returns
        -------
        str
            A single sentence stating direction, magnitude, and uncertainty.

        Notes
        -----
        Have students write this sentence themselves before revealing this
        method. Otherwise the convenience quietly becomes the learning
        objective.
        """
        lo, hi = self.interval(level=level)
        pct = f"{level:.0%}"
        point = self._fmt(self.point())
        span = self._fmt_range(lo, hi)

        if self.is_comparison:
            prob = self.probability(">", self.direction_reference)
            if prob >= 0.5:
                higher, lower, p = self.higher_label, self.lower_label, prob
            else:
                higher, lower, p = self.lower_label, self.higher_label, 1.0 - prob
            return (
                f"{higher} is higher than {lower} with {p:.0%} probability; the "
                f"gap is most likely {point}, and the data are consistent with "
                f"anything from {span} ({pct} credible interval)."
            )
        return (
            f"The {self.subject} is most likely {point}, and the data are "
            f"consistent with anything from {span} ({pct} credible interval)."
        )

    def translate(self, level: float = 0.95) -> _Report:
        """Set the two frameworks' claims side by side, in words.

        The method behind the final-exam requirement: translate a
        posterior-based finding into frequentist terms for a skeptical audience
        who will ask for a p-value, without misstating either one.

        Parameters
        ----------
        level : float, default 0.95
            Credible level to quote.

        Returns
        -------
        _Report
            Printable text contrasting what the p-value claims with what the
            posterior claims for this specific data.
        """
        lo, hi = self.interval(level=level)
        prob = self.probability(">", self.direction_reference)
        pct = f"{level:.0%}"

        exceeds = (
            "is greater than zero"
            if self.direction_reference == 0.0
            else f"exceeds {self._fmt(self.direction_reference)}"
        )

        lines = [f"THE QUANTITY: {self.quantity}", "", "What the posterior says"]
        lines += _wrap(
            f"Given this data and the stated prior, there is a {pct} probability "
            f"the value lies in the range {self._fmt_range(lo, hi)}, and a "
            f"{prob:.0%} probability it {exceeds}. Both are statements about "
            "the quantity itself.",
            prefix="  ",
        )
        lines += [""]

        if self.frequentist is not None:
            f = self.frequentist
            lines += [f"What the {f.test} says"]
            lines += _wrap(f.claims(), prefix="  ")
            lines += _wrap(f.disclaims(), prefix="  ")
            if f.interval is not None:
                lines += _wrap(
                    f.interval_claims(formatter=self._fmt_range), prefix="  "
                )
            lines += [""] + self._reconciliation()
        return _Report("\n".join(lines))

    def _reconciliation(self) -> list[str]:
        """Explain how the two answers relate for this particular data.

        Returns
        -------
        list of str
            Unindented lines; callers add their own left margin.
        """
        if self.frequentist is None:
            return []
        prob = self.probability_of_direction()
        p = self.frequentist.pvalue
        sig = self.frequentist.significant
        null = self.frequentist.null_statement or "the null hypothesis held"
        thing = "difference" if self.is_comparison else "value"

        if not sig and prob >= 0.9:
            title = "Why they look like they disagree"
            body = (
                "They are answering different questions. The test asks how "
                f"surprising this data would be in a world where {null}, and "
                f"{p:.1%} is not surprising enough to clear the conventional "
                f"bar. The posterior asks how plausible each possible {thing} "
                f"is given the data you have, and puts {prob:.0%} of that "
                "plausibility on one side. Neither is wrong. Only the second "
                "one is an input to a decision."
            )
        elif sig and prob >= 0.9:
            title = "How they line up"
            body = (
                "Both point the same way here, so the choice between them is "
                "not about the conclusion but about what you can say. "
                "'Significant' licenses a claim about the data; the posterior "
                "licenses a claim about the size of the effect, which is what a "
                "recommendation actually needs."
            )
        else:
            title = "How they line up"
            body = (
                "Neither framework finds much to report. The useful difference "
                "is that the posterior says so by showing you an interval wide "
                "enough to include outcomes you would act on differently, which "
                "is more informative than a p-value above 0.05 and much harder "
                "to mistake for evidence of no effect."
            )
        return [title, *_wrap(body, prefix="  ", width=68)]

    # -- Bayes factor ------------------------------------------------------

    def bayes_factor(self) -> BayesFactor:
        """Evidence for a difference relative to no difference.

        Deliberately a method call rather than part of the printed summary.
        Bayes factors answer a model-comparison question, not an estimation
        question, and they depend on the prior in a way credible intervals
        largely stop doing once there is a reasonable amount of data. Reach for
        this when a reviewer asks for it, not when writing a recommendation.

        Returns
        -------
        BayesFactor
            The Bayes factor with its interpretation and caveat.

        Raises
        ------
        NotImplementedError
            If this analysis does not define one.
        """
        if self.log_bf10 is None:
            raise NotImplementedError(
                self.no_bayes_factor_reason
                or f"{self.analysis or 'this analysis'} does not define a "
                "Bayes factor. The posterior and credible interval above "
                "answer the estimation question directly."
            )
        bf10 = float(np.exp(self.log_bf10))
        caveat = self.bf_caveat or (
            "Depends on the prior, not just the data. Run .sensitivity() before "
            "quoting it."
        )
        return BayesFactor(
            bf10=bf10,
            bf01=float(np.exp(-self.log_bf10)),
            log_bf10=float(self.log_bf10),
            interpretation=_grade_bayes_factor(bf10, self.bf_alternative),
            caveat=caveat,
        )

    # -- sensitivity -------------------------------------------------------

    def sensitivity(self, priors: Sequence[Any] | None = None) -> _Report:
        """Rerun the analysis under a range of priors and show what moves.

        A first-class method rather than an advanced footnote, because prior
        sensitivity is the strongest legitimate objection to canned Bayesian
        analysis, and a package that reports it by default converts that
        objection into a scheduled lecture. No frequentist package has an
        equivalent -- which is not because frequentist methods have no
        assumptions, only because theirs are not parameters you can vary.

        Parameters
        ----------
        priors : sequence, optional
            Prior specifications to try. Defaults to this analysis's ladder,
            ordered from least to most constraining.

        Returns
        -------
        _Report
            A table of the interval, the direction probability, and the Bayes
            factor under each prior, plus a verdict on whether the conclusion
            survived.

        Raises
        ------
        NotImplementedError
            If this analysis cannot be refit under a different prior.
        """
        if self._refit is None or (priors is None and not self.prior_ladder):
            raise NotImplementedError(
                self.no_sensitivity_reason
                or f"{self.analysis or 'this analysis'} cannot be refit under "
                "a different prior."
            )
        specs = list(priors) if priors is not None else list(self.prior_ladder)
        if not specs:
            raise ValueError("no priors to compare.")

        lines = ["HOW MUCH DOES THE PRIOR MATTER?"]
        lines += _wrap(self.quantity, prefix="quantity: ")
        if self.unit:
            lines += [f"interval in {self.unit}"]
        lines += [
            "",
            f"{'prior':<30}{'95% interval':<21}{'P(dir)':>9}{'BF10':>12}",
            _THIN,
        ]
        probs, bounds, bfs = [], [], []
        for spec in specs:
            res = self._refit(spec)
            lo, hi = res.interval()
            signed = res.probability(">", res.direction_reference)
            shown = max(signed, 1.0 - signed) if self.is_comparison else signed
            probs.append(signed)
            bounds.append((lo, hi))
            label = getattr(res.prior, "label", str(spec))
            span = res._fmt_range(lo, hi).removesuffix(self.unit).strip()
            try:
                bf10 = res.bayes_factor().bf10
                bfs.append(bf10)
                bf = f"{bf10:.3g}"
            except NotImplementedError:
                bf = "n/a"
            lines += [f"{label:<30}{span:<21}{shown:>9.3f}{bf:>12}"]

        lines += [""] + self._sensitivity_verdict(probs, bounds, bfs)
        return _Report("\n".join(lines))

    def _sensitivity_verdict(self, probs, bounds, bfs=()) -> list[str]:
        """Say whether the conclusion survived the range of priors tried.

        Reports the estimate and the Bayes factor separately, because they do
        not have the same prior sensitivity and pretending otherwise is the
        mistake this method exists to prevent. A credible interval built on a
        few hundred observations is usually robust to any reasonable prior; a
        Bayes factor can move by an order of magnitude over the same range,
        because it is comparing models rather than locating a value.

        Returns
        -------
        list of str
            Wrapped verdict lines.
        """
        spread = max(probs) - min(probs)
        same_side = all(p >= 0.5 for p in probs) or all(p < 0.5 for p in probs)
        widths = [hi - lo for lo, hi in bounds]
        width_change = (max(widths) - min(widths)) / max(widths)

        if spread < 0.05 and same_side:
            width_text = (
                "by less than 1%" if width_change < 0.01 else f"by {width_change:.0%}"
            )
            body = (
                "the estimate barely moves. The direction probability shifts by "
                f"less than 0.05 across every assumption tried and the interval "
                f"width {width_text}, so this conclusion is driven by the data. "
                "Report it, and note that you checked."
            )
        elif same_side:
            body = (
                "the direction holds under every prior tried, but its strength "
                f"moves by {spread:.2f}. Report the range rather than a single "
                "number, and say which prior you used and why."
            )
        else:
            body = (
                "the conclusion flips depending on the prior. That is a real "
                "finding, not a failure — it means this data alone cannot settle "
                "the question, and the answer currently rests on the assumption "
                "rather than on the evidence."
            )
        out = _wrap(body, prefix="Verdict: ")

        usable = [b for b in bfs if np.isfinite(b) and b > 0]
        if len(usable) > 1:
            ratio = max(usable) / min(usable)
            if ratio >= 3.0:
                out += [""] + _wrap(
                    f"the Bayes factor is a different story: it moves by a "
                    f"factor of {ratio:.0f} over the same priors, while the "
                    "interval hardly budged. That gap is not a bug in either "
                    "number — it is why this package leads with the estimate. "
                    "Locating a value is a question the data can mostly answer "
                    "on its own; grading the evidence for one model against "
                    "another cannot be separated from what you assumed the "
                    "alternative looked like.",
                    prefix="But note: ",
                )
            else:
                out += [""] + _wrap(
                    f"the Bayes factor moves by a factor of {ratio:.1f} over the "
                    "same priors, which is mild by the standards of Bayes "
                    "factors. Quote it with the prior stated anyway.",
                    prefix="Also: ",
                )
        return out

    # -- output ------------------------------------------------------------

    def _fmt(self, value: float) -> str:
        """Format a value in display units, with the unit appended."""
        scaled = float(value) * self.display_scale
        text = f"{scaled:.{self.decimals}f}".replace("-", "−")
        return _join_unit(text, self.unit)

    def _fmt_range(self, low: float, high: float) -> str:
        """Format an interval with the unit stated once, at the end."""
        lo = f"{float(low) * self.display_scale:.{self.decimals}f}"
        hi = f"{float(high) * self.display_scale:.{self.decimals}f}"
        lo, hi = lo.replace("-", "−"), hi.replace("-", "−")
        return _join_unit(f"{lo} to {hi}", self.unit)

    def _direction_label(self) -> str:
        """Short label for the reported direction probability."""
        if self.is_comparison:
            prob = self.probability(">", self.direction_reference)
            higher = self.higher_label if prob >= 0.5 else self.lower_label
            return f"P({higher} higher)"
        return f"P(above {self._fmt(self.direction_reference)})"

    def _next_steps_line(self) -> str:
        """Suggest the methods that actually work on this result."""
        if self.next_steps:
            return self.next_steps
        parts = [".probability('>', x)", ".decide(rope=(lo, hi))"]
        if self._refit is not None and self.prior_ladder:
            parts.append(".sensitivity()")
        return "  ".join(parts)

    def summary(self, level: float = 0.95, kind: str = "hdi") -> _Report:
        """Build the full side-by-side report: posterior first, then its twin.

        Parameters
        ----------
        level : float, default 0.95
            Credible and confidence level to report.
        kind : {'hdi', 'eti'}, default 'hdi'
            Interval type for the Bayesian half.

        Returns
        -------
        _Report
            Printable text. Use ``print(res.summary())`` in a script, or let
            Jupyter display it directly.
        """
        pct = f"{level:.0%}"
        lo, hi = self.interval(level=level, kind=kind)
        prob = self.probability(">", self.direction_reference)
        prob = max(prob, 1.0 - prob) if self.is_comparison else prob
        pad = 26

        lines = [_RULE, *_wrap(self.quantity, prefix=" "), _RULE, ""]

        # -- Bayesian half
        lines += [
            " BAYESIAN — what the data say about the quantity itself",
            "",
            f"   {'most likely value':<{pad}}{self._fmt(self.point())}",
            f"   {pct + ' credible interval':<{pad}}"
            f"{self._fmt_range(lo, hi)}  ({kind.upper()})",
            f"   {self._direction_label():<{pad}}{prob:.3f}",
            "",
        ]
        lines += _wrap(self.sentence(level=level), prefix="   Read: ")
        lines += [""]

        # -- frequentist half
        if self.frequentist is not None:
            f = self.frequentist
            dof = f" ({_fmt_dof(f.dof)} df)" if f.dof is not None else ""
            verdict = (
                "significant at 0.05" if f.significant else "not significant at 0.05"
            )
            lines += [
                f" FREQUENTIST — {f.test}",
                "",
                f"   {f.statistic_name + dof:<{pad}}{f.statistic:.4g}",
                f"   {'p-value':<{pad}}{f.pvalue:.4g}  ({verdict})",
            ]
            if f.interval is not None:
                flo, fhi = f.interval
                lines += [
                    f"   {pct + ' confidence interval':<{pad}}"
                    f"{self._fmt_range(flo, fhi)}  ({f.interval_method})"
                ]
            lines += [""]
            lines += _wrap(f.claims(), prefix="   Read: ")
            lines += ["", "   " + _THIN[:68], ""]
            lines += ["   " + line if line else "" for line in self._reconciliation()]
            lines += [""]

        # -- footer
        prior_label = getattr(self.prior, "label", None)
        lines += [_THIN]
        if prior_label:
            lines += [f" assumption   prior: {prior_label}"]
        if self.exact:
            lines += [
                " precision    exact — the posterior is closed form, so these "
                "digits are not\n              estimates and there is nothing "
                "to converge"
            ]
        else:
            mcse = self.monte_carlo_error()
            lines += [
                f" precision    {self.n_draws:,} independent draws; Monte Carlo "
                f"error on the mean\n              is ±{mcse:.2g}. These are "
                "exact samples, not a Markov chain."
            ]
        for note in self.notes:
            lines += _wrap(note, prefix=" note         ")
        lines += _wrap(self._next_steps_line(), prefix=" next         ")
        lines += [_RULE]
        return _Report("\n".join(lines))

    def to_dict(self, level: float = 0.95) -> dict[str, Any]:
        """Flat dictionary of the headline numbers, for autograding.

        Parameters
        ----------
        level : float, default 0.95
            Credible level for the reported interval.

        Returns
        -------
        dict
            Keys are stable across releases; compare with a tolerance.
        """
        lo, hi = self.interval(level=level)
        out: dict[str, Any] = {
            "analysis": self.analysis,
            "quantity": self.quantity,
            "point_median": self.point("median"),
            "point_mean": self.point("mean"),
            "interval_low": lo,
            "interval_high": hi,
            "interval_level": level,
            "probability_of_direction": self.probability_of_direction(),
            "probability_above_reference": self.probability(
                ">", self.direction_reference
            ),
            "exact": self.exact,
            "n_draws": self.n_draws,
            "monte_carlo_se": self.monte_carlo_error(),
            "prior": getattr(self.prior, "label", None),
        }
        if self.log_bf10 is not None:
            out["log_bf10"] = float(self.log_bf10)
            out["bf10"] = float(np.exp(self.log_bf10))
        if self.frequentist is not None:
            out["frequentist_test"] = self.frequentist.test
            out["frequentist_statistic"] = self.frequentist.statistic
            out["p_value"] = self.frequentist.pvalue
            if self.frequentist.interval is not None:
                out["ci_low"], out["ci_high"] = self.frequentist.interval
        return out

    # -- plotting ----------------------------------------------------------

    def plot_kinds(self) -> list[str]:
        """List the plot kinds this particular result supports.

        Returns
        -------
        list of str
            Names accepted by :meth:`plot`.
        """
        kinds = ["posterior"]
        if self.prior is not None and self.prior.dist() is not None:
            kinds.append("prior_posterior")
        if self.components:
            kinds.append("components")
        kinds.extend(sorted(self.custom_plots))
        return kinds

    def plot(
        self,
        kind: str = "posterior",
        threshold: float | None = None,
        level: float = 0.95,
        ax=None,
        **kwargs,
    ):
        """Plot the posterior, optionally shaded at a threshold.

        Parameters
        ----------
        kind : str, default 'posterior'
            What to draw. ``'posterior'`` and ``'prior_posterior'`` are always
            available; ``'components'`` shows each group's posterior
            separately, which is the picture that explains where a difference
            came from. Some analyses add their own -- a correlation offers
            ``'scatter'``. See ``.plot_kinds()``.
        threshold : float, optional
            Shade the posterior beyond this value and label the shaded mass.
            Defaults to ``direction_reference`` for comparisons.
        level : float, default 0.95
            Credible level to mark.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. Created if omitted.
        **kwargs
            Passed to the underlying fill.

        Returns
        -------
        matplotlib.axes.Axes
            The axes drawn on.

        Raises
        ------
        ImportError
            If matplotlib is not installed. Install with
            ``pip install bayesplain[plot]``.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as err:  # pragma: no cover - environment dependent
            raise ImportError(
                "plotting needs matplotlib. Install it with "
                "`pip install bayesplain[plot]`."
            ) from err

        if ax is None:
            _, ax = plt.subplots(figsize=(7.5, 4.0))
        scale = self.display_scale

        if kind in self.custom_plots:
            return self.custom_plots[kind](ax)
        if kind == "components":
            return self._plot_components(ax, scale)

        grid, density = self._density_curve()
        ax.plot(grid * scale, density, color="#22223b", lw=2, zorder=3)
        ax.fill_between(
            grid * scale, density, color="#4a4e69", alpha=0.18, zorder=2, **kwargs
        )

        if kind == "prior_posterior":
            # Not every prior has a one-dimensional density to overlay: a
            # Dirichlet over a table lives on a simplex, so its .dist() is
            # None and the prior layer is simply skipped.
            prior_dist = self.prior.dist() if self.prior is not None else None
            if prior_dist is None:
                raise ValueError(
                    f"the prior for this {self.analysis or 'analysis'} has no "
                    "single density curve to draw against the posterior — it "
                    "is a distribution over a whole table, not over one "
                    "number. Use kind='posterior' or kind='components'."
                )
            ax.plot(
                grid * scale,
                prior_dist.pdf(grid),
                color="#9a8c98",
                lw=1.6,
                ls="--",
                zorder=3,
                label=f"prior ({getattr(self.prior, 'name', 'prior')})",
            )
            ax.legend(frameon=False)

        if threshold is None and self.is_comparison:
            threshold = self.direction_reference
        if threshold is not None:
            mass = self.probability(">", threshold)
            label = (
                self._direction_label()
                if threshold == self.direction_reference
                else f"P(above {self._fmt(threshold)})"
            )
            if mass < 0.5:
                mass, label = 1.0 - mass, label.replace("higher", "lower")
                beyond = grid <= threshold
            else:
                beyond = grid >= threshold
            ax.fill_between(
                grid[beyond] * scale,
                density[beyond],
                color="#c9184a",
                alpha=0.28,
                zorder=2,
            )
            ax.axvline(threshold * scale, color="#c9184a", lw=1.2, zorder=4)
            ax.annotate(
                f"{label} = {mass:.2f}",
                xy=(0.98, 0.94),
                xycoords="axes fraction",
                ha="right",
                fontsize=10,
                color="#c9184a",
            )

        lo, hi = self.interval(level=level)
        ax.plot(
            [lo * scale, hi * scale],
            [0, 0],
            color="#22223b",
            lw=4,
            solid_capstyle="butt",
            zorder=5,
        )
        ax.annotate(
            f"{level:.0%} credible interval",
            xy=(0.5 * (lo + hi) * scale, 0),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#22223b",
        )
        self._style_axes(ax)
        return ax

    def _style_axes(self, ax) -> None:
        """Apply the shared axis treatment for every posterior plot.

        The density axis carries no tick labels on purpose. Its numbers are
        not interpretable by the audience this package is for, and because the
        horizontal axis is rescaled for display -- a difference in proportions
        shown as percentage points -- printed density values would not
        integrate to one against it. Height and shaded area are the only things
        a reader should take from the vertical direction, and both survive
        without a scale.
        """
        ax.set_xlabel(_join_unit(self.quantity + "   ", self.unit).strip())
        ax.set_ylabel("how plausible each value is")
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_ylim(bottom=0)

    def _plot_components(self, ax, scale):
        """Draw each group's own posterior on shared axes.

        This plot is on a different scale from the headline quantity: the
        components are each group's rate, not the difference or ratio between
        them, so it uses ``component_scale`` and ``component_axis`` rather than
        the display settings of the estimand.
        """
        if not self.components:
            raise ValueError(
                "this result has no per-group components to plot; use kind='posterior'."
            )
        scale = self.component_scale if self.component_scale is not None else scale
        for label, obj in self.components.items():
            sample = obj.rvs(size=20_000) if hasattr(obj, "rvs") else np.asarray(obj)
            lo, hi = np.quantile(sample, [0.001, 0.999])
            grid = np.linspace(lo, hi, 400)
            if hasattr(obj, "pdf"):
                density = obj.pdf(grid)
            else:
                density = np.histogram(sample, bins=grid, density=True)[0]
                grid = 0.5 * (grid[1:] + grid[:-1])
            ax.plot(grid * scale, density, lw=2, label=str(label))
            ax.fill_between(grid * scale, density, alpha=0.15)
        ax.legend(frameon=False)
        ax.set_xlabel(self.component_axis or "value for each group")
        ax.set_ylabel("how plausible each value is")
        ax.set_yticks([])
        ax.set_ylim(bottom=0)
        ax.spines[["top", "right", "left"]].set_visible(False)
        return ax

    def _density_curve(self, points: int = 512):
        """Grid and density for the posterior, analytic where possible."""
        if self.exact:
            lo, hi = self.posterior.ppf(0.0005), self.posterior.ppf(0.9995)
            grid = np.linspace(lo, hi, points)
            return grid, self.posterior.pdf(grid)
        lo, hi = np.quantile(self.draws, [0.0005, 0.9995])
        pad = 0.05 * (hi - lo)
        grid = np.linspace(lo - pad, hi + pad, points)
        try:
            from scipy.stats import gaussian_kde

            return grid, gaussian_kde(self.draws)(grid)
        except Exception:  # pragma: no cover - degenerate posteriors only
            counts, edges = np.histogram(self.draws, bins=points, density=True)
            return 0.5 * (edges[1:] + edges[:-1]), counts

    # -- repr --------------------------------------------------------------

    def __repr__(self) -> str:
        lo, hi = self.interval()
        return (
            f"<{self.analysis or 'Result'}: {self.quantity} = "
            f"{self._fmt(self.point())} "
            f"[{self._fmt_range(lo, hi)}]  "
            f"P(direction) = {self.probability_of_direction():.2f}>"
            "\nCall .summary() for the full report."
        )
