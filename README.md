# bayesplain

**Bayesian answers to the questions frequentist tests are usually asked.**

`bayesplain` gives you the Bayesian version of the handful of tests an
introductory statistics course is built around — proportions, means,
contingency tables, correlation, group comparisons — and prints the
conventional test alongside it every time.

It exists because there is a real hole in the Python ecosystem. `pingouin`
gives you Bayes factors for t-tests and correlations; `PyMC` and `Bambi` own the
modeling end. Nothing covers the middle: contingency tables, proportions, and
group comparisons with one consistent interface and output written to be
*explained* rather than pasted into a methods section. This fills that middle,
and it is not a port of R's `BayesFactor` — the positioning difference is the
product.

## Three design commitments

**1. Estimation first, Bayes factors second.** The headline output is a
posterior, a credible interval, and the probability of clearing a threshold you
name. `BF = 4.2` is not a sentence anyone puts in a memo, so the Bayes factor is
a method call (`.bayes_factor()`) rather than a printed default.

**2. Every result carries its frequentist twin.** The chi-square statistic, the
t, the p-value, the confidence interval — computed and printed next to the
posterior, with a plain-English note on what each does and does not claim. You
cannot use this package without seeing both numbers for every analysis you run.

**3. No sampler in the dependency chain.** numpy and scipy, full stop, with
matplotlib optional. Everything is a conjugate posterior, a closed-form Bayes
factor, or a one-dimensional integral. It installs in a Colab cell in seconds,
with no compiler and no convergence warnings, ever. This constraint is also the
scope police: anything that would need MCMC is out of scope by construction, and
the docs hand you off to `Bambi` at that boundary.

## Install

```bash
pip install bayesplain          # numpy + scipy only
pip install bayesplain[plot]    # adds matplotlib
```

## The example the whole thing is built around

Two districts' eviction filing rates. The Bayesian answer is actionable; the
frequentist answer is "not significant"; both come from the same four numbers.

```python
import bayesplain as bp

res = bp.compare_proportions(
    successes=[34, 51],
    n=[220, 240],
    labels=["District A", "District B"],
)
print(res.summary())
```

```
==========================================================================
 difference in rate (District B − District A)
==========================================================================

 BAYESIAN — what the data say about the quantity itself

   most likely value         5.7 percentage points
   95% credible interval     −1.2 to 12.9 percentage points  (HDI)
   P(District B higher)      0.945

   Read: District B is higher than District A with 94% probability; the
         gap is most likely 5.7 percentage points, and the data are
         consistent with anything from −1.2 to 12.9 percentage points
         (95% credible interval).

 FREQUENTIST — two-proportion z-test (equivalently chi-square, 1 df)

   chi-square (1 df)         2.559
   p-value                   0.1096  (not significant at 0.05)
   95% confidence interval   −1.2 to 12.8 percentage points  (Wald)

   Read: If the two rates were identical, data at least this extreme would
         turn up 11.0% of the time.

   --------------------------------------------------------------------

   Why they look like they disagree
     They are answering different questions. The test asks how surprising
     this data would be in a world where the two rates were identical,
     and 11.0% is not surprising enough to clear the conventional bar.
     The posterior asks how plausible each possible difference is given
     the data you have, and puts 94% of that plausibility on one side.
     Neither is wrong. Only the second one is an input to a decision.

--------------------------------------------------------------------------
 assumption   prior: uninformed — Beta(1, 1)
 precision    100,000 independent draws; Monte Carlo error on the mean
              is ±0.00011. These are exact samples, not a Markov chain.
 next         .probability('>', x)  .decide(rope=(lo, hi))  .sensitivity()
==========================================================================
```

Note the two intervals nearly coincide. The numbers agree; only the sentences
differ. That is the lesson, not a coincidence to explain away.

## What you do with a Result

Every analysis returns the same object with the same methods, so what you learn
in week 2 still works in week 9.

```python
res.probability(">", 0.05)     # P(the gap exceeds 5 points)
res.interval(0.95, kind="hdi") # or kind="eti" to line up with a CI
res.decide(rope=(-0.02, 0.02)) # HDI + ROPE, in place of "reject the null"
res.sentence()                 # one memo-ready line
res.translate()                # what the p-value claims vs the posterior
res.sensitivity()              # how the conclusion moves as the prior widens
res.plot(kind="components")    # where the difference came from
res.plot_kinds()               # what this particular result can draw
res.to_dict()                  # autograding hook
res.draws                      # ndarray, so you can compute anything
res.bayes_factor()             # available, documented, never the headline
```

`.sensitivity()` is a first-class method, not an advanced footnote. Prior
sensitivity is the strongest legitimate objection to canned Bayesian analysis,
and reporting it by default converts that objection into a scheduled lecture. On
the example above it also surfaces something worth teaching: the credible
interval barely moves across every prior in the ladder, while the Bayes factor
moves by a factor of 19. That asymmetry is the reason this package leads with
the estimate.

## Priors have names

Nobody should have to type a shape parameter to get started, and nobody should
be able to hide which assumption they used.

```python
bp.priors.describe()                                  # what each preset assumes
bp.proportion(34, 220, prior="skeptical")             # named preset
bp.proportion(34, 220, prior=(2, 5))                  # explicit (a, b)
bp.priors.from_previous_study(successes=12, n=80)     # an earlier study as a prior
```

The resolved prior and its parameters print in every summary.

## Reproducibility

The default seed is fixed (`bp.get_seed()` → `12345`). That is unusual for a
research package and correct for a teaching one: if every student gets different
digits in the last decimal place, office hours fill with questions about Monte
Carlo noise instead of questions about statistics. Having hidden that noise, the
package then surfaces it — every sampled quantity prints its Monte Carlo
standard error. Use `bp.set_seed(None)` to demonstrate that the wobble is real.

## The seven analyses

All built, tested, and documented.

| Function | Frequentist counterpart printed alongside | Method |
| --- | --- | --- |
| `proportion` | exact binomial test, Wilson interval | Beta-Binomial, exact |
| `compare_proportions` | two-proportion z / chi-square, Wald interval | two Betas, exact draws |
| `contingency` | Pearson chi-square test of independence | Gunel-Dickey, all four sampling schemes; posterior over Cramér's V or the log odds ratio |
| `mean` | one-sample t-test | closed-form Student-t posterior |
| `compare_means` | Welch's t-test (or pooled) | Student-t per group; Behrens-Fisher by draws |
| `correlation` | Pearson r, Fisher-z interval | exact sampling density, grid + inverse CDF |
| `compare_groups` | one-way ANOVA | pairwise-first, optional closed-form partial pooling |

Explicitly out of scope: multi-factor ANOVA, interactions, regression with more
than one predictor, mixed models, anything needing importance sampling over
multiple hyperparameters. For those, go to `Bambi` — the docs include a worked
handoff. There is also deliberately no *omnibus* Bayes factor for
`compare_groups`; `.pairwise()` reports a validated one per pair instead, and
the method explains why when you ask for the omnibus version.

## Bundled datasets

Four small, real, pre-cleaned planning datasets ship inside the package, so no
problem set dies because a city reorganised its open-data portal.

```python
bp.datasets.describe()             # one line each
bp.datasets.load_collisions()      # LA and San Diego collisions, 2019-2023
bp.datasets.load_tracts()          # LA County ACS tracts
bp.datasets.load_blockgroups()     # the same variables, finer geography
bp.datasets.load_parcels()         # Baltimore residential parcels
```

The tract/block-group pair exists for one purpose: identical variables,
identical county, two aggregations, and non-overlapping credible intervals for
the same correlation (0.781 vs 0.695). The modifiable areal unit problem, with
the arithmetic to show it is not sampling noise.

## Validation

Nothing is asserted on the strength of having been transcribed carefully.

**Published reference values.** The contingency Bayes factors reproduce the
worked examples in Jamil et al. (2017) under all four Gunel-Dickey schemes:
log BF₁₀ = 23.0337 against a published 23.03 for the doll-preference study,
BF₁₀ = 9.194 against 9.19 (Poisson), 3.043 against 3.04 (hypergeometric), and
the posterior log odds ratio at 2.47 with a 95% interval of (1.73, 3.26) against
a published (1.73, 3.26).

**Independent implementations.** The JZS t-test Bayes factor matches
`pingouin.bayesfactor_ttest` to six significant figures across one-sample,
two-sample, and every prior width; the correlation Bayes factor matches
`pingouin.bayesfactor_pearson` to seven. `pingouin` is a validation-only
dependency, never required at runtime.

**Self-validation.** Marginal likelihoods are checked against brute-force
quadrature of the same integral and against summing to 1 over all outcomes. The
exact correlation sampling density is checked against its own normalisation over
(−1, 1), which pins the hypergeometric transcription without reference to
anyone's code.

**Behavioural properties.** Posterior means fall between the prior mean and the
MLE; interval widths shrink as 1/√n; wider priors weaken the Bayes factor for a
small effect; swapping group order flips the sign and nothing else; results are
invariant to relabelling categories.

**Pedagogical regression tests** (marked `pedagogy`). The numbers a lecture
depends on are pinned as assertions. If a refactor breaks a lecture, CI says so
rather than a student.

Two notes on the contingency Bayes factor, since the earlier version of this
README flagged it as unverified. It is now validated against the published
values above. It is computed in a form that differs algebraically from Jamil et
al.'s Equation 10 — each row drawing its own column distribution rather than
their successive conditioning — and the two agree exactly once expanded in gamma
functions; the advantage of this form is that it accepts a per-column
concentration vector, so a `Beta(a, b)` prior on a rate carries into the Bayes
factor unchanged. Separately, `bayesplain` evaluates the correlation marginal
likelihood in log space, which means it returns a finite answer where `pingouin`
returns `inf` for strongly correlated large samples — including the ~2,400-tract
case in the bundled data.

## Documentation

```bash
pip install -e ".[docs]"
cd docs && make html
```

Sphinx with `sphinx-immaterial` and executed notebooks, in the same shape as
`bayespecon`. Builds clean with no warnings.

## Development

```bash
pip install -e ".[tests,plot,data,validate]"
pytest                  # fast suite, including doctests
pytest -m ''            # add the slow Monte Carlo validation runs
pytest -m validation    # only the cross-implementation checks
pytest -m pedagogy      # only the numbers a lecture depends on
ruff check . && ruff format --check .
```

## License

BSD 3-Clause.
