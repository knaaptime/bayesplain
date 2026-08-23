# bayesplain

**Bayesian answers to the questions frequentist tests are usually asked.**

`bayesplain` gives you the Bayesian version of the handful of tests an
introductory statistics course is built around — proportions, means,
contingency tables, correlation, group comparisons — and prints the
conventional test alongside it every time.

It exists to fill a real gap. [`pingouin`](https://pingouin-stats.org) gives you
Bayes factors for t-tests and correlations; [`PyMC`](https://www.pymc.io) and
[`Bambi`](https://bambinos.github.io/bambi/) own the modelling end. Nothing
covers the middle: contingency tables, proportions, and group comparisons with
one consistent interface and output written to be *explained* rather than
pasted into a methods section.

```python
import bayesplain as bp

res = bp.compare_proportions(
    successes=[34, 51],
    n=[220, 240],
    labels=["District A", "District B"],
)
print(res.summary())
```

```text
==========================================================================
 difference in rate (District B − District A)
==========================================================================

 BAYESIAN — what the data say about the quantity itself

   most likely value         5.7 percentage points
   95% credible interval     −1.2 to 12.9 percentage points  (HDI)
   P(District B higher)      0.945

 FREQUENTIST — two-proportion z-test (equivalently chi-square, 1 df)

   chi-square (1 df)         2.559
   p-value                   0.1096  (not significant at 0.05)
   95% confidence interval   −1.2 to 12.8 percentage points  (Wald)
==========================================================================
```

The two intervals nearly coincide. The numbers agree; only the sentences
differ. That is the lesson, not a coincidence to explain away.

## Three design commitments

**Estimation first, Bayes factors second.** The headline output is a posterior,
a credible interval, and the probability of clearing a threshold you name.
`BF = 4.2` is not a sentence anyone puts in a memo, so the Bayes factor is a
method call rather than a printed default.

**Every result carries its frequentist twin.** The chi-square statistic, the t,
the p-value, the confidence interval — computed and printed next to the
posterior, with a plain-English note on what each does and does not claim.

**No sampler in the dependency chain.** numpy and scipy, full stop, with
matplotlib optional. Everything is a conjugate posterior, a closed-form Bayes
factor, or a one-dimensional integral. It installs in a Colab cell in seconds,
with no compiler and no convergence warnings, ever.

```{toctree}
:maxdepth: 1
:caption: Getting started

installation
user-guide/quickstart
```

```{toctree}
:maxdepth: 1
:caption: Guides

user-guide/priors
user-guide/frequentist-twins
user-guide/datasets
```

```{toctree}
:maxdepth: 1
:caption: Reference

api
validation
scope
references
```
