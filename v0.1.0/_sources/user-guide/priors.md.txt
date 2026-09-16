---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Priors, and how much they matter

A student who has never seen a Beta distribution should still be able to state
an assumption and be held to it. So priors are addressed by name, with the
numbers they resolve to printed in every summary and one keystroke away for
anyone who wants them.

```{code-cell} ipython3
import bayesplain as bp

print(bp.priors.describe())
```

The names describe *what the assumption says*, not what family it belongs to.
None of them is neutral, and the summary output says so.

## Four families, one argument

Each analysis takes `prior=`, and what it means depends on what is being
estimated.

| Analysis | Prior on | Presets |
| --- | --- | --- |
| `proportion`, `compare_proportions` | the rate, as `Beta(a, b)` | `uninformed`, `jeffreys`, `gentle`, `skeptical` |
| `contingency` | cell probabilities, as `Dirichlet(a)` | same four names |
| `mean`, `compare_means` | standardised effect size, as `Cauchy(scale)` | `modest`, `conventional`, `uninformed`, `generous` |
| `correlation` | the correlation, as a stretched beta | `concentrated`, `modest`, `uninformed`, `generous` |

The first two line up exactly rather than by analogy: for a 2×2 table of
successes and failures, concentration `a` on the columns *is* a `Beta(a, a)`
prior on each group's rate.

```{code-cell} ipython3
bp.priors.resolve_proportion("gentle"), bp.priors.resolve_table("gentle")
```

## Saying it in your own numbers

```{code-cell} ipython3
bp.proportion(34, 220, prior=(2, 5)).prior.label
```

```{code-cell} ipython3
prior = bp.priors.from_previous_study(successes=12, n=80)
prior, prior.prior_mean, prior.prior_weight
```

An earlier study is treated as data observed under a flat prior, which makes
the strength of the assumption legible: 80 prior cases are worth 80 cases, no
more.

## Watching the prior stop mattering

```{code-cell} ipython3
import pandas as pd

rows = []
for n in (20, 100, 500, 2000):
    for name in bp.priors.SENSITIVITY_LADDER:
        res = bp.proportion(int(0.3 * n), n, prior=name)
        rows.append({"n": n, "prior": name, "estimate": round(res.point(), 4)})

pd.DataFrame(rows).pivot(index="n", columns="prior", values="estimate")
```

At *n* = 20 the choice moves the answer in the second decimal place. By
*n* = 2000 the columns are indistinguishable. This is the week-3 lesson in one
table: the prior matters exactly when the data are thin, which is exactly when
you should be saying so out loud.

## Reporting it honestly

`.sensitivity()` is a first-class method rather than an advanced footnote,
because prior sensitivity is the strongest legitimate objection to canned
Bayesian analysis, and reporting it by default converts that objection into a
scheduled lecture.

```{code-cell} ipython3
res = bp.compare_proportions([34, 51], [220, 240], labels=["A", "B"])
print(res.sensitivity())
```

Two things to notice. The interval barely moves — this conclusion is driven by
the data. The Bayes factor moves by a factor of nearly twenty over the same
range. Both facts are true, and reporting only the second would be misleading
in one direction while reporting only the first would be misleading in the
other.

## The cleanest demonstration

For means, the prior does not touch the estimate at all. The posterior uses the
standard reference prior and is fixed by the data; `prior=` enters only the
Bayes factor.

```{code-cell} ipython3
import numpy as np

rng = np.random.default_rng(0)
x, y = rng.normal(0, 1, 50), rng.normal(0.4, 1, 50)

rows = []
for name in bp.priors.EFFECT_SENSITIVITY_LADDER:
    res = bp.compare_means(x, y, prior=name)
    lo, hi = res.interval()
    rows.append({
        "prior": name,
        "interval": f"{lo:.4f} to {hi:.4f}",
        "BF10": round(float(np.exp(res.log_bf10)), 3),
    })

pd.DataFrame(rows)
```

Identical intervals to four decimal places; a Bayes factor that moves by half
again. Locating a value is a question the data can mostly answer on its own.
Grading the evidence for one model against another cannot be separated from
what you assumed the alternative looked like.
