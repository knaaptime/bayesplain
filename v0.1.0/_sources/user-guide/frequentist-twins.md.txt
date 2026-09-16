---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# The frequentist twin

No result in this package appears without the conventional test that would
ordinarily have been run on the same data. The point is not even-handedness for
its own sake: it is that nobody should finish a course able to produce a
posterior but unable to read a p-value in someone else's report.

| Analysis | Counterpart it prints |
| --- | --- |
| `proportion` | exact binomial test, Wilson score interval |
| `compare_proportions` | two-proportion *z* / chi-square, Wald interval |
| `contingency` | Pearson chi-square test of independence |
| `mean` | one-sample *t*-test |
| `compare_means` | Welch's *t*-test (or pooled Student's) |
| `correlation` | Pearson *r*, Fisher-*z* interval |
| `compare_groups` | one-way ANOVA |

## They usually agree on the numbers

```{code-cell} ipython3
import numpy as np
import bayesplain as bp

rng = np.random.default_rng(0)
x = rng.normal(50, 12, 40)

res = bp.mean(x)
print("credible interval :", np.round(res.interval(kind="eti"), 6))
print("confidence interval:", np.round(res.frequentist.interval, 6))
```

Identical to six decimal places, and not by luck: under the reference prior the
posterior for a mean is exactly the *t* distribution the confidence interval is
built from. The arithmetic is the same. What differs is the sentence you are
allowed to say afterwards.

```{code-cell} ipython3
print(res.translate())
```

## And sometimes they seem not to

```{code-cell} ipython3
res = bp.compare_proportions(
    [34, 51], [220, 240], labels=["District A", "District B"]
)
print(res.summary())
```

"Not significant" and "94% probability District B is higher" are both correct
statements about this data. They are answers to different questions. The test
asks how surprising the data would be in a world where the rates match; the
posterior asks how plausible each difference is given the data in hand. Only
the second is an input to a decision.

## What each one claims, in words

Every twin can state its own case:

```{code-cell} ipython3
twin = res.frequentist
print(twin.claims())
print()
print(twin.disclaims())
print()
print(twin.interval_claims())
```

That last one is the confidence-interval hook. Most people — including plenty
of published researchers — read a 95% CI as "95% chance the true value is in
here". That is wrong, and the error is durable because it is the sentence
people actually want to say. A credible interval is what finally lets them say
it correctly.

## Using the twins on their own

They are ordinary functions, usable without any Bayesian machinery:

```{code-cell} ipython3
from bayesplain import frequentist as fq

fq.two_proportions([34, 51], [220, 240])
```

```{code-cell} ipython3
rng = np.random.default_rng(1)
groups = [rng.normal(m, 3, 40) for m in (10, 12, 11)]
fq.one_way_anova(groups)
```

## Where the Bayesian version genuinely does more

Two places, both worth a class session.

**Effect size with uncertainty.** A chi-square test gives you a statistic and a
p-value; ask how *strong* the association is and it offers a point estimate with
nothing attached. Drawing from the Dirichlet posterior gives that quantity a
full distribution.

```{code-cell} ipython3
import pandas as pd

parcels = bp.datasets.load_parcels()
table = pd.crosstab(parcels.construction, parcels.condition)

res = bp.contingency(table)
lo, hi = res.interval()
print(f"Cramér's V: {res.point():.3f}, 95% credible interval {lo:.3f} to {hi:.3f}")
print(f"chi-square point estimate of V: {res.frequentist.estimate:.3f} (no interval available)")
```

**Sparse tables.** The chi-square approximation needs roughly five expected
cases per cell and misbehaves below that. The posterior does not degrade; it
simply widens, which is the honest response to thin data.

```{code-cell} ipython3
sparse = bp.contingency([[1, 20], [3, 18]])
for note in sparse.notes:
    print("-", note)
```
