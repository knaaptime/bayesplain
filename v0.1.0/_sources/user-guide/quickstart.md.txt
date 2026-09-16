---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Quickstart

Every analysis in `bayesplain` returns the same object with the same methods,
so what you learn here works for all seven.

```{code-cell} ipython3
import bayesplain as bp

bp.__version__
```

## One analysis, both frameworks

Two counties' rates of alcohol involvement in traffic collisions, from the
bundled data.

```{code-cell} ipython3
df = bp.datasets.load_collisions()
counts = df.groupby("county").alcohol_involved.agg(["sum", "size"])
counts
```

```{code-cell} ipython3
res = bp.compare_proportions(
    successes=counts["sum"].to_numpy(),
    n=counts["size"].to_numpy(),
    labels=list(counts.index),
)
print(res.summary())
```

The posterior comes first because it is what a recommendation is written from.
The conventional test comes second because you will meet it in every report you
read, and because seeing both is the only reliable way to learn what each one
actually claims.

## Asking a question the p-value cannot answer

```{code-cell} ipython3
res.probability(">", 0.02)   # P(the gap exceeds 2 percentage points)
```

```{code-cell} ipython3
print(res.decide(rope=(-0.01, 0.01)))
```

A "ROPE" is a region of practical equivalence: a range you decided in advance
is too small to act on. Comparing the credible interval against it replaces
"reject the null" with a rule that refers to consequences.

## The memo sentence

```{code-cell} ipython3
res.sentence()
```

Have students write this themselves before revealing the method. Otherwise the
convenience quietly becomes the learning objective.

## Translating between frameworks

```{code-cell} ipython3
print(res.translate())
```

## How much does the prior matter?

```{code-cell} ipython3
print(res.sensitivity())
```

Notice what this usually shows: the credible interval barely moves across every
prior, while the Bayes factor moves a great deal. They are answering different
questions, and that asymmetry is why this package leads with the estimate.

## Seeing it

```{code-cell} ipython3
res.plot();
```

```{code-cell} ipython3
res.plot(kind="components");
```

The second plot is the one that explains where a difference came from.

```{code-cell} ipython3
res.plot_kinds()
```

## The Bayes factor, when someone asks

```{code-cell} ipython3
print(res.bayes_factor())
```

Deliberately a method call rather than part of the summary.

## The other six analyses

```{code-cell} ipython3
parcels = bp.datasets.load_parcels()

rowhouses = parcels.loc[parcels.rowhouse, "assessed_value"]
detached = parcels.loc[~parcels.rowhouse, "assessed_value"]

means = bp.compare_means(
    rowhouses, detached,
    labels=["rowhouse", "detached"], unit="dollars",
)
means.sentence()
```

```{code-cell} ipython3
groups = {
    name: sub.assessed_value.to_numpy()
    for name, sub in parcels.groupby("construction")
}
res_groups = bp.compare_groups(groups, unit="dollars", pool=True)
print(res_groups.pairwise(only=[("Stucco", "Brick"), ("Frame", "Brick")]))
```

```{code-cell} ipython3
res_groups.plot(kind="forest");
```

```{code-cell} ipython3
import pandas as pd

table = pd.crosstab(parcels.construction, parcels.condition)
print(bp.contingency(table).summary())
```

```{code-cell} ipython3
tracts = bp.datasets.load_tracts().dropna(subset=["median_income", "median_rent"])
corr = bp.correlation(
    tracts.median_income, tracts.median_rent,
    labels=["median income", "median rent"], aggregated=True,
)
corr.sentence()
```

## Reproducibility in a classroom

The seed is fixed by default, so every student sees identical digits:

```{code-cell} ipython3
bp.get_seed(), bp.get_draws()
```

Having hidden that noise, the package surfaces it — every sampled quantity
prints its Monte Carlo standard error. Use `bp.set_seed(None)` to show a class
that the wobble is real.
