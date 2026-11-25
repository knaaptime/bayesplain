---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# The bundled datasets

Four small, pre-cleaned planning datasets travel with the package rather than
being downloaded. That matters more than it looks: a problem set built on a
live URL stops working the month a city reorganises its open-data portal, and a
class that spends its first twenty minutes on data wrangling never gets to the
statistics.

They are also real — extracts of public data, cleaned but not invented — so the
numbers a student reports are numbers about an actual place.

```{code-cell} ipython3
import bayesplain as bp

print(bp.datasets.describe())
```

```{code-cell} ipython3
print(bp.datasets.describe("collisions"))
```

## Collisions: the two-group comparison

```{code-cell} ipython3
df = bp.datasets.load_collisions()
counts = df.groupby("county").alcohol_involved.agg(["sum", "size"])
counts
```

```{code-cell} ipython3
res = bp.compare_proportions(
    counts["sum"].to_numpy(), counts["size"].to_numpy(),
    labels=list(counts.index),
)
res.sentence()
```

And a contingency table from the same file:

```{code-cell} ipython3
import pandas as pd

table = pd.crosstab(df.road_surface, df.severity)
table
```

```{code-cell} ipython3
print(bp.contingency(table).summary())
```

## Tracts and block groups: the same place, twice

This pair exists for one purpose. Identical variables, identical county, two
aggregations — and a different answer.

```{code-cell} ipython3
tracts = bp.datasets.load_tracts().dropna(subset=["median_income", "median_rent"])
blocks = bp.datasets.load_blockgroups().dropna(subset=["median_income", "median_rent"])

for name, d in [("tracts", tracts), ("block groups", blocks)]:
    res = bp.correlation(
        d.median_income, d.median_rent,
        labels=["median income", "median rent"], aggregated=True,
    )
    lo, hi = res.interval()
    print(f"{name:13} n={len(d):5}  r={res.point():.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
```

The intervals do not overlap. This is not sampling noise that more data would
settle — it is the modifiable areal unit problem, and it means the correlation
is partly a property of the boundaries someone drew. Redraw them and the number
moves.

The related warning is ecological correlation: a relationship between area
averages can differ from the relationship among individuals, sometimes in sign.
Passing `aggregated=True` attaches both caveats to the result, so they travel
with the analysis rather than living only in a lecture.

```{code-cell} ipython3
res = bp.correlation(
    tracts.median_income, tracts.median_rent, aggregated=True
)
for note in res.notes:
    print("-", note)
```

## Parcels: groups, and why the smallest one lies

```{code-cell} ipython3
parcels = bp.datasets.load_parcels()
parcels.groupby("construction").assessed_value.agg(["count", "median"])
```

```{code-cell} ipython3
groups = {
    name: sub.assessed_value.to_numpy()
    for name, sub in parcels.groupby("construction")
}

raw = bp.compare_groups(groups, unit="dollars")
pooled = bp.compare_groups(groups, unit="dollars", pool=True)
```

```{code-cell} ipython3
import numpy as np

comparison = pd.DataFrame({
    "n": [len(groups[g]) for g in raw.group_names],
    "unpooled": [np.median(raw.group_draws[g]) for g in raw.group_names],
    "pooled": [np.median(pooled.group_draws[g]) for g in raw.group_names],
    "weight": pooled.pooling["weights"],
}, index=raw.group_names).round(2)
comparison.sort_values("n")
```

The groups with the fewest parcels have the smallest weights, meaning they were
pulled hardest toward the overall average. That is the correction working: a
small group has room to wander, and its observed average is weaker evidence
about its true value than a large group's is.

```{code-cell} ipython3
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharex=True)
raw.plot(kind="forest", ax=axes[0])
axes[0].set_title("no pooling", fontsize=10)
pooled.plot(kind="forest", ax=axes[1])
axes[1].set_title("pool=True", fontsize=10)
fig.tight_layout()
```

## Provenance

Every dataset records where it came from:

```{code-cell} ipython3
for name in bp.datasets.available():
    print(f"{name}: {bp.datasets.DATASETS[name].source}\n")
```
