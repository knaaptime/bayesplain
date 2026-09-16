# Scope, and what is deliberately missing

A package that names its limits is more trustworthy than one that quietly
degrades. Here is what `bayesplain` does not do, and why.

## Not implemented, on purpose

**Multi-factor ANOVA, interactions, and regression with more than one
predictor.** These require integrating over several hyperparameters at once,
which means importance sampling, which means a sampler in the dependency chain.
That would break the constraint the whole package is organised around. Use
[Bambi](https://bambinos.github.io/bambi/), which reads much like `statsmodels`
and is built for exactly this:

```python
import bambi as bmb

model = bmb.Model("assessed_value ~ sqft + construction", data)
fitted = model.fit()
```

**Mixed models and anything with random effects beyond one level of pooling.**
`compare_groups(..., pool=True)` gives you closed-form partial pooling across a
single grouping. Anything nested or crossed belongs in a real modelling
framework.

**An omnibus Bayes factor for `compare_groups`.** This one is a judgement call
rather than a technical limit. The omnibus question — "is there a difference
somewhere among these groups?" — is one a planner almost never needs answered,
and grading the evidence for it requires a prior over every pattern of group
differences at once. `.pairwise()` reports a validated Bayes factor for each
pair you actually care about, which is both more useful and better grounded.

**The hypergeometric contingency scheme for tables larger than 2×2.** The sum
runs over every table sharing the observed margins, which is rarely what a
study design justifies anyway. The error message says so and points at the
independent-multinomial scheme.

**Causal inference of any kind.** Nothing in this package makes a finding
causal, and the correlation analysis attaches a note saying so to every result.

## Known limitations of what is implemented

**Partial pooling is empirical Bayes.** `compare_groups(pool=True)` estimates
the between-group variance and then treats it as known. That is what buys the
closed form. It understates uncertainty slightly when the number of groups is
small — under about five groups, prefer the unpooled estimates and say the
smallest group is uncertain, rather than leaning on a τ² estimated from four
numbers.

**Means assume roughly normal data.** The Student-t posterior comes from a
normal likelihood. For small samples the package adds a note telling you to
look at a histogram. For heavy-tailed or strongly skewed data — house prices,
for instance — consider working on a log scale and reporting the result as a
ratio.

**The correlation posterior is grid-based.** Accurate to the resolution of the
grid, which is adaptive and fine enough that the error is far below anything
that would change a conclusion. It is not a closed form, and the documentation
does not pretend otherwise.

**Bayes factors are prior-sensitive, always.** This is not a limitation of the
implementation but of the quantity. `.sensitivity()` exists to make it visible
rather than to make it go away, and its output frequently shows the interval
barely moving while the Bayes factor moves by an order of magnitude. That is the
honest picture.

## Where the line sits

The rule is simple enough to apply without arguing about it: if a question can
be answered by a conjugate update, a closed-form Bayes factor, or a
one-dimensional integral, it belongs here. Otherwise it belongs in a modelling
framework, and the documentation should hand you over cleanly rather than
growing a worse version of one.
