# Validation

Nothing here is asserted on the strength of having been transcribed carefully.
Every closed form is checked against something computed a different way, and
the checks run in CI.

## Published reference values

The contingency-table Bayes factors are validated against the worked examples
in {cite:t}`jamil2017`, under each of the four Gunel-Dickey sampling schemes.

| Example | Scheme | Published | `bayesplain` |
| --- | --- | --- | --- |
| Doll preference {cite:p}`hraba1970` | independent multinomial | log BF₁₀ = 23.03 | 23.0337 |
| Simulation, *c* = 10 | Poisson | BF₁₀ = 9.19 | 9.194 |
| Simulation, *c* = 10 | hypergeometric | BF₁₀ = 3.04 | 3.043 |
| Doll preference, posterior | — | median log OR 2.47, 95% CI (1.73, 3.26) | 2.47, (1.73, 3.26) |

The evidence ordering the paper derives — Poisson > joint multinomial >
independent multinomial > hypergeometric — is asserted as well, since it must
hold for any data.

## Independent implementations

The two integrals this package shares with existing tools are checked against
those tools directly. `pingouin` is a validation-only dependency
(`pip install bayesplain[validate]`), never required at runtime.

| Quantity | Checked against | Agreement |
| --- | --- | --- |
| JZS *t*-test Bayes factor {cite:p}`rouder2009` | `pingouin.bayesfactor_ttest` | 6 significant figures, one- and two-sample, across prior widths |
| Correlation Bayes factor {cite:p}`ly2016` | `pingouin.bayesfactor_pearson` | 7 significant figures, across prior widths |

One difference is worth naming. `bayesplain` evaluates the correlation
marginal likelihood in log space, shifting by the peak of the integrand before
exponentiating. `pingouin` returns `inf` or `nan` for strongly correlated large
samples — including the roughly 2,400-tract case in the bundled ACS data, where
the true log Bayes factor is around 1,125. Both agree wherever both produce a
number.

## Self-validation

Some quantities can be pinned without any external reference, which is the
strongest form of check available:

- **Marginal likelihoods against brute-force integration.** The Beta-Binomial
  marginal likelihood is compared to `scipy.integrate.quad` over the same
  integral, and to the requirement that it sums to 1 across every possible
  number of successes.
- **The contingency Bayes factor against quadrature.** For a two-column table
  each marginal likelihood reduces to a one-dimensional integral evaluated to
  machine precision; for three columns it is checked by Monte Carlo integration
  over the Dirichlet priors.
- **The correlation sampling density against its own normalisation.** The exact
  density of an observed correlation must integrate to 1 over (−1, 1) for every
  ρ and *n*. It does, to 1 × 10⁻⁶, which pins the transcription of the
  hypergeometric expression without reference to anyone else's code.

The independent-multinomial Bayes factor is computed in a form that differs
algebraically from Equation 10 of {cite:t}`jamil2017` — each row drawing its own
column distribution, rather than the paper's successive conditioning. Expanding
both in gamma functions, every term involving Γ(*RCa*) and Γ(*y..* + *RCa*)
cancels and the two agree exactly. The advantage of the form used here is that
it accepts a per-column concentration vector, so a `Beta(a, b)` prior on a rate
carries into the Bayes factor unchanged.

## Behavioural properties

Invariances and monotonicities that must hold whatever the formula:

- posterior means fall between the prior mean and the maximum-likelihood
  estimate
- interval widths shrink roughly as 1/√*n*
- a wider prior always weakens the Bayes factor for a small observed effect
- swapping group order flips the sign of a difference and changes nothing else
- results are invariant to relabelling categories, and to reordering rows
- a stronger table prior pulls an association toward zero

## Pedagogical regression tests

Unusual, and the reason for them is practical: if a refactor breaks a lecture,
CI should say so rather than a student. Marked `pedagogy` and run by default.

- the eviction-style two-proportion example yields P ≈ 0.94 while the
  chi-square p-value exceeds 0.05, and the two intervals agree to within a
  percentage point
- the prior stops mattering as *n* grows, by at least a factor of five between
  *n* = 20 and *n* = 2000
- the credible interval for a mean coincides with the *t* interval to nine
  decimal places
- in the bundled ACS data, the income-rent correlation at tract level and at
  block-group level have non-overlapping credible intervals — the modifiable
  areal unit problem, with the arithmetic to prove it is not sampling noise
- in the bundled parcel data, the smallest group tops the ranking until
  `pool=True`, and then does not

## Running the checks

```bash
pytest                    # everything above except the slow Monte Carlo runs
pytest -m ''              # including those
pytest -m validation      # only the cross-implementation checks
pytest -m pedagogy        # only the numbers a lecture depends on
```

Full citations for everything referenced above are on the
{doc}`references <references>` page.
