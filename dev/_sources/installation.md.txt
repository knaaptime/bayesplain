# Installation

```bash
pip install bayesplain
```

That is numpy and scipy and nothing else. There is no compiler step, no MCMC
backend, and no optional accelerator that you will later discover was
mandatory. In a Colab cell it finishes in seconds.

## Extras

```bash
pip install "bayesplain[plot]"      # matplotlib, for .plot()
pip install "bayesplain[data]"      # pandas, for the bundled datasets
pip install "bayesplain[plot,data]" # the usual classroom combination
```

The extras are genuinely optional. Every analysis, every summary, every
interval and Bayes factor works without them; you lose `.plot()` and
`bayesplain.datasets` and nothing else.

## Development install

```bash
git clone https://github.com/knaaptime/bayesplain
cd bayesplain
pip install -e ".[tests,plot,data,validate]"

pytest                  # fast suite, including doctests
pytest -m ''            # add the slow Monte Carlo validation runs
ruff check . && ruff format --check .
```

The `validate` extra pulls in `pingouin`, which several tests cross-check
against. Those tests skip cleanly when it is absent, so it is never needed to
run the suite.

## Building the docs

```bash
pip install -e ".[docs]"
cd docs && make html
```

## Why the dependency list is this short

It is a design constraint rather than an accident, and it does real work.
Everything in the package is a conjugate posterior, a closed-form Bayes factor,
or a one-dimensional integral — so anything that would tempt the package toward
a sampler is out of scope by construction rather than by discipline. When a
question genuinely needs MCMC, the documentation says so and points at
[Bambi](https://bambinos.github.io/bambi/) rather than growing a half-hidden
sampler of its own.

The classroom benefit is more mundane and just as real: no student loses an
afternoon to a build failure, and nobody ever sees a convergence warning they
have not been taught to interpret.
