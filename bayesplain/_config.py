"""Package-wide defaults, chosen for a classroom rather than for a lab.

The seed is fixed by default. That is unusual for a research package and
correct for a teaching one: if every student in the room gets different digits
in the last decimal place, office hours fill up with questions about Monte
Carlo noise instead of questions about statistics.

Having hidden that noise, the package then surfaces it deliberately -- every
summary of a sampled quantity prints its Monte Carlo standard error, and one
short lesson on why the last digit wobbles costs less than a term of
confusion.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "DEFAULT_SEED",
    "DEFAULT_DRAWS",
    "get_seed",
    "set_seed",
    "get_draws",
    "set_draws",
    "make_rng",
]

DEFAULT_SEED = 12345
DEFAULT_DRAWS = 100_000

_state = {"seed": DEFAULT_SEED, "draws": DEFAULT_DRAWS}


def get_seed() -> int | None:
    """Return the current default seed.

    Returns
    -------
    int or None
        The seed used when a call does not supply its own. ``None`` means
        draws are non-reproducible.
    """
    return _state["seed"]


def set_seed(seed: int | None) -> None:
    """Set the default seed for every subsequent analysis.

    Parameters
    ----------
    seed : int or None
        Seed to use. Pass ``None`` for genuinely random draws, which is the
        right choice when demonstrating that Monte Carlo error is real.
    """
    if seed is not None:
        seed = int(seed)
    _state["seed"] = seed


def get_draws() -> int:
    """Return the current default number of posterior draws.

    Returns
    -------
    int
        Draws taken for quantities that require sampling.
    """
    return _state["draws"]


def set_draws(n_draws: int) -> None:
    """Set the default number of posterior draws.

    Parameters
    ----------
    n_draws : int
        Must be at least 1000; below that, Monte Carlo error starts showing up
        in the second decimal place of a reported probability.
    """
    n_draws = int(n_draws)
    if n_draws < 1000:
        raise ValueError(
            f"n_draws must be at least 1000, got {n_draws}. Below that, Monte "
            "Carlo error is large enough to change a reported probability."
        )
    _state["draws"] = n_draws


def make_rng(seed: int | None = "unset") -> np.random.Generator:
    """Build a random generator, honouring the package default.

    Parameters
    ----------
    seed : int, None, or 'unset'
        An explicit seed, ``None`` for non-reproducible draws, or the sentinel
        ``'unset'`` to fall back to the package default.

    Returns
    -------
    numpy.random.Generator
        A fresh generator.
    """
    if isinstance(seed, str) and seed == "unset":
        seed = _state["seed"]
    return np.random.default_rng(seed)
