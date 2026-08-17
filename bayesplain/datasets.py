"""Small, pre-cleaned planning datasets, shipped inside the package.

Every dataset here travels with the package rather than being downloaded. That
matters more than it looks: a problem set built on a live URL stops working the
month a city reorganises its open-data portal, and a class that spends its
first twenty minutes on data wrangling never gets to the statistics. These are
small, tidy, and permanent.

They are also real. Every one is an extract of a public dataset that the
accompanying course already uses, cleaned but not invented, so the numbers a
student reports are numbers about an actual place.

Loading requires pandas (``pip install bayesplain[data]``).

Examples
--------
>>> import bayesplain as bp
>>> bp.datasets.available()
['collisions', 'la_blockgroups', 'la_tracts', 'parcels']
>>> print(bp.datasets.describe("collisions"))   # doctest: +SKIP
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "available",
    "describe",
    "load",
    "load_collisions",
    "load_tracts",
    "load_blockgroups",
    "load_parcels",
]

_DATA_DIR = Path(__file__).parent / "data"


@dataclass(frozen=True)
class DatasetInfo:
    """What a shipped dataset contains and what it is good for.

    Attributes
    ----------
    name : str
        Loader key.
    rows : int
        Number of rows.
    summary : str
        One-line description.
    source : str
        Where the underlying data came from.
    columns : dict
        Column name to description.
    good_for : list of str
        Analyses this dataset supports well, as teaching examples.
    """

    name: str
    rows: int
    summary: str
    source: str
    columns: dict[str, str] = field(default_factory=dict)
    good_for: list[str] = field(default_factory=list)


DATASETS: dict[str, DatasetInfo] = {
    "collisions": DatasetInfo(
        name="collisions",
        rows=12_000,
        summary=(
            "Traffic collisions in Los Angeles and San Diego counties, "
            "2019-2023, one row per collision."
        ),
        source=(
            "California SWITRS, via github.com/oturns/example_datasets; "
            "a random sample of collisions falling inside either county."
        ),
        columns={
            "county": "Los Angeles or San Diego",
            "year": "year of the collision",
            "severity": "fatal, severe injury, other injury, complaint of "
            "pain, or property damage only",
            "road_surface": "dry, wet, snowy or icy, slippery",
            "alcohol_involved": "whether alcohol was recorded as involved",
            "pedestrian": "whether a pedestrian was involved",
            "bicycle": "whether a bicycle was involved",
            "hit_and_run": "whether the collision was a hit and run",
            "injured": "number of people injured",
            "killed": "number of people killed",
        },
        good_for=[
            "proportion — the share of collisions involving alcohol",
            "compare_proportions — alcohol involvement in the two counties, "
            "which is the two-group comparison the course pivots on",
            "contingency — severity against road surface",
            "compare_means — people injured per collision, by county",
        ],
    ),
    "la_tracts": DatasetInfo(
        name="la_tracts",
        rows=2_412,
        summary=(
            "American Community Survey estimates for Los Angeles County "
            "census tracts, 2021."
        ),
        source=("ACS 5-year estimates, via github.com/oturns/example_datasets."),
        columns={
            "geoid": "census tract identifier",
            "median_income": "median household income, dollars",
            "median_rent": "median contract rent, dollars",
            "median_home_value": "median home value, dollars",
            "pct_owner_occupied": "percent of units owner-occupied",
            "pct_multifamily": "percent of units in multi-unit structures",
            "pct_over_60": "percent of persons over 60",
            "pct_poverty": "percent of persons below the poverty line",
            "n_total_pop": "total population",
        },
        good_for=[
            "correlation — income against rent, at tract level",
            "correlation with aggregated=True, paired with la_blockgroups to "
            "show the same relationship changing with the geography",
            "mean — the average tract's median rent",
        ],
    ),
    "la_blockgroups": DatasetInfo(
        name="la_blockgroups",
        rows=5_361,
        summary=(
            "The same ACS variables as la_tracts, for Los Angeles County "
            "block groups — a finer geography over identical ground."
        ),
        source="ACS 5-year estimates, via github.com/oturns/example_datasets.",
        columns={
            "geoid": "census block group identifier",
            "median_income": "median household income, dollars",
            "median_rent": "median contract rent, dollars",
            "median_home_value": "median home value, dollars",
            "pct_owner_occupied": "percent of units owner-occupied",
            "pct_multifamily": "percent of units in multi-unit structures",
            "pct_over_60": "percent of persons over 60",
            "n_total_pop": "total population",
        },
        good_for=[
            "correlation — the modifiable areal unit problem, made concrete: "
            "run the same income-rent correlation here and on la_tracts and "
            "watch the number move without the underlying place changing"
        ],
    ),
    "parcels": DatasetInfo(
        name="parcels",
        rows=7_869,
        summary=(
            "Residential parcels in Baltimore with assessed values, size, "
            "age, and construction type."
        ),
        source=(
            "Maryland PropertyView assessor records, via "
            "github.com/oturns/example_datasets; residential parcels only, "
            "with implausible sizes and values trimmed."
        ),
        columns={
            "tract": "census tract identifier",
            "construction": "Brick, Frame, Siding, Stucco, Shingle Wood, "
            "Shingle Asbestos",
            "rowhouse": "whether the structure is a townhouse or rowhouse",
            "stories": "number of stories",
            "condition": "assessor's structure grade, binned",
            "year_built": "year the structure was built",
            "sqft": "structure square footage",
            "land_value": "assessed land value, dollars",
            "improvement_value": "assessed improvement value, dollars",
            "assessed_value": "land plus improvement value, dollars",
        },
        good_for=[
            "compare_means — assessed value for rowhouses against detached",
            "compare_groups — assessed value across construction types, "
            "where the rarest type looks extreme until it is pooled",
            "correlation — square footage against assessed value",
            "contingency — construction type against condition",
        ],
    ),
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def available() -> list[str]:
    """List the datasets shipped with the package.

    Returns
    -------
    list of str
        Names accepted by :func:`load` and :func:`describe`.
    """
    return sorted(DATASETS)


def describe(name: str | None = None) -> str:
    """Explain what a dataset holds and which analyses it suits.

    Parameters
    ----------
    name : str, optional
        Dataset name. ``None`` gives a one-line summary of each.

    Returns
    -------
    str
        Printable description.
    """
    if name is None:
        return "\n".join(
            f"{info.name:<16} {info.rows:>6,} rows   {info.summary}"
            for info in (DATASETS[k] for k in available())
        )
    info = _lookup(name)
    lines = [
        f"{info.name}  ({info.rows:,} rows)",
        "",
        info.summary,
        "",
        f"Source: {info.source}",
        "",
        "Columns",
        "-------",
    ]
    width = max(len(c) for c in info.columns)
    lines += [f"  {col:<{width}}  {desc}" for col, desc in info.columns.items()]
    lines += ["", "Good for", "--------"]
    lines += [f"  - {item}" for item in info.good_for]
    return "\n".join(lines)


def _lookup(name: str) -> DatasetInfo:
    key = str(name).strip().lower()
    if key not in DATASETS:
        raise ValueError(
            f"unknown dataset {name!r}. Available: {', '.join(available())}."
        )
    return DATASETS[key]


def load(name: str):
    """Load a shipped dataset by name.

    Parameters
    ----------
    name : str
        One of :func:`available`.

    Returns
    -------
    pandas.DataFrame
        The dataset.

    Raises
    ------
    ImportError
        If pandas is not installed.
    ValueError
        If the name is not recognised.
    """
    info = _lookup(name)
    try:
        import pandas as pd
    except ImportError as err:  # pragma: no cover - environment dependent
        raise ImportError(
            "loading the bundled datasets needs pandas. Install it with "
            "`pip install bayesplain[data]`."
        ) from err

    path = _DATA_DIR / f"{info.name}.csv.gz"
    if not path.exists():  # pragma: no cover - packaging failure
        raise FileNotFoundError(
            f"the {info.name} dataset is missing from the installed package "
            f"(expected at {path}). Reinstall bayesplain."
        )
    return pd.read_csv(path, compression="gzip")


def load_collisions():
    """Load traffic collisions for Los Angeles and San Diego counties.

    Returns
    -------
    pandas.DataFrame
        12,000 collisions, 2019-2023. See ``describe('collisions')``.

    Examples
    --------
    The week 4 comparison, on real data rather than an invented one:

    >>> import bayesplain as bp                             # doctest: +SKIP
    >>> df = bp.datasets.load_collisions()                  # doctest: +SKIP
    >>> counts = df.groupby("county").alcohol_involved.agg(["sum", "size"])
    ... # doctest: +SKIP
    """
    return load("collisions")


def load_tracts():
    """Load ACS estimates for Los Angeles County census tracts.

    Returns
    -------
    pandas.DataFrame
        2,412 tracts. See ``describe('la_tracts')``.
    """
    return load("la_tracts")


def load_blockgroups():
    """Load ACS estimates for Los Angeles County block groups.

    The companion to :func:`load_tracts`: same variables, same county, finer
    geography. Running an identical correlation on both is the cleanest
    demonstration of the modifiable areal unit problem available in one line
    of code.

    Returns
    -------
    pandas.DataFrame
        5,361 block groups. See ``describe('la_blockgroups')``.
    """
    return load("la_blockgroups")


def load_parcels():
    """Load Baltimore residential parcels with assessed values.

    Returns
    -------
    pandas.DataFrame
        7,869 parcels. See ``describe('parcels')``.
    """
    return load("parcels")
