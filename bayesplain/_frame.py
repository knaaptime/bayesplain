"""Reading columns out of a data frame by name.

Students meet data as a table long before they meet it as arrays, so every
analysis that takes samples also accepts column names plus ``data=``. Nothing
here imports pandas: anything that can be indexed by a column name and whose
columns have ``.to_numpy()`` works.
"""

from __future__ import annotations

import numpy as np

__all__ = ["column", "split_by"]


def column(value, data, role: str = "x"):
    """Resolve ``value`` to an array, looking it up in ``data`` if it is a name.

    Parameters
    ----------
    value : str or array_like
        A column name, or the values themselves.
    data : data frame or None
        Where to look names up.
    role : str
        Which argument this is, for error messages.

    Returns
    -------
    tuple
        ``(values, name)``, where ``name`` is the column name or ``None``.
    """
    if not isinstance(value, str):
        if data is not None and value is not None:
            raise ValueError(
                f"data= was given, so {role} should be a column name, not values."
            )
        return value, None
    if data is None:
        raise ValueError(
            f"{role}={value!r} looks like a column name, but no data= was given "
            "to look it up in. Pass data=your_frame, or pass the values "
            "themselves."
        )
    try:
        series = data[value]
    except (KeyError, IndexError, TypeError) as err:
        columns = ", ".join(repr(c) for c in list(getattr(data, "columns", []))[:12])
        raise ValueError(
            f"no column named {value!r} in data. Columns include: {columns}."
        ) from err
    values = series.to_numpy() if hasattr(series, "to_numpy") else series
    return np.asarray(values), value


def split_by(values, keys, name: str | None = None):
    """Split one column of values into groups by a second column of labels.

    Parameters
    ----------
    values : array_like
        The observations.
    keys : array_like
        One group label per observation. Rows with a missing label are
        dropped.
    name : str, optional
        The label column's name. Used to name groups whose labels are bare
        booleans, so that ``rowhouse=True`` reads as more than ``True``.

    Returns
    -------
    tuple
        ``(names, samples)``, with groups in sorted order of their labels so
        that the same data always produce the same orientation.
    """
    values = np.asarray(values).ravel()
    keys = np.asarray(keys).ravel()
    if values.size != keys.size:
        raise ValueError(
            f"there are {values.size} values but {keys.size} group labels; they "
            "must line up one to one."
        )
    present = np.array([k is not None and k == k for k in keys], dtype=bool)
    levels = sorted(set(keys[present].tolist()), key=lambda k: (str(type(k)), k))
    names = []
    for level in levels:
        if isinstance(level, (bool, np.bool_)) and name:
            names.append(f"{name}={level}")
        else:
            names.append(str(level))
    samples = [values[present & (keys == level)] for level in levels]
    return names, samples
