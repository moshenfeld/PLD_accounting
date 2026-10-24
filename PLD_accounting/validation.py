"""Constraint primitives for public-package input checks.

Helpers are named by the constraint they enforce, not by a caller. Messages follow
``{name} must be {constraint}, got {value!r}``, or ``{type(value).__name__}`` for
type errors. Representation contracts live on the types that own them; operation
compatibility stays at the operation.

Scalar helpers accept either one ``value`` / ``name`` pair or parallel sequences of
values and names. Multi-input helpers are keyword-only. All validators return
``None``. Checked casting functions return the original object with a narrowed
type and exist only where a caller's static type narrows; numeric conversions and
repairs are explicit.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from enum import Enum
from numbers import Integral, Real
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

E = TypeVar("E", bound=Enum)
T = TypeVar("T")


# =============================================================================
# Scalars
# =============================================================================


def require_finite_real(*, value: object, name: str | Sequence[str]) -> None:
    """Require a non-boolean finite real scalar, or a parallel sequence of them."""
    if not isinstance(name, str):
        _require_each(checker=_cast_finite_real, values=value, names=name)
    else:
        _cast_finite_real(value=value, name=name)


def require_positive_real(*, value: object, name: str | Sequence[str]) -> None:
    """Require a finite real strictly greater than zero, or a parallel sequence of them."""
    if not isinstance(name, str):
        _require_each(checker=_positive_real, values=value, names=name)
    else:
        _positive_real(value=value, name=name)


def require_nonnegative_real(*, value: object, name: str | Sequence[str]) -> None:
    """Require a finite real at least zero, or a parallel sequence of them."""
    if not isinstance(name, str):
        _require_each(checker=_nonnegative_real, values=value, names=name)
    else:
        _nonnegative_real(value=value, name=name)


def require_open_unit_interval(*, value: object, name: str | Sequence[str]) -> None:
    """Require a finite real in ``(0, 1)``, or a parallel sequence of them."""
    if not isinstance(name, str):
        _require_each(checker=_open_unit_interval, values=value, names=name)
    else:
        _open_unit_interval(value=value, name=name)


def require_unit_interval_left_open(*, value: object, name: str | Sequence[str]) -> None:
    """Require a finite real in ``(0, 1]``, or a parallel sequence of them."""
    if not isinstance(name, str):
        _require_each(checker=_unit_interval_left_open, values=value, names=name)
    else:
        _unit_interval_left_open(value=value, name=name)


def require_closed_unit_interval(*, value: object, name: str | Sequence[str]) -> None:
    """Require a finite real in ``[0, 1]``, or a parallel sequence of them."""
    if not isinstance(name, str):
        _require_each(checker=_closed_unit_interval, values=value, names=name)
    else:
        _closed_unit_interval(value=value, name=name)


def require_integer(*, value: object, name: str | Sequence[str]) -> None:
    """Require an integer scalar while rejecting booleans, or a parallel sequence of them."""
    if not isinstance(name, str):
        _require_each(checker=_cast_integer, values=value, names=name)
    else:
        _cast_integer(value=value, name=name)


def require_positive_int(*, value: object, name: str | Sequence[str]) -> None:
    """Require an integer at least 1, or a parallel sequence of them."""
    if not isinstance(name, str):
        _require_each(checker=_positive_int, values=value, names=name)
    else:
        _positive_int(value=value, name=name)


def require_nonnegative_int(*, value: object, name: str | Sequence[str]) -> None:
    """Require an integer at least 0, or a parallel sequence of them."""
    if not isinstance(name, str):
        _require_each(checker=_nonnegative_int, values=value, names=name)
    else:
        _nonnegative_int(value=value, name=name)


# =============================================================================
# Arrays / masses
# =============================================================================


def require_finite_array(*, values: NDArray[np.float64], name: str) -> None:
    """Require every entry in a numeric array to be finite."""
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} entries must be finite")


def require_increasing_array(*, values: NDArray[np.float64], name: str) -> None:
    """Require a one-dimensional, finite, strictly increasing array of at least two entries."""
    array = np.asarray(values)
    if array.ndim != 1 or array.size < 2:
        raise ValueError(f"{name} must be one-dimensional with at least two entries")
    require_finite_array(values=array, name=name)
    if not np.all(np.diff(array) > 0.0):
        raise ValueError(f"{name} must be strictly increasing")


def require_nonnegative_masses(
    *,
    prob_arr: NDArray[np.float64],
    p_min: float,
    p_max: float,
) -> None:
    """Require a 1-D nonnegative PMF and nonnegative boundary masses."""
    prob_arr = np.asarray(prob_arr, dtype=np.float64)
    if prob_arr.ndim != 1:
        raise ValueError("PMF must be 1-D array")
    if prob_arr.size == 0:
        raise ValueError("PMF must contain at least one finite-support bin")
    require_finite_array(values=prob_arr, name="PMF")
    if np.any(prob_arr < 0.0):
        raise ValueError("PMF must be nonnegative")
    require_nonnegative_real(value=[p_min, p_max], name=["p_min", "p_max"])


# =============================================================================
# Types / enums
# =============================================================================


def cast_type(*, value: object, expected_type: type[T], name: str) -> T:
    """Check ``expected_type`` and return the original object with that type.

    For a caller whose static type narrows; an argument is never checked against its own class.
    """
    if not isinstance(value, expected_type):
        raise TypeError(f"{name} must be {expected_type.__name__}, got {type(value).__name__}")
    return value


def require_enum(
    *,
    value: object,
    enum_cls: type[E],
    name: str,
    allowed: tuple[E, ...] | None = None,
) -> None:
    """Require an enum member, optionally restricted to ``allowed``."""
    if not isinstance(value, enum_cls):
        raise TypeError(f"{name} must be {enum_cls.__name__}, got {type(value).__name__}")
    if allowed is not None and value not in allowed:
        allowed_names = ", ".join(member.name for member in allowed)
        raise ValueError(f"{name} must be one of {{{allowed_names}}}, got {value!r}")


# =============================================================================
# Cross-field accounting counts
# =============================================================================


def require_allocation_counts(
    *, num_steps: object, num_selected: object, num_epochs: object
) -> None:
    """Require positive allocation counts with ``num_selected <= num_steps``."""
    require_positive_int(
        value=[num_steps, num_selected, num_epochs],
        name=["num_steps", "num_selected", "num_epochs"],
    )
    steps = _cast_integer(value=num_steps, name="num_steps")
    selected = _cast_integer(value=num_selected, name="num_selected")
    if selected > steps:
        raise ValueError(f"num_selected ({selected}) cannot exceed num_steps ({steps})")


# =============================================================================
# Internal scalar checkers
# =============================================================================


def _require_each(
    *,
    checker: Callable[..., object],
    values: object,
    names: Sequence[str],
) -> None:
    """Apply a keyword-only scalar ``value`` / ``name`` checker to parallel sequences."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("values must be a sequence when names is a sequence")
    value_list = list(values)
    name_list = list(names)
    if len(value_list) != len(name_list):
        raise ValueError(
            f"values and names must have equal length, got {len(value_list)} and {len(name_list)}"
        )
    if not name_list:
        raise ValueError("names must be non-empty")
    for name in name_list:
        if not isinstance(name, str):
            raise TypeError(f"each name must be a str, got {type(name).__name__}")
    for value, name in zip(value_list, name_list, strict=True):
        checker(value=value, name=name)


def _cast_finite_real(*, value: object, name: str) -> float:
    """Check a non-boolean finite real scalar and convert it to a float."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _positive_real(*, value: object, name: str) -> None:
    """Require a finite real strictly greater than zero."""
    number = _cast_finite_real(value=value, name=name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def _nonnegative_real(*, value: object, name: str) -> None:
    """Require a finite real at least zero."""
    number = _cast_finite_real(value=value, name=name)
    if number < 0.0:
        raise ValueError(f"{name} must be nonnegative, got {value!r}")


def _open_unit_interval(*, value: object, name: str) -> None:
    """Require a finite real in ``(0, 1)``."""
    number = _cast_finite_real(value=value, name=name)
    if not 0.0 < number < 1.0:
        raise ValueError(f"{name} must be in (0, 1), got {value!r}")


def _unit_interval_left_open(*, value: object, name: str) -> None:
    """Require a finite real in ``(0, 1]``."""
    number = _cast_finite_real(value=value, name=name)
    if not 0.0 < number <= 1.0:
        raise ValueError(f"{name} must be in (0, 1], got {value!r}")


def _closed_unit_interval(*, value: object, name: str) -> None:
    """Require a finite real in ``[0, 1]``."""
    number = _cast_finite_real(value=value, name=name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")


def _cast_integer(*, value: object, name: str) -> int:
    """Check an integer scalar, reject booleans, and convert it to a Python integer."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    return int(value)


def _positive_int(*, value: object, name: str) -> None:
    """Require an integer at least 1."""
    number = _cast_integer(value=value, name=name)
    if number < 1:
        raise ValueError(f"{name} must be >= 1, got {value!r}")


def _nonnegative_int(*, value: object, name: str) -> None:
    """Require an integer at least 0."""
    number = _cast_integer(value=value, name=name)
    if number < 0:
        raise ValueError(f"{name} must be nonnegative, got {value!r}")
