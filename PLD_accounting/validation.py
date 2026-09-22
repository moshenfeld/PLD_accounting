"""Constraint primitives for public-package input checks.

Helpers are named by the constraint they enforce, not by a caller. Messages follow
``{name} must be {constraint}, got {value!r}``, or ``{type(value).__name__}`` for
type errors. Representation contracts live on the types that own them; operation
compatibility stays at the operation.

Scalar helpers accept either one ``value`` / ``name`` pair or parallel sequences of
values and names. Multi-input helpers are keyword-only.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from enum import Enum
from numbers import Integral, Real
from typing import TypeVar, cast, overload

import numpy as np
from numpy.typing import NDArray

E = TypeVar("E", bound=Enum)
R = TypeVar("R")
T = TypeVar("T")


# =============================================================================
# Scalars
# =============================================================================


@overload
def require_finite_real(*, value: object, name: str) -> float:
    ...


@overload
def require_finite_real(
    *, value: Sequence[object], name: list[str] | tuple[str, ...]
) -> list[float]:
    ...


def require_finite_real(*, value: object, name: str | Sequence[str]) -> float | list[float]:
    """Require a non-boolean finite real scalar, or a parallel sequence of them."""
    if not isinstance(name, str):
        return _require_each(checker=_finite_real, values=value, names=name)
    return _finite_real(value=value, name=name)


@overload
def require_positive_real(*, value: object, name: str) -> float:
    ...


@overload
def require_positive_real(
    *, value: Sequence[object], name: list[str] | tuple[str, ...]
) -> list[float]:
    ...


def require_positive_real(*, value: object, name: str | Sequence[str]) -> float | list[float]:
    """Require a finite real strictly greater than zero, or a parallel sequence of them."""
    if not isinstance(name, str):
        return _require_each(checker=_positive_real, values=value, names=name)
    return _positive_real(value=value, name=name)


@overload
def require_nonnegative_real(*, value: object, name: str) -> float:
    ...


@overload
def require_nonnegative_real(
    *, value: Sequence[object], name: list[str] | tuple[str, ...]
) -> list[float]:
    ...


def require_nonnegative_real(*, value: object, name: str | Sequence[str]) -> float | list[float]:
    """Require a finite real at least zero, or a parallel sequence of them."""
    if not isinstance(name, str):
        return _require_each(checker=_nonnegative_real, values=value, names=name)
    return _nonnegative_real(value=value, name=name)


@overload
def require_open_unit_interval(*, value: object, name: str) -> float:
    ...


@overload
def require_open_unit_interval(
    *, value: Sequence[object], name: list[str] | tuple[str, ...]
) -> list[float]:
    ...


def require_open_unit_interval(*, value: object, name: str | Sequence[str]) -> float | list[float]:
    """Require a finite real in ``(0, 1)``, or a parallel sequence of them."""
    if not isinstance(name, str):
        return _require_each(checker=_open_unit_interval, values=value, names=name)
    return _open_unit_interval(value=value, name=name)


@overload
def require_unit_interval_left_open(*, value: object, name: str) -> float:
    ...


@overload
def require_unit_interval_left_open(
    *, value: Sequence[object], name: list[str] | tuple[str, ...]
) -> list[float]:
    ...


def require_unit_interval_left_open(
    *, value: object, name: str | Sequence[str]
) -> float | list[float]:
    """Require a finite real in ``(0, 1]``, or a parallel sequence of them."""
    if not isinstance(name, str):
        return _require_each(checker=_unit_interval_left_open, values=value, names=name)
    return _unit_interval_left_open(value=value, name=name)


@overload
def require_closed_unit_interval(*, value: object, name: str, atol: float = 0.0) -> float:
    ...


@overload
def require_closed_unit_interval(
    *, value: Sequence[object], name: list[str] | tuple[str, ...], atol: float = 0.0
) -> list[float]:
    ...


def require_closed_unit_interval(
    *, value: object, name: str | Sequence[str], atol: float = 0.0
) -> float | list[float]:
    """Require a finite real in ``[0, 1]``, or a parallel sequence of them.

    Returns the value clamped to the interval, so callers take the coerced result
    rather than clamping again themselves.
    """
    atol = require_nonnegative_real(value=atol, name="atol")

    def _check(*, value: object, name: str) -> float:
        return _closed_unit_interval(value=value, name=name, atol=atol)

    if not isinstance(name, str):
        return _require_each(checker=_check, values=value, names=name)
    return _check(value=value, name=name)


@overload
def require_integer(*, value: object, name: str) -> int:
    ...


@overload
def require_integer(*, value: Sequence[object], name: list[str] | tuple[str, ...]) -> list[int]:
    ...


def require_integer(*, value: object, name: str | Sequence[str]) -> int | list[int]:
    """Require an integer scalar while rejecting booleans, or a parallel sequence of them."""
    if not isinstance(name, str):
        return _require_each(checker=_integer, values=value, names=name)
    return _integer(value=value, name=name)


@overload
def require_positive_int(*, value: object, name: str) -> int:
    ...


@overload
def require_positive_int(
    *, value: Sequence[object], name: list[str] | tuple[str, ...]
) -> list[int]:
    ...


def require_positive_int(*, value: object, name: str | Sequence[str]) -> int | list[int]:
    """Require an integer at least 1, or a parallel sequence of them."""
    if not isinstance(name, str):
        return _require_each(checker=_positive_int, values=value, names=name)
    return _positive_int(value=value, name=name)


@overload
def require_nonnegative_int(*, value: object, name: str) -> int:
    ...


@overload
def require_nonnegative_int(
    *, value: Sequence[object], name: list[str] | tuple[str, ...]
) -> list[int]:
    ...


def require_nonnegative_int(*, value: object, name: str | Sequence[str]) -> int | list[int]:
    """Require an integer at least 0, or a parallel sequence of them."""
    if not isinstance(name, str):
        return _require_each(checker=_nonnegative_int, values=value, names=name)
    return _nonnegative_int(value=value, name=name)


# =============================================================================
# Arrays / masses
# =============================================================================


def require_finite_array(*, values: NDArray[np.float64], name: str) -> None:
    """Require every entry in a numeric array to be finite."""
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} entries must be finite")


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


def require_type(*, value: object, expected_type: type[T], name: str) -> T:
    """Require ``value`` to be an instance of ``expected_type``."""
    if not isinstance(value, expected_type):
        raise TypeError(f"{name} must be {expected_type.__name__}, got {type(value).__name__}")
    return value


def require_enum(
    *,
    value: object,
    enum_cls: type[E],
    name: str,
    allowed: tuple[E, ...] | None = None,
) -> E:
    """Require an enum member, optionally restricted to ``allowed``."""
    if not isinstance(value, enum_cls):
        raise TypeError(f"{name} must be {enum_cls.__name__}, got {type(value).__name__}")
    if allowed is not None and value not in allowed:
        allowed_names = ", ".join(member.name for member in allowed)
        raise ValueError(f"{name} must be one of {{{allowed_names}}}, got {value!r}")
    return cast(E, value)


# =============================================================================
# Cross-field accounting counts
# =============================================================================


def require_allocation_counts(
    *, num_steps: object, num_selected: object, num_epochs: object
) -> None:
    """Require positive allocation counts with ``num_selected <= num_steps``."""
    steps, selected, _epochs = require_positive_int(
        value=[num_steps, num_selected, num_epochs],
        name=["num_steps", "num_selected", "num_epochs"],
    )
    if selected > steps:
        raise ValueError(f"num_selected ({selected}) cannot exceed num_steps ({steps})")


# =============================================================================
# Internal scalar checkers
# =============================================================================


def _require_each(
    *,
    checker: Callable[..., R],
    values: object,
    names: Sequence[str],
) -> list[R]:
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
    return [
        checker(value=value, name=name) for value, name in zip(value_list, name_list, strict=True)
    ]


def _finite_real(*, value: object, name: str) -> float:
    """Require a non-boolean finite real scalar."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return float(value)


def _positive_real(*, value: object, name: str) -> float:
    """Require a finite real strictly greater than zero."""
    number = _finite_real(value=value, name=name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return number


def _nonnegative_real(*, value: object, name: str) -> float:
    """Require a finite real at least zero."""
    number = _finite_real(value=value, name=name)
    if number < 0.0:
        raise ValueError(f"{name} must be nonnegative, got {value!r}")
    return number


def _open_unit_interval(*, value: object, name: str) -> float:
    """Require a finite real in ``(0, 1)``."""
    number = _finite_real(value=value, name=name)
    if not 0.0 < number < 1.0:
        raise ValueError(f"{name} must be in (0, 1), got {value!r}")
    return number


def _unit_interval_left_open(*, value: object, name: str) -> float:
    """Require a finite real in ``(0, 1]``."""
    number = _finite_real(value=value, name=name)
    if not 0.0 < number <= 1.0:
        raise ValueError(f"{name} must be in (0, 1], got {value!r}")
    return number


def _closed_unit_interval(*, value: object, name: str, atol: float) -> float:
    """Require a finite real in ``[0, 1]``, clamping overshoot within ``atol``.

    A boundary mass accumulated in floating point can land just outside the interval.
    Overshoot up to ``atol`` is a representation artifact and is clamped; anything
    beyond it is a real out-of-range value and raises.
    """
    number = _finite_real(value=value, name=name)
    if not -atol <= number <= 1.0 + atol:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")
    return min(max(number, 0.0), 1.0)


def _integer(*, value: object, name: str) -> int:
    """Require an integer scalar while rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    return int(value)


def _positive_int(*, value: object, name: str) -> int:
    """Require an integer at least 1."""
    number = _integer(value=value, name=name)
    if number < 1:
        raise ValueError(f"{name} must be >= 1, got {value!r}")
    return number


def _nonnegative_int(*, value: object, name: str) -> int:
    """Require an integer at least 0."""
    number = _integer(value=value, name=name)
    if number < 0:
        raise ValueError(f"{name} must be nonnegative, got {value!r}")
    return number
