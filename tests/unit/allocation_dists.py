"""The directional loss distributions of a random-allocation query, for tests.

The public API returns a ``dp_accounting`` PLD. Tests that inspect the remove or
the add distribution itself read the pair that the API hands to
``compose_full_pld``.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

import PLD_accounting.random_allocation_api as random_allocation_api_module
from PLD_accounting.discrete_dist import DenseDiscreteDist
from PLD_accounting.types import BoundType, Direction


def allocation_dists(
    *, pld_builder: Callable[..., Any], **kwargs: Any
) -> dict[Direction, DenseDiscreteDist]:
    """Return the remove and add distributions that ``pld_builder(**kwargs)`` composes.

    ``pld_builder`` is ``gaussian_allocation_pld`` or ``general_allocation_pld``.
    """
    composed: dict[Direction, DenseDiscreteDist] = {}

    def recording_compose_full_pld(
        *, remove_dist: DenseDiscreteDist, add_dist: DenseDiscreteDist, bound_type: BoundType
    ) -> None:
        del bound_type
        composed[Direction.REMOVE] = remove_dist
        composed[Direction.ADD] = add_dist

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            random_allocation_api_module, "compose_full_pld", recording_compose_full_pld
        )
        pld_builder(**kwargs)
    return composed
