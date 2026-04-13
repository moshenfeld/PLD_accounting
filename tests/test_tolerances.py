"""
Centralized tolerance and threshold configuration for all tests.

This module defines strict tolerances used across the test suite to ensure
consistency and maximum accuracy.

ALL TESTS USE STRICT TOLERANCES ONLY.

Usage:
    from tests.test_tolerances import TestTolerances as TOL

    assert abs(actual_mass - 1.0) < TOL.MASS_CONSERVATION
    assert actual_mean >= TOL.MEAN_ACCURACY * expected_mean
"""


class TestTolerances:
    """Strict tolerances for test assertions across the entire test suite.

    All values are tuned for maximum accuracy with large grids and minimal truncation.
    """

    def __repr__(self) -> str:
        return "TestTolerances"

    def __str__(self) -> str:
        return "TestTolerances"

    # ========================================================================
    # MASS CONSERVATION (Probability sums to 1.0)
    # ========================================================================

    MASS_CONSERVATION = 1e-10
    """Mass conservation tolerance: 1e-10

    Achieved with:
    - Large grids (10000 points)
    - Very small beta (1e-12)
    - Fine discretization (1e-4)

    Used for all mass conservation tests (analytical and numerical).
    """

    # ========================================================================
    # DISTRIBUTION COMPARISON (PMF/PDF point-wise comparison)
    # ========================================================================

    DISTRIBUTION_COMPARISON = 1e-10
    """Distribution comparison tolerance: 1e-10

    Used for comparing computed distributions vs analytical/reference.
    """

    # ========================================================================
    # MEAN/MOMENT ACCURACY (Statistical moments)
    # ========================================================================

    MEAN_ACCURACY = 0.98
    """Mean accuracy: actual >= 98% of expected

    Achieved with:
    - Large grid (10000 points)
    - Very small beta (1e-12)
    - Fine discretization (1e-4)

    Works even for T=16 convolutions.
    """

    # ========================================================================
    # EPSILON/DELTA ACCURACY (Privacy parameters)
    # ========================================================================

    EPSILON_ABSOLUTE = 0.01
    """Absolute tolerance for epsilon differences."""

    EPSILON_RELATIVE = 0.05
    """Relative tolerance for epsilon: 5%

    Used for mean value comparisons in convolution tests.
    """

    DELTA_ABSOLUTE = 1e-10
    """Absolute tolerance for delta differences."""

    # ========================================================================
    # GRID CONFIGURATION (For accuracy tests)
    # ========================================================================

    GRID_SIZE = 10000
    """Large grid for strict accuracy."""

    BETA = 1e-12
    """Very small beta (virtually no truncation)."""

    DISCRETIZATION = 1e-4
    """Fine discretization."""

    # ========================================================================
    # GRID / SPACING (aligned with PLD_accounting.distribution_utils; defined here
    # so tests do not import spacing tolerances from production code.)
    # ========================================================================

    SPACING_ATOL = 1e-12
    SPACING_RTOL = 1e-6

    # ========================================================================
    # TIGHT ARRAY & COUPLED-DISTRIBUTION CHECKS
    # ========================================================================

    ARRAY_RTOL_ULTRA = 1e-12
    PMF_COUPLED_RTOL = 1e-3
    PMF_COUPLED_ATOL = 5e-4
    INF_MASS_RTOL = 1e-9
    STOCHASTIC_DOM_SLACK = 1e-6
    NEG_INF_STRICT_LT = 1e-6
    PMF_NONNEGATIVE_SLACK = 1e-15
    TAIL_LINEAR_RELATION_ATOL = 1e-24
    GRID_EXACT_RTOL = 0.0
    GRID_EXACT_ATOL = 0.0
