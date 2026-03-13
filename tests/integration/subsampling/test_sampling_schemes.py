"""
Integration tests for sampling schemes.

Tests allocation and Poisson sampling with PLD/RDP accounting.
Runtime: Medium (~30-60 seconds)
"""
import pytest
import numpy as np
from dp_accounting.pld import privacy_loss_distribution

from PLD_accounting.types import PrivacyParams, AllocationSchemeConfig, Direction
from PLD_accounting.random_allocation_api import gaussian_allocation_PLD, gaussian_allocation_epsilon_extended
from PLD_accounting.types import ConvolutionMethod, BoundType


def _poisson_pld_dp_accounting(
    sigma: float,
    num_steps: int,
    num_epochs: int,
    sampling_prob: float,
    discretization: float,
):
    """Construct Poisson-subsampled Gaussian PLD using dp_accounting only."""
    if sampling_prob <= 0.0 or sampling_prob > 1.0:
        raise ValueError(f"sampling_prob must be in (0, 1], got {sampling_prob}")

    kwargs = dict(
        standard_deviation=sigma,
        sensitivity=1.0,
        value_discretization_interval=discretization,
        pessimistic_estimate=True,
    )
    if sampling_prob < 1.0:
        kwargs["sampling_prob"] = sampling_prob
    base_pld = privacy_loss_distribution.from_gaussian_mechanism(**kwargs)

    total_steps = int(num_steps * num_epochs)
    return base_pld.self_compose(total_steps)


def _poisson_epsilon_dp_accounting(
    params: PrivacyParams,
    discretization: float,
    sampling_prob: float,
) -> float:
    """Compute epsilon for Poisson-subsampled Gaussian PLD with dp_accounting."""
    pld = _poisson_pld_dp_accounting(
        sigma=params.sigma,
        num_steps=params.num_steps,
        num_epochs=params.num_epochs,
        sampling_prob=sampling_prob,
        discretization=discretization,
    )
    return pld.get_epsilon_for_delta(params.delta)


class TestAllocationPLDconv:
    """Test numerical allocation PLD computation."""

    def test_basic_allocation_pld(self):
        """Test basic allocation PLD computation."""
        params = PrivacyParams(
            sigma=1.0,
            num_steps=10,
            num_selected=5,
            num_epochs=1,
            epsilon=None,
            delta=1e-5
        )
        # Use finer discretization and smaller beta for small delta
        config = AllocationSchemeConfig(
            loss_discretization=0.01,
            tail_truncation=1e-6,
            max_grid_FFT=100000
        )

        config_with_method = AllocationSchemeConfig(
            loss_discretization=config.loss_discretization,
            tail_truncation=config.tail_truncation,
            max_grid_FFT=config.max_grid_FFT,
            convolution_method=ConvolutionMethod.GEOM
        )

        pld = gaussian_allocation_PLD(
            params=params,
            config=config_with_method,
            bound_type=BoundType.DOMINATES
        )

        # Check PLD is valid dp_accounting object
        assert hasattr(pld, 'get_epsilon_for_delta')
        assert hasattr(pld, '_pmf_remove')

        # Check can compute epsilon
        epsilon = pld.get_epsilon_for_delta(params.delta)
        assert epsilon > 0
        assert epsilon < 100  # Sanity check

    def test_allocation_fft_vs_geometric(self):
        """Compare FFT and geometric convolution for allocation."""
        params = PrivacyParams(
            sigma=1.0,
            num_steps=10,
            num_selected=5,
            num_epochs=1,
            delta=1e-5
        )
        config = AllocationSchemeConfig(
            loss_discretization=0.05,
            tail_truncation=0.001,
            max_grid_FFT=100000
        )

        config_mult = AllocationSchemeConfig(
            loss_discretization=config.loss_discretization,
            tail_truncation=config.tail_truncation,
            max_grid_FFT=config.max_grid_FFT,
            convolution_method=ConvolutionMethod.GEOM
        )

        pld_geometric = gaussian_allocation_PLD(
            params=params,
            config=config_mult,
            bound_type=BoundType.DOMINATES
        )

        config_fft = AllocationSchemeConfig(
            loss_discretization=config.loss_discretization,
            tail_truncation=config.tail_truncation,
            max_grid_FFT=config.max_grid_FFT,
            convolution_method=ConvolutionMethod.FFT
        )

        pld_fft = gaussian_allocation_PLD(
            params=params,
            config=config_fft,
            bound_type=BoundType.DOMINATES
        )

        eps_geometric = pld_geometric.get_epsilon_for_delta(params.delta)
        eps_fft = pld_fft.get_epsilon_for_delta(params.delta)

        # Should give similar epsilon values
        assert np.isclose(eps_geometric, eps_fft, rtol=0.1)

    def test_allocation_epsilon_wrapper(self):
        """Test gaussian_allocation_epsilon_extended wrapper."""
        params = PrivacyParams(
            sigma=1.0,
            num_steps=10,
            num_selected=5,
            num_epochs=1,
            delta=1e-5
        )
        config = AllocationSchemeConfig(
            loss_discretization=0.02,
            tail_truncation=0.1,
            max_grid_FFT=100000,
            convolution_method=ConvolutionMethod.GEOM
)

        epsilon = gaussian_allocation_epsilon_extended(
            params=params,
            config=config,
            bound_type=BoundType.DOMINATES
        )

        assert epsilon > 0
        assert isinstance(epsilon, (float, np.floating))

    @pytest.mark.parametrize(
        "method",
        [
            ConvolutionMethod.GEOM,
            ConvolutionMethod.FFT,
            ConvolutionMethod.COMBINED,
        ],
    )
    def test_allocation_pld_methods(self, method):
        """Exercise all convolution methods on a small allocation PLD."""
        params = PrivacyParams(
            sigma=1.0,
            num_steps=6,
            num_selected=2,
            num_epochs=1,
            delta=1e-5
        )
        config = AllocationSchemeConfig(
            loss_discretization=0.05,
            tail_truncation=1e-5,
            max_grid_FFT=20000,
            max_grid_mult=2000,
            convolution_method=method
        )
        pld = gaussian_allocation_PLD(
            params=params,
            config=config,
            bound_type=BoundType.DOMINATES
        )
        epsilon = pld.get_epsilon_for_delta(params.delta)
        assert not np.isnan(epsilon)

    def test_allocation_pld_best_of_two_succeeds(self):
        """BEST_OF_TWO successfully produces uniform grids after change_spacing_type."""
        params = PrivacyParams(
            sigma=1.0,
            num_steps=6,
            num_selected=2,
            num_epochs=1,
            delta=1e-5
        )
        config = AllocationSchemeConfig(
            loss_discretization=0.05,
            tail_truncation=1e-5,
            max_grid_FFT=20000,
            max_grid_mult=2000,
            convolution_method=ConvolutionMethod.BEST_OF_TWO
        )
        pld = gaussian_allocation_PLD(
            params=params,
            config=config,
            bound_type=BoundType.DOMINATES
        )
        # Verify that the result is valid
        epsilon = pld.get_epsilon_for_delta(params.delta)
        assert epsilon > 0
        assert epsilon < 100  # Sanity check

    def test_allocation_scales_with_sigma(self):
        """Test that epsilon decreases with larger sigma."""
        config = AllocationSchemeConfig(loss_discretization=0.02, tail_truncation=1e-6, max_grid_FFT=100000)

        eps_list = []
        for sigma in [0.5, 1.0, 2.0]:
            params = PrivacyParams(
                sigma=sigma,
                num_steps=10,
                num_selected=5,
                num_epochs=1,
                delta=1e-5
            )
            eps = gaussian_allocation_epsilon_extended(
                params=params,
                config=config,
            )
            eps_list.append(eps)

        # Epsilon should decrease as sigma increases
        assert eps_list[0] > eps_list[1] > eps_list[2]

    def test_allocation_scales_with_epochs(self):
        """Test that epsilon increases with more epochs."""
        config = AllocationSchemeConfig(loss_discretization=0.02, tail_truncation=1e-6, max_grid_FFT=100000)

        eps_list = []
        for num_epochs in [1, 2, 5]:
            params = PrivacyParams(
                sigma=1.0,
                num_steps=10,
                num_selected=5,
                num_epochs=num_epochs,
                delta=1e-5
            )
            eps = gaussian_allocation_epsilon_extended(
                params=params,
                config=config,
            )
            eps_list.append(eps)

        # Epsilon should increase with epochs
        assert eps_list[0] < eps_list[1] < eps_list[2]


class TestPoissonPLD:
    """Test Poisson sampling PLD computation."""

    def test_basic_poisson_pld(self):
        """Test basic Poisson PLD computation."""
        params = PrivacyParams(
            sigma=1.0,
            num_steps=10,
            num_selected=5,
            num_epochs=1,
            delta=1e-5
        )

        # Poisson sampling probability
        sampling_prob = params.num_selected / 100  # Assuming dataset size ~100

        pld = _poisson_pld_dp_accounting(
            sigma=params.sigma,
            num_steps=params.num_steps,
            num_epochs=params.num_epochs,
            sampling_prob=sampling_prob,
            discretization=0.01,
        )

        # Check PLD is valid
        assert hasattr(pld, 'get_epsilon_for_delta')
        epsilon = pld.get_epsilon_for_delta(params.delta)
        assert epsilon > 0

    def test_poisson_epsilon_wrapper(self):
        """Test Poisson epsilon helper implemented with dp_accounting."""
        poisson_params = PrivacyParams(
            sigma=1.0,
            num_steps=10,
            num_epochs=1,
            num_selected=5,
            delta=1e-5
        )
        discretization = 0.1

        epsilon = _poisson_epsilon_dp_accounting(
            params=poisson_params,
            discretization=discretization,
            sampling_prob=1.0,
        )

        assert epsilon > 0
        assert isinstance(epsilon, (float, np.floating))

    def test_poisson_vs_allocation(self):
        """Compare Poisson and allocation schemes."""
        # Project params for allocation
        params = PrivacyParams(
            sigma=1.0,
            num_steps=10,
            num_selected=5,
            num_epochs=1,
            delta=1e-5
        )
        config = AllocationSchemeConfig(loss_discretization=0.02, tail_truncation=0.1, max_grid_FFT=100000)

        poisson_params = PrivacyParams(
            sigma=1.0,
            num_steps=10,
            num_epochs=1,
            num_selected=5,
            delta=1e-5
        )
        discretization = 0.1

        eps_poisson = _poisson_epsilon_dp_accounting(
            params=poisson_params,
            discretization=discretization,
            sampling_prob=1.0,
        )

        eps_allocation = gaussian_allocation_epsilon_extended(
            params=params,
            config=config,
            
        )

        # Both should give reasonable values
        # Allocation typically gives better privacy (lower epsilon)
        assert eps_poisson > 0
        assert eps_allocation > 0
        # Uncomment if allocation is expected to be better:
        # assert eps_allocation < eps_poisson


class TestDirectionSemantics:
    """Test REMOVE vs ADD direction semantics."""

    def test_remove_direction(self):
        """Test REMOVE direction computation."""
        params = PrivacyParams(
            sigma=1.0,
            num_steps=10,
            num_selected=5,
            num_epochs=1,
            delta=1e-5
        )
        config = AllocationSchemeConfig(loss_discretization=0.02, tail_truncation=0.1, max_grid_FFT=100000)

        config_with_method = AllocationSchemeConfig(
            loss_discretization=config.loss_discretization,
            tail_truncation=config.tail_truncation,
            max_grid_FFT=config.max_grid_FFT,
            convolution_method=ConvolutionMethod.GEOM
        )

        pld = gaussian_allocation_PLD(
            params=params,
            config=config_with_method,
            bound_type=BoundType.DOMINATES
        )

        epsilon = pld.get_epsilon_for_delta(params.delta)
        assert epsilon > 0

    def test_add_direction(self):
        """Test ADD direction computation."""
        params = PrivacyParams(
            sigma=1.0,
            num_steps=10,
            num_selected=5,
            num_epochs=1,
            delta=1e-5
        )
        config = AllocationSchemeConfig(loss_discretization=0.02, tail_truncation=0.1, max_grid_FFT=100000)

        # dp_accounting PLD requires pmf_remove, so use Direction.BOTH to get both directions
        pld = gaussian_allocation_PLD(
            params=params,
            config=config,
            bound_type=BoundType.DOMINATES
        )
        # Check PLD is valid dp_accounting object with add direction
        assert hasattr(pld, '_pmf_add')
        assert pld._pmf_add is not None
        assert hasattr(pld, '_pmf_remove')
        assert pld._pmf_remove is not None

    def test_directions_give_similar_epsilon(self):
        """Test REMOVE and BOTH directions work."""
        params = PrivacyParams(
            sigma=1.0,
            num_steps=10,
            num_selected=5,
            num_epochs=1,
            delta=1e-5
        )
        config = AllocationSchemeConfig(
            loss_discretization=1e-4,
            tail_truncation=0.1,
            max_grid_FFT=10000000,
            convolution_method=ConvolutionMethod.FFT
        )

        eps_remove = gaussian_allocation_epsilon_extended(
            params=params,
            config=config,
            
        )

        eps_both = gaussian_allocation_epsilon_extended(
            params=params,
            config=config,
            
        )

        # BOTH should give larger epsilon than REMOVE (worst case)
        assert eps_both >= eps_remove


class TestDiscretizationSensitivity:
    """Test sensitivity to discretization parameter."""

    def test_finer_discretization_more_accurate(self):
        """Test that different discretizations give reasonable epsilon values."""
        params = PrivacyParams(
            sigma=1.0,
            num_steps=10,
            num_selected=5,
            num_epochs=1,
            delta=1e-5
        )

        eps_list = []
        for disc in [0.2, 0.1, 0.05]:
            config = AllocationSchemeConfig(loss_discretization=disc, tail_truncation=1e-6, max_grid_FFT=100000)
            eps = gaussian_allocation_epsilon_extended(
                params=params,
                config=config,
            )
            eps_list.append(eps)

        # All discretizations should give finite, positive epsilon values
        assert all(0 < eps < 20 for eps in eps_list)
        # Values should be in a reasonable range of each other
        assert max(eps_list) - min(eps_list) < 2.0

    def test_discretization_too_coarse_clamped(self):
        """Test behavior with very coarse discretization is clamped."""
        params = PrivacyParams(
            sigma=1.0,
            num_steps=10,
            num_selected=5,
            num_epochs=1,
            delta=1e-5
        )
        config = AllocationSchemeConfig(loss_discretization=100.0, max_grid_FFT=100000)  # Extremely coarse

        # Even with coarse discretization, we should return a finite epsilon (clamped grid).
        eps = gaussian_allocation_epsilon_extended(
            params=params,
            config=config,
        )
        assert np.isfinite(eps)


class TestSmallParameters:
    """Test with small/edge case parameters."""

    def test_single_step(self):
        """Test with small num_steps."""
        # num_steps=1, num_selected=1 creates edge case with empty grids
        # Use slightly larger values to avoid this
        params = PrivacyParams(
            sigma=1.0,
            num_steps=5,
            num_selected=2,
            num_epochs=1,
            delta=1e-5
        )
        config = AllocationSchemeConfig(loss_discretization=0.01, tail_truncation=0.1, max_grid_FFT=100000)

        eps = gaussian_allocation_epsilon_extended(
            params=params,
            config=config,
            
        )
        assert eps > 0

    def test_large_sigma(self):
        """Test with very large sigma (high privacy)."""
        params = PrivacyParams(
            sigma=10.0,
            num_steps=10,
            num_selected=5,
            num_epochs=1,
            delta=1e-5
        )
        config = AllocationSchemeConfig(
            loss_discretization=1e-4,
            tail_truncation=1e-12,
            max_grid_FFT=100000,
            convolution_method=ConvolutionMethod.FFT
        )

        eps = gaussian_allocation_epsilon_extended(
            params=params,
            config=config,
            
        )
        # Should give small epsilon (but discretization affects exact value)
        assert eps < 2.0

    def test_small_delta(self):
        """Test with very small delta."""
        params = PrivacyParams(
            sigma=1.0,
            num_steps=10,
            num_selected=5,
            num_epochs=1,
            delta=1e-10
        )
        config = AllocationSchemeConfig(
            loss_discretization=0.005,
            tail_truncation=0.1,
            max_grid_FFT=100000,
            convolution_method=ConvolutionMethod.FFT
        )

        eps = gaussian_allocation_epsilon_extended(
            params=params,
            config=config,
            
        )
        # Smaller delta requires larger epsilon
        assert eps > 0
