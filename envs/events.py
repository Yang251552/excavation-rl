"""Domain randomization events for the excavation environment.

Implements parameter randomization applied at episode reset to improve
policy robustness and sim-to-real transfer. Randomization is applied
conditionally based on curriculum stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .excavation_env_cfg import DomainRandomizationCfg


@dataclass
class RandomizedParams:
    """Container for all randomized parameters in a single episode.

    Stores the sampled values so they can be logged and reproduced.
    """

    # Soil physics
    soil_density: float = 1600.0
    soil_friction: float = 0.6
    soil_restitution: float = 0.05
    soil_cohesion: float = 100.0

    # Soil geometry
    heap_height: float = 0.25
    heap_radius: float = 0.2
    heap_offset_x: float = 0.0
    heap_offset_y: float = 0.0

    # Robot
    joint_friction_scale: float = 1.0
    payload_mass: float = 0.0
    action_delay_steps: int = 0
    action_noise_std: float = 0.0

    # Sensors
    observation_noise_std: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "dr/soil_density": self.soil_density,
            "dr/soil_friction": self.soil_friction,
            "dr/soil_restitution": self.soil_restitution,
            "dr/soil_cohesion": self.soil_cohesion,
            "dr/heap_height": self.heap_height,
            "dr/heap_radius": self.heap_radius,
            "dr/heap_offset_x": self.heap_offset_x,
            "dr/heap_offset_y": self.heap_offset_y,
            "dr/joint_friction_scale": self.joint_friction_scale,
            "dr/payload_mass": self.payload_mass,
            "dr/action_delay_steps": float(self.action_delay_steps),
            "dr/action_noise_std": self.action_noise_std,
            "dr/observation_noise_std": self.observation_noise_std,
        }


class EventManager:
    """Manages domain randomization events at episode reset.

    Samples randomized parameters from configured ranges and applies
    them to the environment. Can be enabled/disabled per curriculum stage.
    """

    def __init__(self, cfg: DomainRandomizationCfg, seed: int = 42):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self._last_params: RandomizedParams | None = None

    @property
    def last_params(self) -> RandomizedParams | None:
        """Parameters from the most recent randomization."""
        return self._last_params

    def sample(self, override_enabled: bool | None = None) -> RandomizedParams:
        """Sample a new set of randomized parameters.

        Args:
            override_enabled: If not None, overrides cfg.enabled.

        Returns:
            RandomizedParams with sampled values.
        """
        enabled = override_enabled if override_enabled is not None else self.cfg.enabled

        if not enabled:
            # Return defaults (no randomization)
            params = RandomizedParams()
            self._last_params = params
            return params

        cfg = self.cfg

        params = RandomizedParams(
            # Soil physics
            soil_density=self.rng.uniform(*cfg.soil_density_range),
            soil_friction=self.rng.uniform(*cfg.soil_friction_range),
            soil_restitution=self.rng.uniform(*cfg.soil_restitution_range),
            soil_cohesion=self.rng.uniform(*cfg.soil_cohesion_range),

            # Soil geometry
            heap_height=self.rng.uniform(*cfg.heap_height_range),
            heap_radius=self.rng.uniform(*cfg.heap_radius_range),
            heap_offset_x=self.rng.uniform(*cfg.heap_position_offset_range),
            heap_offset_y=self.rng.uniform(*cfg.heap_position_offset_range),

            # Robot
            joint_friction_scale=self.rng.uniform(*cfg.joint_friction_scale_range),
            payload_mass=self.rng.uniform(*cfg.payload_mass_range),
            action_delay_steps=self.rng.integers(*cfg.action_delay_range, endpoint=True),
            action_noise_std=self.rng.uniform(*cfg.action_noise_std_range),

            # Sensors
            observation_noise_std=cfg.observation_noise_std,
        )

        self._last_params = params
        return params

    def apply_to_soil_properties(self, params: RandomizedParams):
        """Create modified SoilProperties from randomized params.

        Returns a new SoilProperties instance with the randomized values.
        """
        from soil.soil_properties import SoilProperties

        return SoilProperties(
            density=params.soil_density,
            friction_static=params.soil_friction,
            friction_dynamic=params.soil_friction * 0.7,
            restitution=params.soil_restitution,
            cohesion=params.soil_cohesion,
        )

    def apply_action_noise(
        self, action: np.ndarray, params: RandomizedParams
    ) -> np.ndarray:
        """Add randomized noise to action.

        Args:
            action: (num_joints,) raw action from policy.
            params: Current randomized parameters.

        Returns:
            Noisy action array.
        """
        if params.action_noise_std > 0:
            noise = self.rng.normal(0, params.action_noise_std, action.shape)
            return action + noise.astype(action.dtype)
        return action

    def apply_observation_noise(
        self, obs: np.ndarray, params: RandomizedParams
    ) -> np.ndarray:
        """Add randomized noise to observation.

        Args:
            obs: Observation vector.
            params: Current randomized parameters.

        Returns:
            Noisy observation array.
        """
        if params.observation_noise_std > 0:
            noise = self.rng.normal(0, params.observation_noise_std, obs.shape)
            return obs + noise.astype(obs.dtype)
        return obs

    def get_heap_config_overrides(self, params: RandomizedParams) -> dict:
        """Get heap geometry overrides from randomized params."""
        return {
            "height": params.heap_height,
            "radius": params.heap_radius,
            "offset_x": params.heap_offset_x,
            "offset_y": params.heap_offset_y,
        }
