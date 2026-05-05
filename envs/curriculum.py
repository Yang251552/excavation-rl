"""Curriculum learning manager for progressive difficulty scaling.

Manages training stages that gradually increase task difficulty:
  Stage 1: Few particles, close heap, no DR, strong guidance rewards
  Stage 2: More particles, medium distance, light DR
  Stage 3: Full particles, random placement, full DR
"""

from __future__ import annotations

from dataclasses import dataclass

from .excavation_env_cfg import CurriculumCfg, CurriculumStageCfg


@dataclass
class CurriculumState:
    """Tracks curriculum progression."""

    total_steps: int = 0
    current_stage_idx: int = 0
    current_stage_name: str = ""
    episodes_in_stage: int = 0
    stage_transitions: list[tuple[int, str]] = None  # (step, stage_name)

    def __post_init__(self):
        if self.stage_transitions is None:
            self.stage_transitions = []


class CurriculumManager:
    """Manages curriculum progression during training.

    Updates environment parameters (particle count, heap distance, DR,
    reward weights) based on the current training step.
    """

    def __init__(self, cfg: CurriculumCfg):
        self.cfg = cfg
        self.state = CurriculumState()
        self._active_stage: CurriculumStageCfg | None = None

        if cfg.enabled and cfg.stages:
            self._active_stage = cfg.stages[0]
            self.state.current_stage_name = self._active_stage.name

    @property
    def active_stage(self) -> CurriculumStageCfg | None:
        """Currently active curriculum stage."""
        return self._active_stage

    @property
    def is_enabled(self) -> bool:
        return self.cfg.enabled

    def update(self, total_steps: int) -> bool:
        """Update curriculum state based on current training step.

        Args:
            total_steps: Total environment steps so far.

        Returns:
            True if the stage changed.
        """
        if not self.cfg.enabled:
            return False

        self.state.total_steps = total_steps
        new_stage = self.cfg.get_stage(total_steps)

        if new_stage.name != self.state.current_stage_name:
            old_name = self.state.current_stage_name
            self._active_stage = new_stage
            self.state.current_stage_name = new_stage.name
            self.state.current_stage_idx += 1
            self.state.episodes_in_stage = 0
            self.state.stage_transitions.append((total_steps, new_stage.name))
            return True

        return False

    def get_num_particles(self) -> int:
        """Get particle count for current stage."""
        if self._active_stage is not None:
            return self._active_stage.num_particles
        return 500  # default

    def get_max_heap_distance(self) -> float:
        """Get maximum heap distance for current stage."""
        if self._active_stage is not None:
            return self._active_stage.max_heap_distance
        return 0.5

    def is_dr_enabled(self) -> bool:
        """Whether domain randomization is active in current stage."""
        if self._active_stage is not None:
            return self._active_stage.dr_enabled
        return False

    def get_reward_weight_overrides(self) -> dict[str, float]:
        """Get reward weight overrides for current stage.

        Returns:
            Dict of weight name → value. Only includes overridden weights.
        """
        overrides = {}
        if self._active_stage is not None:
            if self._active_stage.approach_weight is not None:
                overrides["approach"] = self._active_stage.approach_weight
            if self._active_stage.transfer_weight is not None:
                overrides["soil_transfer"] = self._active_stage.transfer_weight
        return overrides

    def get_info(self) -> dict:
        """Get curriculum info for logging."""
        return {
            "curriculum/stage_idx": self.state.current_stage_idx,
            "curriculum/stage_name": self.state.current_stage_name,
            "curriculum/episodes_in_stage": self.state.episodes_in_stage,
            "curriculum/num_particles": self.get_num_particles(),
            "curriculum/max_heap_distance": self.get_max_heap_distance(),
            "curriculum/dr_enabled": int(self.is_dr_enabled()),
        }
