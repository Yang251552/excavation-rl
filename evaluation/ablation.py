"""Ablation experiment runner.

Automates the ablation experiments defined in Section 4.4:
  1. Reward function ablation
  2. Domain randomization ablation
  3. Curriculum learning ablation
  4. Observation space comparison (optional)

Each experiment trains multiple configurations with multiple seeds
and produces comparison tables and training curves.

Usage:
    python -m evaluation.ablation --experiment reward --seeds 3
    python -m evaluation.ablation --experiment dr --seeds 3
    python -m evaluation.ablation --experiment all --seeds 3
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from envs.excavation_env_cfg import (
    ExcavationEnvCfg,
    RewardWeights,
    DomainRandomizationCfg,
    CurriculumCfg,
    CurriculumStageCfg,
)


def get_reward_ablation_configs() -> dict[str, ExcavationEnvCfg]:
    """Generate configurations for reward ablation experiments.

    Returns:
        Dict of config_name → ExcavationEnvCfg.
    """
    configs = {}

    # Full (baseline)
    cfg_full = ExcavationEnvCfg()
    configs["full"] = cfg_full

    # No approach reward
    cfg_no_approach = ExcavationEnvCfg()
    cfg_no_approach.reward.weights.approach = 0.0
    configs["no_approach"] = cfg_no_approach

    # No bucket load reward
    cfg_no_load = ExcavationEnvCfg()
    cfg_no_load.reward.weights.bucket_load = 0.0
    configs["no_load"] = cfg_no_load

    # Sparse only (only soil transfer)
    cfg_sparse = ExcavationEnvCfg()
    cfg_sparse.reward.weights.approach = 0.0
    cfg_sparse.reward.weights.bucket_load = 0.0
    cfg_sparse.reward.weights.transport = 0.0
    cfg_sparse.reward.weights.action_smoothness = 0.0
    cfg_sparse.reward.weights.time_penalty = 0.0
    configs["sparse_only"] = cfg_sparse

    return configs


def get_dr_ablation_configs() -> dict[str, ExcavationEnvCfg]:
    """Generate configurations for domain randomization ablation."""
    configs = {}

    # No DR
    cfg_no_dr = ExcavationEnvCfg()
    cfg_no_dr.domain_randomization.enabled = False
    cfg_no_dr.curriculum.enabled = False
    configs["no_dr"] = cfg_no_dr

    # Soil DR only
    cfg_soil_dr = ExcavationEnvCfg()
    cfg_soil_dr.domain_randomization.enabled = True
    cfg_soil_dr.domain_randomization.joint_friction_scale_range = (1.0, 1.0)
    cfg_soil_dr.domain_randomization.payload_mass_range = (0.0, 0.0)
    cfg_soil_dr.domain_randomization.action_noise_std_range = (0.0, 0.0)
    cfg_soil_dr.domain_randomization.observation_noise_std = 0.0
    cfg_soil_dr.curriculum.enabled = False
    configs["soil_dr"] = cfg_soil_dr

    # Full DR
    cfg_full_dr = ExcavationEnvCfg()
    cfg_full_dr.domain_randomization.enabled = True
    cfg_full_dr.curriculum.enabled = False
    configs["full_dr"] = cfg_full_dr

    return configs


def get_curriculum_ablation_configs() -> dict[str, ExcavationEnvCfg]:
    """Generate configurations for curriculum learning ablation."""
    configs = {}

    # No curriculum (hardest difficulty from start)
    cfg_no = ExcavationEnvCfg()
    cfg_no.curriculum.enabled = False
    cfg_no.scene.soil_num_particles = 1000
    cfg_no.domain_randomization.enabled = True
    configs["no_curriculum"] = cfg_no

    # 2-stage curriculum
    cfg_2stage = ExcavationEnvCfg()
    cfg_2stage.curriculum.enabled = True
    cfg_2stage.curriculum.stages = [
        CurriculumStageCfg(
            name="easy", entry_step=0,
            num_particles=300, max_heap_distance=0.3,
            dr_enabled=False, approach_weight=2.0,
        ),
        CurriculumStageCfg(
            name="hard", entry_step=400_000,
            num_particles=1000, max_heap_distance=0.8,
            dr_enabled=True, approach_weight=0.1,
        ),
    ]
    configs["2_stage"] = cfg_2stage

    # 3-stage curriculum (default)
    cfg_3stage = ExcavationEnvCfg()  # uses default 3-stage curriculum
    configs["3_stage"] = cfg_3stage

    return configs


def get_observation_ablation_configs() -> dict[str, ExcavationEnvCfg]:
    """Generate configurations for observation space ablation (NICE-TO-HAVE)."""
    configs = {}

    # Height map 5x5
    cfg_hm5 = ExcavationEnvCfg()
    cfg_hm5.observation.soil_obs_mode = "height_map"
    cfg_hm5.observation.height_map_resolution = 5
    configs["height_map_5x5"] = cfg_hm5

    # Height map 10x10
    cfg_hm10 = ExcavationEnvCfg()
    cfg_hm10.observation.soil_obs_mode = "height_map"
    cfg_hm10.observation.height_map_resolution = 10
    configs["height_map_10x10"] = cfg_hm10

    # Centroid only
    cfg_centroid = ExcavationEnvCfg()
    cfg_centroid.observation.soil_obs_mode = "centroid"
    configs["centroid_only"] = cfg_centroid

    return configs


EXPERIMENT_REGISTRY = {
    "reward": get_reward_ablation_configs,
    "dr": get_dr_ablation_configs,
    "curriculum": get_curriculum_ablation_configs,
    "observation": get_observation_ablation_configs,
}


def run_ablation_experiment(
    experiment_name: str,
    num_seeds: int = 3,
    max_iterations: int = 1000,
    output_dir: str = "results/ablation_reports",
) -> None:
    """Run an ablation experiment.

    Args:
        experiment_name: Name of the experiment ("reward", "dr", "curriculum", "observation").
        num_seeds: Number of random seeds per configuration.
        max_iterations: Max training iterations per run.
        output_dir: Directory to save results.
    """
    if experiment_name not in EXPERIMENT_REGISTRY:
        raise ValueError(
            f"Unknown experiment: {experiment_name}. "
            f"Choose from {list(EXPERIMENT_REGISTRY.keys())}"
        )

    configs = EXPERIMENT_REGISTRY[experiment_name]()
    seeds = list(range(42, 42 + num_seeds))

    exp_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)

    results = {}

    print(f"Running ablation: {experiment_name}")
    print(f"  Configs: {list(configs.keys())}")
    print(f"  Seeds: {seeds}")
    print(f"  Max iterations: {max_iterations}")
    print()

    for config_name, cfg in configs.items():
        config_results = {"seeds": {}}
        print(f"  Config: {config_name}")

        for seed in seeds:
            print(f"    Seed {seed}...")
            cfg.seed = seed

            # In a full implementation, this would call the trainer
            # and collect results. For now, we save the config for
            # manual or scripted execution.
            config_path = os.path.join(exp_dir, f"{config_name}_seed{seed}_config.json")
            config_dict = {
                "config_name": config_name,
                "seed": seed,
                "max_iterations": max_iterations,
                "reward_weights": {
                    "soil_transfer": cfg.reward.weights.soil_transfer,
                    "approach": cfg.reward.weights.approach,
                    "bucket_load": cfg.reward.weights.bucket_load,
                    "transport": cfg.reward.weights.transport,
                    "action_smoothness": cfg.reward.weights.action_smoothness,
                    "time_penalty": cfg.reward.weights.time_penalty,
                    "collision": cfg.reward.weights.collision,
                },
                "dr_enabled": cfg.domain_randomization.enabled,
                "curriculum_enabled": cfg.curriculum.enabled,
                "observation_mode": cfg.observation.soil_obs_mode,
            }
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)

            config_results["seeds"][seed] = {"config_path": config_path}

        results[config_name] = config_results

    # Save experiment manifest
    manifest_path = os.path.join(exp_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "experiment": experiment_name,
                "configs": list(configs.keys()),
                "seeds": seeds,
                "max_iterations": max_iterations,
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\nAblation configs saved to: {exp_dir}")
    print(f"Manifest: {manifest_path}")
    print("\nTo run each configuration:")
    print(f"  python -m training.train --stage 2 --seed <SEED> --experiment-name <CONFIG_NAME>")


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument(
        "--experiment", type=str, required=True,
        choices=list(EXPERIMENT_REGISTRY.keys()) + ["all"],
        help="Which ablation experiment to run",
    )
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="results/ablation_reports")
    args = parser.parse_args()

    if args.experiment == "all":
        for exp_name in EXPERIMENT_REGISTRY:
            run_ablation_experiment(
                exp_name, args.seeds, args.max_iterations, args.output_dir
            )
    else:
        run_ablation_experiment(
            args.experiment, args.seeds, args.max_iterations, args.output_dir
        )


if __name__ == "__main__":
    main()
