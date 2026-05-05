"""Generate behaviour-cloning demo data: (obs, action) pairs from the
hand-coded scoop trajectory across multiple seeded envs with small jitter.

Saves to results/bc_demos.npz with arrays:
  obs:     (N, obs_dim)
  actions: (N, action_dim)
"""
from __future__ import annotations
import sys
sys.path.insert(0, "/home/ubuntu/excavation-rl")

import numpy as np
from envs.excavation_env_cfg import ExcavationEnvCfg
from envs.excavation_env import ExcavationEnv
from scripts.scoop_demo import build_scoop_trajectory, trajectory_to_action_deltas

OUT_PATH = "/home/ubuntu/excavation-rl/results/bc_demos.npz"
SNAPSHOT_PATH = "/home/ubuntu/excavation-rl/results/scoop_snapshot.npz"


def _pad_carried(carried_lists, max_n):
    """Pad variable-length carried index lists to a fixed-width matrix with -1 sentinel."""
    out = -np.ones((len(carried_lists), max_n), dtype=np.int32)
    for t, idxs in enumerate(carried_lists):
        if len(idxs) > 0:
            out[t, :len(idxs)] = idxs
    return out
N_EPISODES = 64           # doubled from 32
N_STEPS = 200
RELEASE_STEP = 170
ACTION_NOISE_STD = 0.05   # raised from 0.02 for broader BC coverage


def main():
    cfg = ExcavationEnvCfg()
    cfg.scene.use_rigid_body_proxy = False
    env = ExcavationEnv(cfg)

    default_q = np.array(env.arm_cfg.default_joint_pos, dtype=np.float32)
    base_traj = build_scoop_trajectory(default_q,
                                        heap_center=cfg.scene.soil_heap_center,
                                        target_center=cfg.scene.target_center,
                                        n_steps=N_STEPS)
    base_actions = trajectory_to_action_deltas(base_traj, default_q,
                                                 env.cfg.action.action_scale)

    all_obs = []
    all_actions = []
    transfers = []
    # For reverse-curriculum particle-state replay: pick the highest-transfer
    # episode and snapshot (joint_pos, particle_positions, particle_velocities,
    # carried_offsets) at each step. Reset can then load any snapshot t0.
    snapshot_joint = []      # (N_STEPS, num_joints)
    snapshot_particles = []  # (N_STEPS, num_particles, 3)
    snapshot_part_vel = []   # (N_STEPS, num_particles, 3)
    snapshot_carried = []    # (N_STEPS, list_of_carried_indices)
    best_transfer = -1.0
    best_snapshot = None

    for ep in range(N_EPISODES):
        rng = np.random.default_rng(1000 + ep)
        env.cfg.seed = 1000 + ep
        env._rng = rng
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result

        # Add small Gaussian noise to actions for diversity
        noise = rng.normal(0, ACTION_NOISE_STD, size=base_actions.shape).astype(np.float32)
        actions = np.clip(base_actions + noise, -1.0, 1.0)

        ep_obs = []
        ep_act = []
        ep_joint = []
        ep_part = []
        ep_part_vel = []
        ep_carried = []
        for t in range(N_STEPS):
            ep_obs.append(obs.copy() if hasattr(obs, "copy") else np.array(obs))
            ep_act.append(actions[t].copy())
            ep_joint.append(env._joint_pos.copy())
            ppos = env.scene.particle_system.get_positions_numpy().copy()
            pvel = env.scene.particle_system.velocities.numpy().reshape(-1, 3).copy()
            ep_part.append(ppos)
            ep_part_vel.append(pvel)
            carried = list(getattr(env.scene, "_carried_particle_offsets", {}).keys())
            ep_carried.append(np.array(carried, dtype=np.int32))
            obs, _, _, _, _ = env.step(actions[t])
            if t == RELEASE_STEP:
                env.scene.release_carried_particles()
        tr = env.scene.get_soil_in_target_ratio()
        transfers.append(tr)
        all_obs.append(np.stack(ep_obs))
        all_actions.append(np.stack(ep_act))
        if tr > best_transfer:
            best_transfer = tr
            best_snapshot = {
                "joint": np.stack(ep_joint),
                "particles": np.stack(ep_part),
                "particle_vel": np.stack(ep_part_vel),
                # carried indices are variable-length; pad with -1 to a max width
                "carried": _pad_carried(ep_carried, ep_part[0].shape[0]),
            }
        if ep % 8 == 0:
            print(f"  ep {ep}: transfer_ratio = {tr:.3f}")

    obs_arr = np.concatenate(all_obs, axis=0).astype(np.float32)
    act_arr = np.concatenate(all_actions, axis=0).astype(np.float32)
    print(f"\n=== Summary ===")
    print(f"Total samples: obs={obs_arr.shape}, actions={act_arr.shape}")
    print(f"Transfer ratio: mean={np.mean(transfers):.3f}, std={np.std(transfers):.3f}, "
          f"min={np.min(transfers):.3f}, max={np.max(transfers):.3f}")
    print(f"Saving BC pairs to {OUT_PATH}")
    import os
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez_compressed(OUT_PATH, obs=obs_arr, actions=act_arr,
                        transfer_ratios=np.array(transfers))
    if best_snapshot is not None:
        print(f"Saving best-episode snapshot (transfer={best_transfer:.3f}) to {SNAPSHOT_PATH}")
        np.savez_compressed(SNAPSHOT_PATH, **best_snapshot,
                             transfer_ratio=np.array([best_transfer]))


if __name__ == "__main__":
    main()
