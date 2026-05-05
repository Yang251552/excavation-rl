"""Hand-coded 5-phase scoop trajectory for behaviour-cloning warm-start.

Given the project's simplified Franka FK in envs/excavation_env.py:
    base_height = 0.333
    l2 = 0.316  (upper arm); l3 = 0.384  (forearm)
    h_reach = l2*cos(q1) + l3*cos(q1+q3)
    v_offset = l2*sin(q1) + l3*sin(q1+q3)
    ee_x = h_reach*cos(q0); ee_y = h_reach*sin(q0); ee_z = base_height + v_offset

We invert this 2D planar IK analytically (q1 = shoulder, q3 = elbow) and
use q0 for azimuth. Wrist (q4) and unused joints (q2, q5, q6) stay at
default values.

The 5 phases:
  1. descend       (steps   0- 40)  EE: start -> heap top
  2. plunge        (steps  40- 70)  EE: heap top -> mid-heap, slight forward push
  3. lift          (steps  70-110)  EE: rises out of heap
  4. swing+carry   (steps 110-160)  EE: swings azimuth toward target
  5. release       (steps 160-200)  EE: descends over target zone

Outputs joint trajectories suitable for both BC (action = delta to next pose)
and reverse-curriculum reset (state = pose at random t).
"""
from __future__ import annotations
import numpy as np


L2 = 0.316
L3 = 0.384
BASE_HEIGHT = 0.333


def _ik_2link(h: float, v: float, elbow_up: bool = False) -> tuple[float, float]:
    """Solve 2-link planar IK (shoulder, elbow) for target (h, v) from base.

    Returns (q_shoulder, q_elbow) in radians. If unreachable, clips to nearest.
    """
    d_sq = h * h + v * v
    d_max = L2 + L3
    if d_sq > d_max * d_max:
        scale = d_max * 0.999 / np.sqrt(d_sq)
        h, v = h * scale, v * scale
        d_sq = h * h + v * v
    cos_q3 = (d_sq - L2 * L2 - L3 * L3) / (2.0 * L2 * L3)
    cos_q3 = np.clip(cos_q3, -1.0, 1.0)
    q3 = np.arccos(cos_q3)
    if not elbow_up:
        q3 = -q3
    q1 = np.arctan2(v, h) - np.arctan2(L3 * np.sin(q3), L2 + L3 * np.cos(q3))
    return float(q1), float(q3)


def fk_ee(q: np.ndarray) -> np.ndarray:
    """Forward kinematics matching envs/excavation_env.py:_update_ee_from_joints."""
    h = L2 * np.cos(q[1]) + L3 * np.cos(q[1] + q[3])
    v = L2 * np.sin(q[1]) + L3 * np.sin(q[1] + q[3])
    ee = np.zeros(3, dtype=np.float32)
    ee[0] = h * np.cos(q[0])
    ee[1] = h * np.sin(q[0])
    ee[2] = BASE_HEIGHT + v
    ee[0] += 0.05 * np.sin(q[4]) * np.cos(q[0])
    ee[1] += 0.05 * np.sin(q[4]) * np.sin(q[0])
    return ee


def ee_target_to_joints(ee_target: np.ndarray, default_joints: np.ndarray) -> np.ndarray:
    """Solve full-arm IK by setting q0 from azimuth and (q1, q3) from planar IK.

    Other joints remain at default. Returns 7-DoF joint vector.
    """
    q = default_joints.copy()
    q[0] = np.arctan2(ee_target[1], ee_target[0])
    h = np.sqrt(ee_target[0] ** 2 + ee_target[1] ** 2)
    v = ee_target[2] - BASE_HEIGHT
    q[1], q[3] = _ik_2link(h, v, elbow_up=False)
    return q


def build_scoop_trajectory(
    default_joint_pos: np.ndarray,
    heap_center: tuple[float, float, float] = (0.5, 0.0, 0.05),
    target_center: tuple[float, float, float] = (0.4, 0.3, 0.05),
    n_steps: int = 200,
) -> np.ndarray:
    """Produce an `n_steps x 7` joint-position trajectory for a scoop.

    Returns absolute joint-position targets. Caller can convert to deltas
    (action = next - current) for BC.
    """
    heap = np.array(heap_center, dtype=np.float32)
    targ = np.array(target_center, dtype=np.float32)

    # EE start (matches default_joint_pos via FK)
    start_ee = fk_ee(default_joint_pos)

    # Waypoints in EE space. Key insight: particles settle to z=0.005 within
    # ~10 env steps, and bucket load_detection_bounds in z = [EE_z - 0.025,
    # EE_z + 0.04]. To detect ground particles need EE_z <= 0.030. Also
    # bucket must dwell at scoop depth long enough to scrape particles
    # forward (single transient pass scoops nothing).
    scoop_z = 0.025  # load detection covers world z=[0, 0.065] -> includes ground particles
    heap_back = np.array([heap[0] - 0.10, heap[1], heap[2] + 0.18], dtype=np.float32)  # behind heap, high
    plunge_start = np.array([heap[0] - 0.10, heap[1], scoop_z], dtype=np.float32)      # behind heap, at scoop depth
    plunge_end = np.array([heap[0] + 0.12, heap[1], scoop_z], dtype=np.float32)        # past heap front, scoop depth
    lift_end = np.array([heap[0] + 0.12, heap[1], heap[2] + 0.30], dtype=np.float32)   # high carry
    swing_end = np.array([targ[0], targ[1], targ[2] + 0.30], dtype=np.float32)
    release_end = np.array([targ[0], targ[1], targ[2] + 0.05], dtype=np.float32)

    phases = [
        (0, 10, start_ee, heap_back),         # move behind heap
        (10, 25, heap_back, plunge_start),    # rapid descent to scoop depth
        (25, 80, plunge_start, plunge_end),   # SLOW horizontal sweep through heap (55 steps for friction to act)
        (80, 110, plunge_end, lift_end),      # lift up with payload
        (110, 160, lift_end, swing_end),      # swing+carry to target
        (160, n_steps, swing_end, release_end),  # release over target
    ]

    joint_traj = np.zeros((n_steps, len(default_joint_pos)), dtype=np.float32)
    for t in range(n_steps):
        # find phase
        for t0, t1, p0, p1 in phases:
            if t0 <= t < t1:
                alpha = (t - t0) / max(1, (t1 - t0))
                # smoothstep for less jerky motion
                alpha = 3 * alpha * alpha - 2 * alpha * alpha * alpha
                ee = (1 - alpha) * p0 + alpha * p1
                break
        else:
            ee = release_end
        joint_traj[t] = ee_target_to_joints(ee, default_joint_pos)
    return joint_traj


def trajectory_to_action_deltas(
    joint_traj: np.ndarray,
    initial_joints: np.ndarray,
    action_scale: float,
    action_clip: float = 1.0,
) -> np.ndarray:
    """Convert absolute joint trajectory to per-step actions (in [-1, 1]).

    Action in env is `joint_delta = action * action_scale`, so
    action = (next_target - current) / action_scale, clipped to [-1, 1].
    """
    n = joint_traj.shape[0]
    actions = np.zeros_like(joint_traj)
    cur = initial_joints.copy()
    for t in range(n):
        delta = joint_traj[t] - cur
        action = delta / action_scale
        action = np.clip(action, -action_clip, action_clip)
        actions[t] = action
        # next step's "current" is cur + applied delta (clipped action * scale)
        cur = cur + action * action_scale
    return actions


if __name__ == "__main__":
    # Run the trajectory through the env and report transfer_ratio
    import sys
    sys.path.insert(0, "/home/ubuntu/excavation-rl")
    from envs.excavation_env_cfg import ExcavationEnvCfg
    from envs.excavation_env import ExcavationEnv

    cfg = ExcavationEnvCfg()
    cfg.scene.use_rigid_body_proxy = False
    env = ExcavationEnv(cfg)

    default_q = np.array(env.arm_cfg.default_joint_pos, dtype=np.float32)
    print(f"Default joint pos: {default_q}")
    print(f"Default EE pos (FK): {fk_ee(default_q)}")

    traj = build_scoop_trajectory(default_q, heap_center=cfg.scene.soil_heap_center,
                                   target_center=cfg.scene.target_center, n_steps=200)
    print(f"Trajectory shape: {traj.shape}")
    print(f"Final joint target: {traj[-1]}")
    print(f"Final EE (FK on traj[-1]): {fk_ee(traj[-1])}")

    # Visualise EE trajectory at key steps
    for t in [0, 40, 70, 110, 160, 199]:
        print(f"  step {t:3d}: joints={traj[t][:4].round(3)} EE_FK={fk_ee(traj[t]).round(3)}")

    # Convert to actions
    actions = trajectory_to_action_deltas(traj, default_q, env.cfg.action.action_scale)
    print(f"Action range: min={actions.min():.3f} max={actions.max():.3f} mean_abs={np.abs(actions).mean():.3f}")
    print(f"Frac actions clipped: {(np.abs(actions) >= 1.0 - 1e-3).mean()*100:.1f}%")

    # Replay in env with detailed diagnostics
    obs = env.reset()
    print(f"\nsim_dt: {env.cfg.sim_dt}")
    bucket_loads = []
    transfer_ratios = []
    ee_zs = []
    ee_xs = []
    particle_z_means = []
    particle_z_mins = []
    n_particles_in_bucket_aabb = []  # particles within full bucket half_extent

    bg = env.scene.bucket_cfg.geometry
    half_d, half_w, half_h = bg.depth/2, bg.width/2, bg.height/2
    released_at = -1
    for t in range(200):
        obs, r, done, trunc, info = env.step(actions[t])
        # Release carried particles when EE arrives over target zone (phase 5+)
        if t == 170 and released_at < 0:
            n = env.scene.release_carried_particles()
            print(f"  >>> Released {n} carried particles at t={t}")
            released_at = t
        bucket_loads.append(env.scene.get_bucket_load(env._ee_pos))
        transfer_ratios.append(env.scene.get_soil_in_target_ratio())
        ee_zs.append(env._ee_pos[2])
        ee_xs.append(env._ee_pos[0])
        ppos = env.scene.particle_system.get_positions_numpy()
        particle_z_means.append(ppos[:,2].mean())
        particle_z_mins.append(ppos[:,2].min())
        # count particles within bucket half_extent box
        in_box = ((np.abs(ppos[:,0] - env._ee_pos[0]) < half_d) &
                  (np.abs(ppos[:,1] - env._ee_pos[1]) < half_w) &
                  (np.abs(ppos[:,2] - env._ee_pos[2]) < half_h))
        n_particles_in_bucket_aabb.append(in_box.sum())
    bucket_loads = np.array(bucket_loads)
    transfer_ratios = np.array(transfer_ratios)
    ee_zs = np.array(ee_zs)
    ee_xs = np.array(ee_xs)
    n_particles_in_bucket_aabb = np.array(n_particles_in_bucket_aabb)

    print(f"\n=== Per-step diagnostics (every 10 steps) ===")
    for t in [0, 10, 20, 30, 40, 50, 60, 70, 100, 150]:
        print(f" t={t:3d}: EE=[{ee_xs[t]:.3f},?,{ee_zs[t]:.3f}] | "
              f"particles in bucket_AABB={n_particles_in_bucket_aabb[t]:>3} | "
              f"bucket_load={bucket_loads[t]:.4f} | particle_z mean={particle_z_means[t]:.3f} min={particle_z_mins[t]:.3f}")

    print(f"\n=== Demo replay results ===")
    print(f"max bucket_load: {bucket_loads.max():.4f} kg (= {bucket_loads.max()/0.0063:.0f} particles)")
    print(f"steps with bucket_load > 0: {(bucket_loads > 0).sum()}/200")
    print(f"max particles in bucket AABB (interaction zone): {n_particles_in_bucket_aabb.max()}")
    print(f"final transfer_ratio: {transfer_ratios[-1]:.4f}")
    print(f"max transfer_ratio:   {transfer_ratios.max():.4f}")
    print(f"EE z trajectory: min={ee_zs.min():.3f} max={ee_zs.max():.3f}")
    print(f"Random baseline transfer_ratio: ~0.15 (for comparison)")
    print(f"PPO trained transfer_ratio:     ~0.02 (for comparison)")
    if transfer_ratios.max() > 0.05:
        print("PASS: demo achieves >5% transfer; suitable for BC")
    else:
        print("FAIL: demo doesn't transfer enough; tune waypoints or IK")
