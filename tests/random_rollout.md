# Random-Policy Baseline (Stage 1)

Establishes the uniform-random transfer-ratio baseline used in [Stage 1 Results](../README.md#stage-1-results) to judge the trained policy's progress against trivial actions.

## Method

A single 500-step episode with the standard Stage 1 environment (Warp particle physics, 500 particles, default heap geometry), where the policy is replaced by `np.random.uniform(-0.3, 0.3)` per joint per step (i.e., a small-magnitude random walk in joint-delta space, intentionally *narrower* than PPO's default `init_noise_std = 0.5` so that the bucket dwells near the heap rather than thrashing past it).

```python
import numpy as np
from envs.excavation_env_cfg import ExcavationEnvCfg
from envs.excavation_env import ExcavationEnv

np.random.seed(42)
cfg = ExcavationEnvCfg()
cfg.scene.use_rigid_body_proxy = False
env = ExcavationEnv(cfg)
env.reset()
loads = []
for _ in range(500):
    act = np.random.uniform(-0.3, 0.3, env.action_space.shape[0]).astype(np.float32)
    env.step(act)
    loads.append(env.scene.get_bucket_load(env._ee_pos))
```

## Result (seed 42)

| Quantity | Value |
|---|---|
| Steps with EE inside heap AABB | 294 / 500 (59 %) |
| Steps with `bucket_load > 0` | 5 / 500 (1 %) |
| Max `bucket_load` over episode | 0.23 kg (≈ 37 particles) |
| Final `soil_in_target_ratio` | **0.15 (15 %)** |
| Final particle COM displacement | from (0.50, 0.00, 0.10) → (0.52, 0.13, 0.005) — pushed sideways and flattened |

The headline number is **`transfer_ratio = 15 %` from a single random rollout**, used as the floor that the trained Stage 1 policy must exceed to claim it has learned anything beyond random chance.

## Notes

- The 15 % is from a single seed; it is variance-heavy and not a true expected baseline. A proper baseline would average ≥ 30 random rollouts. We use 15 % as a sanity floor: the trained 3-seed mean is 1.82 % which is *below* even this single-seed sample, so the qualitative conclusion ("trained policy is below random") does not depend on the exact baseline value.
- The narrow `[-0.3, 0.3]` action range matters: PPO's default action distribution is wider, which makes the bucket thrash through the heap rather than dwell in it. A wide-action random rollout would likely score much lower than 15 %.
