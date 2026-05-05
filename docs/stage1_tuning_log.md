# Stage 1 调参过程记录 — wandb API 驱动的诊断 (v17 → v22)

记录用 wandb API 拉数据 + 针对性改一个参数 + 重测的迭代过程。

写下这份文档的动机：早期我们用 `ssh + tail + grep` 读训练日志，结果**漏看了 40% 的关键数据**——wandb summary 在 stdout 里被截断到 9 个 key 加 `+15 ...`，被 grep 过滤掉的就是所有 `reward/*` 分量。错过了"成功事件存在但稀疏"这个关键信号，让我们做出"reward shaping 已经穷尽"的错误判断。换上 wandb API 之后，第一次拉到完整数据就把诊断完全推翻了。

---

## 1. 为什么不能继续用 ssh + tail + grep

### 看到的（错的）

```bash
ssh ubuntu@... "tail -30 logs/stage1_v17_seed7.log | grep -E 'Performance|approx_kl|clip_fraction|explained_variance|🚀'"
```

输出：
```
Performance/episodic_length 500
Performance/episodic_return 219.6
Performance/soil_transfer_ratio 0.01387
Performance/success_rate 0
Policy/approx_kl 4.56
Policy/clip_fraction 0.97
Policy/explained_variance 0.71
+15 ...      ← 这一行就是 reward 分量被截断的信号
```

我们当时盯着 `Performance` + `Policy` 这两组，没追问 `+15 ...` 是什么。

### 漏掉的（对的）

那 15 个被隐藏的 key 里包含完整的 `reward/*` 分量分解：

```
reward/_total, reward/approach, reward/dig, reward/load,
reward/transport, reward/transfer, reward/success_bonus,
reward/smooth_penalty, reward/time_penalty,
reward/collision_penalty, reward/joint_limit_penalty
```

后来用户截一张 wandb dashboard 截图给我，我立刻看出 `reward/success_bonus` 在 50%+ 的 log 窗口都非零（不是 success_rate=0 的"0%"），`reward/transfer` 在 seed 123 偶发 0.92。这些才是真正的诊断信号。

### 教训

**wandb summary 的 stdout 输出不可信。永远用 wandb API 拉完整数据。** 截断点是 9 keys，剩下被 `+N ...` 隐藏。

---

## 2. wandb API 工具脚本

写了两份分析脚本固化下来：

### `scripts/analyze_v17.py` — 单版本完整 dump

```python
import wandb, numpy as np
api = wandb.Api()
run = api.run("yangchenghan2515-eth-z-rich/excavation-rl/<run_id>")
rows = list(run.scan_history(keys=ALL_KEYS))   # 完整历史，无 sample
series = {k: np.array([r.get(k) for r in rows if r.get(k) is not None], dtype=float)
          for k in ALL_KEYS}
```

关键点：
- 用 `scan_history` 不用 `history(samples=N)`，否则只拿到等距采样
- 服务器无 pandas 时用纯 dict（每行 `r.get(k)`）
- `~/.netrc` 有 wandb token 自动认证，不用 prompt

### `scripts/analyze_v18.py` — 多版本对比

加了 `delta = v_new - v_old` 列 + `pct = delta / |v_old| * 100` 计算，能一眼看出每个改动的效果幅度。

---

## 3. 调参迭代时间线

每行 = 一次"改一个参数 → 跑 50 iter smoke 或 500 iter 多 seed → 用 wandb API 拉完整数据 → 决定下一步"。

### V17 baseline (3 seeds × 500 iter, pure PPO, fixed 8 bugs)

最初手动 grep 读到的："plateau at 1.4% transfer"。

wandb API 拉完后**真实数据**：

| 指标 | seed 7 | seed 42 | seed 123 |
|---|---|---|---|
| Performance/soil_transfer_ratio (last5 mean) | 0.86% | 0.47% | **2.88%** |
| Performance/soil_transfer_ratio (max single window) | 1.5% | 3.0% | **9.2%** |
| reward/success_bonus 触发频率 | 65% of windows | 22% | 61% |
| Policy/approx_kl (final) | 4.56 | 8.78 | 6.68 |
| Policy/clip_fraction (final) | 0.97 | 0.96 | 0.97 |

**信号**：
- success_bonus 频繁触发（22-65% log 窗口）说明**transfer_ratio 偶尔超过 0.3**，但 final state 的 transfer_ratio 是 1-3%——粒子被推进 target zone 又被推出来
- KL = 7（vs stage 0 的 <0.1）+ clip = 96% → PPO 严重 over-clipped → 97% 的梯度更新被丢弃

**改 v18 的依据**（不再是"reward 平台期"，而是**两个具体 PPO 健康问题**）：
1. KL 漂移：obs_rms 持续 update 让 normalised obs 慢慢变化
2. smoothness penalty 平均 -10/iter 跟正向 reward 同量级，压制 scoop 需要的快速动作

---

### V18 (obs_rms freeze + 2× KL stop + smoothness 0.01 → 0.001)

| 指标 | v17 | v18 | delta | 解读 |
|---|---|---|---|---|
| reward/transfer | 0.10 | 0.17 | **+64%** | smoothness 降低让动作不被压制 |
| reward/success_bonus | 0.16 | 0.23 | +45% | 更多 transient 成功 |
| transfer_ratio mean | 1.40% | **2.40%** | +71% | 实质改善 |
| Policy/approx_kl seed 7 | 6.54 | 6.21 | -5% | 几乎没动 |
| Policy/approx_kl **seed 123** | 7.52 | **3104.75** | **+41000%** | **灾难** |
| Policy/approx_kl seed 123 max | 19.17 | **22820** | — | 同上 |

**意外发现**：seed 123 KL 直接爆到 22820。但 seed 7、42 的 KL 没动。为什么？

继续拉数据：seed 123 的 KL 在 iter 0-10 已经是 2322（freeze 是 iter 50 才触发）。所以**freeze 不是病因**——seed 123 的初始随机化恰好让 policy 在 iter 0 就遇到极端 obs，gradient 爆炸。

**改 v19 的依据**：保留 smoothness + KL stop 两个有效改动，撤回无效的 freeze。

---

### V19 (v18 - freeze)

| 指标 | v17 | v18 | v19 | 解读 |
|---|---|---|---|---|
| Policy/approx_kl seed 123 max | 19 | **22820** | **11** | freeze 移除后 seed 123 完全正常 ✓ 假设确认 |
| transfer_ratio mean | 1.40% | 2.40% | **2.17%** | 小幅回退但在 v18 的窗口内 |
| reward/_total | 209 | 198 | **230** | 实际更高（critic 没被 freeze 拖累） |

**信号**：
- v18 的 freeze 在健康 seed 上**零收益**（KL 几乎没变），但在病态 seed 上引爆 → 是**单纯有害**的改动
- transfer 没继续涨（2.17% ≈ 2.40%）。说明 reward shaping 这个维度的杠杆已经接近极限

**结论**：纯 reward + PPO hyperparameter tuning 的池子已经用完。需要换路线。

---

### Demo replay（不训 RL，验证任务可解性）

写 `scripts/scoop_demo.py`：用 5 阶段 EE 轨迹（descend → plunge → lift → swing → release）+ 平面 2-link IK 求解关节角，直接 replay 在 env 里。

第一次跑：transfer = 0%。诊断：

| 时刻 | EE pos | 粒子在 bucket AABB | bucket_load | particle z mean |
|---|---|---|---|---|
| t=10 | [0.49, 0, 0.23] | 0 | 0 | 0.005（已沉降） |
| t=50 | [0.56, 0, 0.025] | **291** | 0.0322 | 0.006 |
| t=70 | [0.60, 0, 0.025] | 95 | 0.040 | 0.005 |
| t=100 | [0.62, 0, 0.27] | 0 | 0 | 0.005 |

发现两个新 bug：
1. **粒子在 ~10 步内就沉降到 z=0.005**，bucket detection 的 z 下限要 0.030 才能扫到。`scoop_z` 从 0.07 降到 0.025
2. **AABB bucket 无壁**，t=70 时 95 颗粒子在 bucket interaction zone 内，但 t=100 bucket 一升 → 粒子全掉光（kernel 的 push_force 是 down，gravity 也 down，没有支撑）

→ 加 **containment 启发式**（particle pinned to bucket frame, auto-release on entering target zone）

最终 demo replay：**transfer = 22.4%**，超过 random baseline 15%。**任务可解，物理 OK，问题在 RL**。

---

### V22 (v19 + containment + BC pretraining)

`scripts/generate_bc_demos.py` 跑 32 episodes 收集 6400 (obs, action) pairs（demo episode mean transfer 21.5%），然后 BC pretrain 2000 步 MSE，再标准 PPO 500 iter。

wandb API 拉数据（vs v19）：

| 指标 | v19 | **v22** | delta | 解读 |
|---|---|---|---|---|
| Policy/approx_kl mean | 6.90 | **0.44** | **-93.6%** | BC 把 policy 起点拉到 reasonable 区域 |
| Policy/clip_fraction | 0.96 | **0.33** | -65.6% | PPO update 真正生效 |
| Performance/episodic_return | 230 | **837** | **+264%** | 巨幅提升 |
| reward/load | 0.19 | **390** | +200000% | containment 让粒子真留在桶里 |
| reward/transport | 0.03 | **43** | +150000% | agent 真的在运 |
| transfer mean | 2.17% | **2.42%** | +12% | 小幅 |
| **transfer peak (seed 42)** | 2.4% | **80%** | — | **demo 级单点成功** |

**KL trajectory 跨 seed**（每个 seed 第一行 = first10 iter mean，第二行 = last10 iter mean）：

| seed | v19 first10 → last10 | v22 first10 → last10 |
|---|---|---|
| 7 | 2.13 → 6.37 | 1.69 → **0.90** |
| 42 | 3.28 → 6.87 | 0.63 → **−0.11** |
| 123 | 4.12 → 7.45 | 0.59 → **0.54** |

v22 KL 全程低于 1，**史上第一次**。

### V22 的剩余问题（用 deterministic eval 测出来的）

写 `_eval_deterministic` 在 BC pretrain 后立刻评估**纯 BC policy**：

```
[BC-only (no PPO)] transfer_ratio across 4 eps: mean=0.0000, max=0.0000, max_load=0.0000
```

加 verbose trace：

```
t=  0: EE=[0.605, 0.000, 0.238] act_mean_abs=0.032   # 正确起点
t= 30: EE=[-0.060,-0.107, 0.005] act_mean_abs=0.792  # 已经跑到基座后方
t= 50: EE=[0.181, 0.045, 0.005] act_mean_abs=1.000   # action 完全饱和
t=100: EE=[0.203, 0.051, 0.005] act_mean_abs=0.999
```

**Distribution shift 经典症状**：
- BC MSE loss 收敛到 0.0019（看着很好）
- 但 closed-loop 时小动作误差（mean_abs ≈ 0.03）累积，30 步后 EE 已经跑到 demo 没见过的状态
- 在 OOD 状态下 policy 输出饱和（action ≈ ±1），形成正反馈把 EE 推得更远

→ 解释了为什么 v22 mean transfer 只有 2.4% 但 peak 80%：seed 42 偶尔走得离 demo 不太远的轨迹能完成 scoop，大部分 episode 偏出 demo 流形后失败

---

## 4. 数据驱动的下一步（已写进 plan 的 Stage 1.5）

每条都直接对应一个 wandb 信号：

| 信号 | 解读 | 方案 |
|---|---|---|
| BC-only deterministic = 0% | distribution shift 闭环失效 | 扩大 demo 集 + 加噪音覆盖更广 state |
| v22 KL 始终 < 1 但 mean transfer = 2.4% | PPO 在 BC 起点附近保持稳定，但 drift 慢慢走开 | PPO loss 加 KL anchor `+ β·KL(π_current ∥ π_BC)` |
| reverse curriculum 50 iter smoke transfer = 0.32% | 只 reset joint 不 reset 粒子 → 在 t=170 reset 时土堆还是初始状态，agent 没东西可放 | demo 生成时记录粒子 positions，reset 时一并恢复 |

---

## 5. 工具教训汇总

1. **stdout 截断不可信**：wandb summary 在 console 里被截到 9 keys，永远走 API
2. **scan_history 优于 history**：后者默认 sample，前者拿全部
3. **每个改动只做一次**：v17→v18 我同时改了 freeze + KL stop + smoothness 三件事，结果 freeze 引爆 seed 123 后没法立刻定位是哪个变量。v19 单独撤 freeze 才证伪
4. **deterministic eval 必须有**：BC + PPO 联合训练时，wandb 的 transfer_ratio 是 stochastic 的，看不出 BC 自己有多差。`_eval_deterministic` 暴露 BC distribution shift 是关键
5. **standalone demo 是 oracle**：demo replay = 22% 在所有调参 iteration 里都没变，是个稳定的"任务能不能解"的 ground truth。每次 RL 卡住先问"demo 还能跑吗"

---

## 6. 数据出处

所有 wandb run IDs（团队 `yangchenghan2515-eth-z-rich`，project `excavation-rl`）：

| Tag | seed 7 | seed 42 | seed 123 |
|---|---|---|---|
| `multi-seed-stage1` (v17, pure PPO + 8 bugs fixed) | wnmqx350 | x1nvjpgq | mtbrd8u4 |
| `multi-seed-v19` (- freeze, - smoothness, + KL stop) | 7bmiruid | q1414nr5 | t6ybiy9r |
| `multi-seed-v22 bc` (+ containment + BC pretrain) | auek7u4r | cc68bjg9 | bc64lv3c |

复现：`python scripts/analyze_v18.py` 在 EC2 上跑（需要 `~/.netrc` 有 wandb token），自动 dump v17/v18/v19/v22 全 dict 比较。

---

## 7. New-Entry Template (per CLAUDE.md §4.5)

Future iterations append below using this template — copy + fill in:

```markdown
### V<n> (one-line title — what changed)

**Hypothesis**: which earlier wandb signal triggered this change. Cite the metric name + value.

**Change** (one bullet per file edited):
- `path/to/file.py:LINE` — `old_value → new_value`. Why this magnitude / sign / name.

**Test**: smoke (50 iter, single seed) or long (500 iter × 3 seeds). Wandb run IDs.

**Result table**:

| Metric | v<n-1> | v<n> | delta | comment |
| --- | --- | --- | --- | --- |
| `Performance/soil_transfer_ratio` (final 5 windows) | … | … | … | … |
| `Policy/approx_kl` (final 10 iter mean) | … | … | … | … |
| `reward/<affected_components>` (final mean) | … | … | … | … |

**Decision**: keep / revert / partial-keep. If reverted, name the version restored to.

**Next**: which signal in this data motivates the next iteration.
```

Entries are immutable. If a later run invalidates an earlier conclusion, add a new entry that links back; do not rewrite the original.
