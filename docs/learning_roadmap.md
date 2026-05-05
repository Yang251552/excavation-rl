# Stage 0 代码学习路线（初学者向）

本文为完全没有 RL / 强化学习背景的读者提供一条**由浅入深**的学习路径，针对本项目 Stage 0 的代码进行划分。每一级都能独立有收获、不需要等到学完下一级才有用。

---

## Level 0：先理解项目"在做什么"（30 分钟）

**只看 README**，不打开任何代码：

1. `README.md` 顶部 "Motivation" + "Task Description" + "Stage 0 Results" 三节
2. 看 `results/figures/stage0/*.png` 三张训练曲线截图

**目标**：能用 3 句话向别人解释"这个项目想做什么、Stage 0 已经做到什么、剩下要做什么"。

**自我验证**：尝试不看 README 重新画一下"系统架构"——4 个核心模块（`envs/` / `training/` / `soil/` / `robot/`）各自负责什么。

---

## Level 1：Python 数据类与配置系统（1–2 小时）

**核心思想**：现代 ML 项目用 `@dataclass` 把所有可调参数集中管理，避免散落在各处。

读这两个文件（共 460 行，但 80% 是重复模板）：

| 文件 | 行数 | 看什么 |
|---|---|---|
| `training/ppo_cfg.py` | 151 | `PPOCfg` 和 `NetworkCfg` 两个 dataclass + 3 个 stage 工厂函数 |
| `envs/excavation_env_cfg.py` | 309 | `ExcavationEnvCfg` 是一个**嵌套**的 dataclass（里面有 `RewardCfg`、`SceneCfg`、`CurriculumCfg`...） |

**学到的东西**：
- `@dataclass` 装饰器的用法
- 嵌套 dataclass 怎么组织复杂配置
- `field(default_factory=lambda: [...])` 为什么要用 lambda（避免可变默认值的陷阱）
- `@property` 怎么在 dataclass 里加"派生字段"（看 `PPOCfg.batch_size`）

**动手验证**：改一个超参重新跑训练
```bash
# 改 ppo_cfg.py 第 102 行 max_iterations=1000 → 50
# 重新跑训练，看是否真的只跑 50 iter 就停
python -m training.train --stage 0 --wandb --max-iterations 50
```

---

## Level 2：reward 函数怎么设计（2–3 小时）

**核心思想**：把"我希望 agent 干什么"翻译成数学公式。

读三个文件（共约 600 行，但每个 reward 分量只有 2–3 行核心逻辑）：

| 文件 | 看什么 |
|---|---|
| `envs/rewards.py` | `compute_reward()` 函数，一行一个 reward 分量 |
| `envs/excavation_env_cfg.py` 的 `RewardWeights` | 8 个 reward 权重的默认值 |
| `tests/test_reward.py` | **关键！** 看测试怎么用具体数值调用 reward 函数验证逻辑 |

**学到的东西**：
- Dense reward 怎么写（每个分量用 `exp(-α·dist)` 引导 agent 靠近目标）
- 稀疏 reward 怎么写（success_bonus 只在 terminal 触发）
- 怎么用 `RewardComponents` 这种 dataclass 把分量分开记录（不只算总和）
- 测试驱动开发：先写测试用例（"agent 如果做 X，应该拿到 reward Y"）再写实现

**动手验证**：自己加一个新 reward 分量
```python
# 在 RewardWeights 加一个新字段：
energy_penalty: float = -0.001  # w9: penalize joint velocity

# 在 compute_reward() 加:
components.energy = w.energy_penalty * float(np.sum(state.joint_vel ** 2))
```
跑一遍 `pytest tests/test_reward.py -v` 看是否过。

---

## Level 3：Gymnasium 环境的 API（1–2 天）

**核心思想**：所有 RL 框架（OpenAI Gym / Gymnasium / Isaac Lab）都遵循同一套 API：`reset()` 和 `step()`。

读三个文件，按这个顺序：

1. `envs/terminations.py`（149 行）— 最简单，纯 `if / else` 判断 episode 结束
2. `envs/excavation_env.py` 里的 `step()` 函数（约 230–320 行）— **核心循环**：
   ```
   action 进来 → 物理仿真一步 → 算 reward → 判断 done → obs/reward/info 出去
   ```
3. `tests/test_env.py`（232 行）— 看怎么用 Gym API 调用环境

**学到的东西**：
- `reset()` 返回什么（`obs, info`）、`step()` 返回什么（`obs, reward, terminated, truncated, info`）
- `terminated` vs `truncated` 区别（任务结束 vs 超时被截断）
- `info` dict 怎么传 episode 级元数据
- 为什么需要 `VecExcavationEnv` 包装（一次 step 多个 env）

**动手验证**：写一个 5 行的随机策略测试
```python
from envs.excavation_env import ExcavationEnv
from envs.excavation_env_cfg import ExcavationEnvCfg

env = ExcavationEnv(ExcavationEnvCfg())
obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()  # 随机动作
    obs, reward, term, trunc, info = env.step(action)
    print(f"reward={reward:.3f}, done={term or trunc}")
    if term or trunc:
        obs, _ = env.reset()
```

---

## Level 4：PPO 算法实现（1–2 周）

**核心思想**：把 RL 算法的数学公式翻译成 PyTorch 代码。这一级是真正的"深度学习 + RL"，需要**先看完一个 PPO 教程**（推荐 OpenAI Spinning Up 或 "37 Implementation Details of PPO" 论文）。

读 `training/train.py`（673 行），分 3 块：

### 4a. ActorCritic 网络（line 46–138，~90 行）
- 学：MLP 怎么写、Gaussian policy 用 `torch.distributions.Normal`、`log_std` 是可学习参数
- 关键函数：`forward()`、`get_action()`、`evaluate_actions()`

### 4b. RolloutBuffer 与 GAE（line 145–214，~70 行）
- 学：怎么存训练数据、`compute_returns_and_advantages()` 里**反向**循环计算 GAE
- GAE 公式：`A_t = δ_t + γλ · A_{t+1}`，从 episode 末尾往前算

### 4c. PPO 主循环 与 update（line 220–475，~250 行）
- 学：rollout 怎么收集（line 295–330）、PPO loss 怎么算（line 393–450）
- 三个 loss：`policy_loss`（clipped surrogate）+ `value_loss`（MSE）+ `entropy_loss`（鼓励探索）
- KL 计算 + adaptive LR：line 478–485

**学到的东西**：

PPO 论文里"clipped surrogate objective"长什么样：
```python
ratio = torch.exp(log_probs - old_log_probs)
surr1 = ratio * advantages
surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages
policy_loss = -torch.min(surr1, surr2).mean()
```
这 4 行就是整个 PPO 的"灵魂"。

其他重要内容：
- 怎么用 PyTorch 做 `loss.backward()` + `optimizer.step()`
- Value function clipping、advantage normalization 这些"实现细节"为什么重要

**动手验证**：把 `clip_param` 从 0.2 改成 0.05 重训，看 reward 曲线是不是变得更稳但学得更慢（trust region 收紧了）。

---

## Level 5：粒子物理 + GPU 编程（1–3 个月，Stage 1 才用到）

这一级 Stage 0 用不到，**先跳过**，等做 Stage 1 时再回来。

涉及：
- `soil/particle_system.py` — Warp GPU kernels（`@wp.kernel`）
- `soil/soil_terrain.py` — 程序化地形生成
- `soil/soil_properties.py` — 5 种土壤参数 preset

需要先学：CUDA 基础概念、SPH / MPM 粒子方法、NVIDIA Warp 教程。

---

## 给初学者的"30 天学习计划"建议

| 周 | Level | 目标 | 检验方式 |
|---|---|---|---|
| 第 1 周 | Level 0–1 | 理解项目，会改超参 | 用不同 LR 跑 3 个 run，比较 wandb 曲线 |
| 第 2 周 | Level 2 | 看懂 reward 设计 | 自己加 1 个新 reward 分量并测试通过 |
| 第 3 周 | Level 3 | 看懂 Gym 环境 | 用随机 policy 在你的环境里跑 100 episode 收集数据 |
| 第 4 周 | Level 4a + 4b | 看懂 ActorCritic 和 buffer | 画出 GAE 反向计算的"数据流图" |
| 1–2 月后 | Level 4c | 看懂 PPO update | 能向别人解释 `surr1`, `surr2`, `torch.min` 这 3 行做什么 |

---

## 重要提醒：哪些**不要**碰

为了避免被复杂度劝退，初学阶段**这些文件可以视而不见**：

| 文件 | 为什么先跳 |
|---|---|
| `soil/particle_system.py` | Warp GPU kernel，门槛高，Stage 0 没用 |
| `envs/curriculum.py` | 概念简单但跟其他模块深度耦合 |
| `envs/events.py` | DR 实现，13 个参数随机化，Stage 0 关掉了 |
| `evaluation/*` | 评估框架，没真正用上 |
| `training/train.py` 里的 `_save_checkpoint` / `_load_checkpoint` | I/O 细节，不是 RL 核心 |

---

## 配套学习资源（按 level 对应）

- **Level 0–1**：Real Python 的 `@dataclass` 教程
- **Level 2**：Sutton & Barto《Reinforcement Learning: An Introduction》第 3 章
- **Level 3**：Gymnasium 官方文档 "Basic Usage" 章节
- **Level 4**：OpenAI Spinning Up 的 PPO 章节（**最重要**）+ Costa Huang 的 "37 Implementation Details of PPO" 论文
- **Level 5**：NVIDIA Warp Tutorial + David Mount《Lecture Notes on Particle Systems》

---

## 学习心法

每学完一级，**用代码动手改一改**（哪怕只是改个超参或改个 print），不要光读。RL 项目最大的学习陷阱就是"读懂了但写不出"。

Stage 0 的好处是：**单 process、CPU、无依赖**，改完能立刻 1 分钟看到结果，反馈快，特别适合做小实验。
