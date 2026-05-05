# RL for Particle-Based Excavation — 实现规格书

> **项目**: Reinforcement Learning for Particle-Based Excavation in Isaac Lab
> **实验室**: ETH Zurich — Robotic Systems Lab (RSL), Prof. Marco Hutter
> **Supervisor**: Lorenzo Terenzi
> **作者背景**: ETH Robotics MSc，RL + sim-to-real 方向

---

## 0. 生成本文档的 Prompt

```
我是 ETH Robotics 硕士生，正在做 RSL 的 semester project：
"Reinforcement Learning for Particle-Based Excavation in Isaac Lab"。

请为这个项目生成一份实现规格书（不是项目管理文档），要求：

1. **系统架构**
   - 数据流架构图：标出 obs / action / reward 在各模块间的流向
   - 场景布局图：标出各物理实体的空间关系
   - 模块依赖关系图

2. **各子系统的代码级设计**
   - 观测空间：逐字段列出名称、维度、物理含义、取值范围
   - 动作空间：列出可选方案、推荐方案及理由
   - 奖励函数：写出完整公式，包含每一项的权重初始值
   - 终止条件：列出每种终止的触发条件和阈值
   - Domain Randomization：列出每个随机化参数的范围和单位
   - Curriculum Learning：定义每个阶段的进入条件和参数变化

3. **分阶段实现计划**
   - 按技术复杂度递进（先跑通简化版 → 再加真实物理）
   - 每阶段的验收标准用 checklist 格式，含具体数值
   - 标注 MUST-HAVE 和 NICE-TO-HAVE

4. **验证体系**
   - 环境正确性验证：在训练前必须通过的检查项
   - 训练过程监控：TensorBoard 中每个指标的含义和健康范围
   - 异常诊断表：常见训练失败现象 → 可能原因 → 排查方法
   - 消融实验：每组实验的配置差异、预期结果、输出格式
   - 定量指标：挖掘体积精度、reward 收敛、训练效率等，含目标值
   - 定性验证：与 Terra baseline 对比，sim-to-real gap 评估

5. **风险与备选方案**
6. **与 RSL 已有论文的对齐分析**
7. **技能可迁移性分析**（各阶段技能对其他项目/岗位的适用性）
8. **关键参考资料索引**（论文 + 技术文档）

用 Markdown 格式，包含 ASCII 图、表格、Python 伪代码、代码目录树。
所有设计决策给出推荐选项和理由。
```

---

## 1. 系统架构

### 1.1 数据流架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Loop                            │
│  ┌──────────┐    obs    ┌──────────┐   action   ┌───────────┐  │
│  │  RL Agent │ ◄──────── │  Env     │ ◄───────── │ RL Agent  │  │
│  │  (PPO)   │ ────────► │ Wrapper  │ ─────────► │  (PPO)    │  │
│  │          │  reward    │          │            │           │  │
│  └──────────┘           └────┬─────┘            └───────────┘  │
│       │                      │                                  │
│       │ rsl_rl               │ Isaac Lab ManagerBasedRLEnv      │
│       │                      │                                  │
│  ┌────▼──────────────────────▼──────────────────────────────┐  │
│  │              Isaac Lab Simulation Core                     │  │
│  │  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐  │  │
│  │  │ Robot Scene  │  │ Soil Particle│  │ Terrain/Ground  │  │  │
│  │  │ (Arm + EE)  │  │ System (Warp)│  │ Plane           │  │  │
│  │  └──────┬──────┘  └──────┬───────┘  └────────┬────────┘  │  │
│  │         │                │                    │            │  │
│  │         ▼                ▼                    ▼            │  │
│  │  ┌──────────────────────────────────────────────────┐     │  │
│  │  │        Isaac Sim / Newton Physics Engine          │     │  │
│  │  │   (GPU-accelerated rigid body + particle sim)     │     │  │
│  │  └──────────────────────────────────────────────────┘     │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 场景布局图

```
场景俯视图:

     ┌────────────────────────────────────┐
     │                                    │
     │    [Soil Heap]     [Target Zone]   │
     │    ░░░░░░░░░░░░    ┌──────────┐   │
     │    ░░░░░░░░░░░░    │          │   │
     │    ░░░░░░░░░░░░    │  目标区  │   │
     │    ░░░░░░░░░░░░    │          │   │
     │         ░░░░░░░    └──────────┘   │
     │                                    │
     │        [Robot Arm]                 │
     │           ╔═╗                      │
     │           ║ ║                      │
     │         ╔═╝ ╚═╗                    │
     │         ║     ║                    │
     │     ════╝     ╚════  ← base       │
     │                                    │
     │   ═══════════════════════════════  │
     │          Ground Plane              │
     └────────────────────────────────────┘

组成：
- 机械臂：Franka Panda (7DOF) 或 UR10 (6DOF)，末端替换为简化铲斗 (rigid body)
- 土壤：Warp 粒子系统，N 个粒子（初始 N=500~2000，按 GPU 能力调）
- 目标区域：固定区域，用于衡量土壤转移量
- 地面：刚体平面，提供碰撞
```

### 1.3 模块划分与技术选型

| 模块 | 职责 | 技术栈 | 依赖 |
|------|------|--------|------|
| **Simulation Environment** | 场景、物理仿真、观测/奖励/终止逻辑 | Isaac Lab (ManagerBasedRLEnv), NVIDIA Warp | Isaac Sim 4.x, USD assets |
| **Physics — Rigid Body** | 挖掘机本体、铲斗、地面刚体碰撞 | Isaac Sim PhysX | Isaac Sim |
| **Physics — Particle** | 土壤粒子仿真、铲斗-粒子交互 | NVIDIA Warp (GPU kernel) | Warp 1.x, CUDA |
| **Training** | PPO 算法、policy/value 网络、并行采样 | rsl_rl (RSL 官方库) | PyTorch, Isaac Lab wrapper |
| **Evaluation** | 指标计算、checkpoint 对比、baseline 复现 | 自定义脚本 + TensorBoard | 训练产出的 checkpoint |
| **Visualization** | 渲染、视频录制、训练曲线 | Isaac Sim renderer, matplotlib, TensorBoard | 训练 log 文件 |
| **Configuration** | 环境参数、训练超参、DR 参数 | Python dataclass / YAML | — |

### 1.4 模块依赖关系

```
Configuration ──────────────────────────────┐
     │                                      │
     ▼                                      ▼
Simulation Environment ◄──── Physics Backend (Warp + PhysX)
     │
     ├──────────► Training (rsl_rl PPO)
     │                 │
     │                 ▼
     │            Checkpoints
     │                 │
     ├──────────► Evaluation ◄─────┘
     │
     └──────────► Visualization
```

### 1.5 代码目录结构

```
excavation-rl/
│
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
│
├── envs/                              # 环境定义
│   ├── __init__.py
│   ├── excavation_env_cfg.py          # ManagerBasedRLEnvCfg 配置
│   ├── excavation_env.py              # 环境主体（Direct RL Env 方式）
│   ├── scene.py                       # 场景构建：机械臂 + 土壤 + 地面 + 目标区
│   ├── observations.py                # ObservationManager 定义
│   ├── rewards.py                     # RewardManager 定义
│   ├── terminations.py                # TerminationManager 定义
│   ├── curriculum.py                  # CurriculumManager
│   └── events.py                      # EventManager (domain randomization)
│
├── soil/                              # 土壤粒子仿真
│   ├── __init__.py
│   ├── particle_system.py             # Warp 粒子系统封装
│   ├── soil_properties.py             # 土壤参数（密度、摩擦、粘聚力）
│   └── soil_terrain.py                # 土壤地形生成（初始堆形）
│
├── robot/                             # 机器人相关
│   ├── __init__.py
│   ├── arm_cfg.py                     # 机械臂配置（Franka/UR10/自定义挖掘臂）
│   └── end_effector.py                # 铲斗末端执行器定义
│
├── training/                          # 训练相关
│   ├── __init__.py
│   ├── train.py                       # 训练入口
│   ├── ppo_cfg.py                     # rsl_rl PPO 超参数配置
│   └── callbacks.py                   # 训练回调（自定义 logging）
│
├── evaluation/                        # 评估与验证
│   ├── __init__.py
│   ├── evaluate.py                    # 评估脚本
│   ├── metrics.py                     # 定量指标计算
│   ├── ablation.py                    # 消融实验脚本
│   ├── visualize.py                   # 可视化工具
│   └── record_video.py               # 录制演示视频
│
├── configs/                           # 实验配置
│   ├── base.yaml                      # 基础配置
│   ├── reward_ablation.yaml           # reward 消融
│   └── dr_ablation.yaml              # domain randomization 消融
│
├── assets/
│   ├── excavator/                     # USD 格式挖掘机模型
│   └── terrain/                       # 地形 mesh 资产
│
├── results/
│   ├── training_curves/               # TensorBoard log
│   ├── videos/                        # 评估视频
│   ├── checkpoints/                   # 模型权重
│   ├── figures/                       # 论文/报告用图表
│   └── ablation_reports/              # 消融实验报告
│
├── docs/
│   ├── architecture.md
│   └── experiment_log.md              # 实验记录
│
└── tests/
    ├── test_env.py
    ├── test_soil.py
    └── test_reward.py
```

---

## 2. 各子系统代码级设计

### 2.1 观测空间 (Observation Space)

```python
observation_space 总维度: ~50-70D (取决于粒子聚合方式)

# 机器人本体感知 (proprioception)
joint_positions       : (7,)    # 关节角度 (rad)
joint_velocities      : (7,)    # 关节角速度 (rad/s)
ee_position           : (3,)    # 末端执行器位置 (m)
ee_orientation        : (4,)    # 末端执行器姿态 (quaternion)
ee_linear_velocity    : (3,)    # 末端线速度
ee_angular_velocity   : (3,)    # 末端角速度

# 铲斗状态
bucket_load           : (1,)    # 铲斗内土壤质量估计 (kg)
bucket_contact_force  : (3,)    # 铲斗受力 (N)

# 土壤状态（聚合表示，避免维度爆炸）
soil_height_map       : (H, W)  # 土壤高度图 (例如 5x5 grid) → 展平为 25D
# 或者:
soil_center_of_mass   : (3,)    # 土壤质心位置
soil_spread           : (3,)    # 土壤分布方差
soil_in_target        : (1,)    # 目标区内的土壤比例

# 目标
target_position       : (3,)    # 目标区域中心

# 上一步 action (有助于平滑控制)
previous_action       : (7,)    # 上一步关节控制指令
```

**粒子状态表示方式选择：**

| 方案 | 维度 | 优点 | 缺点 | 推荐阶段 |
|------|------|------|------|---------|
| A. 高度图 (height map) | H×W (如5×5=25) | 紧凑、空间结构保留 | 需离散化、精度有限 | **Stage 1-2 推荐** |
| B. 质心+方差 | 6 | 极度紧凑 | 丢失分布细节 | 快速原型 |
| C. 全粒子位置 | 3×N (3000+) | 信息完整 | 维度爆炸，训练困难 | 不推荐 |
| D. PointNet 编码 | 固定 (如64) | 灵活、可学习 | 增加模型复杂度 | Stage 3 可尝试 |

**推荐**：阶段一用方案 A（高度图），阶段二尝试方案 D（PointNet）。符合 Progressive-Resolution 论文思路。

### 2.2 动作空间 (Action Space)

```python
# 方案1: 关节空间控制 ← 推荐起步方案
action_space = Box(low=-1, high=1, shape=(7,))
# 映射: action * max_delta → 关节位置增量
# 底层用 PD 控制器跟踪目标关节角

# 方案2: 末端执行器空间控制（可选进阶）
action_space = Box(low=-1, high=1, shape=(6,))
# (dx, dy, dz, droll, dpitch, dyaw) → 通过 IK 转换为关节指令

# 方案3: 混合控制（铲斗 + 移动）
action_space = Box(low=-1, high=1, shape=(8,))
# 7 关节 + 1 铲斗开合角度
```

**推荐**：从方案 1（关节空间）开始。关节空间比末端空间更稳定，避免 IK 奇异性问题。确认可训练后再尝试方案 2。

### 2.3 奖励函数 (Reward Function)

```python
# ===== 核心奖励 =====

# R1: 土壤转移奖励 (最关键)
r_transfer = w1 * delta(soil_mass_in_target)           # w1 = 10.0

# R2: 接近奖励 (引导铲斗靠近土壤)
r_approach = w2 * exp(-alpha * dist(ee, soil_centroid)) # w2 = 1.0, alpha = 5.0

# ===== 辅助奖励 =====

# R3: 铲斗载荷奖励 (鼓励铲起土)
r_load = w3 * bucket_load_normalized                    # w3 = 2.0

# R4: 运输奖励 (铲起土后靠近目标区)
r_transport = w4 * bucket_load * exp(-beta * dist(ee, target))  # w4 = 3.0

# ===== 惩罚项 =====

# R5: 动作平滑惩罚
r_smooth = w5 * ||a_t - a_{t-1}||^2                    # w5 = -0.05

# R6: 关节限位惩罚
r_limit = w6 * sum(max(0, |q_i| - q_limit_i))          # w6 = -1.0

# R7: 时间效率惩罚 (每步小惩罚，鼓励快速完成)
r_time = w7                                              # w7 = -0.01

# R8: 碰撞惩罚 (防止铲斗撞地面/自身)
r_collision = w8 * has_collision                         # w8 = -5.0

# ===== 总奖励 =====
reward = r_transfer + r_approach + r_load + r_transport
       + r_smooth + r_limit + r_time + r_collision
```

### 2.4 终止条件 (Termination)

```python
# 成功终止
soil_in_target_ratio > 0.8       # 80% 以上土壤到达目标区

# 失败终止
time_steps > max_episode_length   # 超时 (如 500 步)
joint_out_of_limits               # 关节超限
bucket_below_ground               # 铲斗穿透地面 (物理异常)
robot_base_moved                  # 基座异常位移 (仿真 bug)

# 不终止但记录
soil_scattered                    # 土壤被打散到环境外 (需检查奖励设计)
```

### 2.5 Domain Randomization (EventManager)

```python
randomization_params = {
    # 土壤物理属性
    "soil_density":      (1400, 2200),    # kg/m^3 (干沙 → 湿粘土)
    "soil_friction":     (0.3, 0.9),      # 静摩擦系数
    "soil_restitution":  (0.0, 0.2),      # 弹性系数（土壤几乎无弹性）
    "soil_cohesion":     (0.0, 500.0),    # Pa (干沙 → 湿土)

    # 土壤几何
    "heap_height":       (0.15, 0.35),    # m
    "heap_radius":       (0.2, 0.4),      # m
    "heap_position_xy":  (-0.05, 0.05),   # m (位置偏移)

    # 机器人
    "joint_friction":    (0.8, 1.2),      # 相对基准值的比例
    "payload_mass":      (0.0, 0.5),      # kg (铲斗附加负载)
    "action_delay":      (0, 2),          # 步 (模拟通信延迟)
    "action_noise_std":  (0.0, 0.02),     # 动作噪声

    # 传感器
    "observation_noise":  0.01,           # 观测高斯噪声标准差
}
```

### 2.6 Curriculum Learning

```python
# 由易到难，分 3 个阶段

Stage 1 (0 ~ 200k steps):
  - 少量粒子 (N=200)
  - 土堆靠近目标区 (distance < 0.3m)
  - 无 domain randomization
  - 大 reward for approach (引导探索)

Stage 2 (200k ~ 600k steps):
  - 中等粒子 (N=500)
  - 土堆中等距离 (distance < 0.5m)
  - 轻度 domain randomization
  - 减小 approach reward，增大 transfer reward

Stage 3 (600k+ steps):
  - 完整粒子 (N=1000+)
  - 随机距离和方位
  - 完整 domain randomization
  - 只保留 transfer reward 为主导
```

---

## 3. 分阶段实现计划

### 阶段总览

```
Stage 0:    刚体简化版 — 验证 RL 训练流程跑通           [DONE]
Stage 1:    引入 Warp 粒子系统                          [BROUGHT UP, plateau 2.4% mean]
Stage 1.5:  BC + KL anchor + 粒子状态 replay (新加)     [SCAFFOLDED]
Stage 2:    完整训练 + 奖励迭代 + DR + Curriculum       [BLOCKED on Stage 1.5]
Stage 3:    消融实验 + 结果完善 + 展示材料              [BLOCKED on Stage 2]
```

**变更说明（2026-05-03）**：原计划假设 stage 1 只是"换粒子"，结果发现 10 个工程性 bug + 3 处 PPO 算法补丁 + 一个铲斗 AABB 结构性限制；纯 RL + reward shaping 在 v17→v22 共 7 轮后 plateau 在 2.4% 转移率。新增 Stage 1.5 处理 BC distribution shift，详见 Stage 1 Post-Mortem 子节。

### Stage 0: 刚体简化版 — 验证流程

**目标**：剥离粒子仿真的复杂性，先用刚体物体代替土壤，验证 RL 训练流程跑通。

**实现内容：**
1. Isaac Lab 场景：Franka + 1 个可推动的 rigid body cube + 目标区域
2. 观测：关节角 + 末端位姿 + cube 位置 + 目标位置
3. 奖励：推 cube 到目标区 → reward
4. 用 rsl_rl PPO 训练

**验收标准：**
- [ ] 4096 envs 并行训练稳定运行
- [ ] Agent 能学会把 cube 推到目标区
- [ ] 训练曲线收敛，reward 持续上升

### Stage 1: 引入 Warp 粒子系统

**目标**：用 Warp 粒子替换 rigid body cube，实现基本的铲斗-粒子交互。

**实现内容：**
1. Warp 粒子系统封装：创建、步进、查询粒子状态
2. 铲斗-粒子碰撞：铲斗作为刚体 collider，粒子对其响应
3. 粒子状态 → 高度图观测
4. 基于粒子位置的 reward（有多少粒子在目标区内）

**关键技术难点：**
- Isaac Lab 与 Warp 的集成（原生支持 vs 手动同步）
- GPU 上粒子仿真与刚体仿真的同步
- 粒子数量对训练速度的影响

**验收标准（实际进度，2026-05-03 更新）：**
- [x] Warp 粒子仿真独立 demo 可运行（铲斗推粒子）— `tests/standalone_bucket_sweep.md` 验证 `bucket_push_force = 1 N` 时粒子被合理推动 366 mm
- [ ] 集成进 Isaac Lab 环境 — 仍在 standalone 模式（CPU env loop + GPU 粒子物理）；Isaac Lab USD 集成挪到 Phase 1 TODO
- [ ] 粒子数 N=500 时仍能 >1000 envs 并行 — standalone 模式下因 CPU env loop 限制只能 32 envs；3 seeds × 32 envs 在 A10G 上并行（~80% GPU util）

---

#### Stage 1 实际偏离与纠偏 (Post-Mortem, v1 → v22)

原计划 Stage 1 = "把 rigid body 换成 Warp 粒子，把奖励切到 8 分量"，但实际过程暴露了**10 个原计划没考虑的工程性 bug**和**3 处 RL 算法缺陷**，以及一个**粒子物理结构性限制**。最终 v22 多 seed 结果：mean transfer 2.4%，单 seed 瞬时 peak 80%，PPO 健康度从灾难级（KL=14, clip=96%）拉回可用区（KL=0.4, clip=33%）。详见 `README.md#stage-1-results`。

**10 个被修复的 bug**（按发现顺序）：

| # | Bug | 影响 | Fix |
|---|-----|------|-----|
| 1 | `scene.build()` 从未被环境构造调用 | particle_system 是 None，stage 1 v1–v7 在跑"无粒子" | 加一行 `self.scene.build()` |
| 2 | `bucket_push_force` 默认用 `contact_damping = 1000 N`（unit confusion，N·s/m 当 N） | 粒子被推 90 米/20 步 | 加独立 `bucket_push_force: float = 1.0` 字段，standalone sweep 校准 |
| 3 | obs 缺归一化（66 维混单位：弧度+米+无量纲+m²） | KL = 7-13、clip = 0.95+ 在 v1–v10 全部出现 | `RunningMeanStd` (Welford) wrap actor-critic 输入 |
| 4 | CurriculumManager 默认 enabled，会静默覆盖显式 reward 权重 | 任何手动调权重都被覆盖 | stage 1 也加 `env_cfg.curriculum.enabled = False` |
| 5 | 默认 `init_noise_std = 1.0` 起点 entropy 过高 | v1 entropy 9.98 持续上升 | 降到 0.5 |
| 6 | 默认 `schedule = "adaptive"` 让 LR 在 v17 ablation 中塌到 5e-7 | policy 冻死 | stage 1 强制 `schedule = "fixed"` |
| 7 | 默认 `collision = -15` 是 binary 惩罚，produces bimodal advantage | std ≈ mean，PPO advantage 归一化失效 | stage 1 关掉，stage 2 改成梯度型按穿透深度 |
| 8 | `approach` reward 用动态 `soil_centroid`，铲入土时质心远离铲斗 | "成功 scoop → reward 下降"逆向激励 | 改用静态 `heap_center` |
| 9 | 铲斗几何是 AABB，无壁，contact kernel 升降时粒子直接掉穿 | `bucket_load` mean = 0.14（粒子从未真正留在桶里） | **Containment 启发式**：bucket interior 的粒子被 sticky-attached 到 bucket 框架，直到 bucket xy 进入 target zone 自动 release。v22 `reward/load` 涨到 390（2786×） |
| 10 | `action_smoothness = -0.01` 平均贡献 -10/iter，跟正向项同量级 | 压制 scoop 需要的快速动作 | 降到 -0.001，v18-v19 看到 transfer +71% |

**3 处 PPO/RL 算法补丁**：

1. **KL 早停**（PPO update loop 内）：每个 mini-batch 后看 `approx_kl > 2 × desired_kl` 就 break。原代码只把 `desired_kl` 用在 adaptive LR（schedule=fixed 时根本不调）
2. **BC pretraining**（`--bc-data --bc-steps`）：6400 (obs, action) pairs 来自 hand-coded 5 阶段 scoop 演示（IK 求解平面 2-link），2000 步 MSE。Demo replay transfer = **22%**，证明任务可解
3. **Reverse-curriculum reset**（`--reverse-curriculum`）：`env.reset()` 以概率 p 把 joint state 设到 demo 轨迹随机点。**只 reset joint 不 reset 粒子状态**——所以 reset 到 t=170（已在 target 上方）时土堆还是初始状态，agent 没东西可放。productionize 需要记录 + replay particle field

**结构性限制（未在原计划设想）**：

- **AABB 铲斗 + 简化 FK** 无法支持真实"舀取"动作（需要桶壁 + 倾角控制）。containment 是 stand-in。Isaac Lab + USD bucket mesh 才是终极解
- **BC distribution shift**：deterministic BC policy (no PPO) transfer = **0%**——demo 训练 MSE 收敛到 0.002 但闭环时小误差累积，~20 步后 EE 跑出 demo 状态分布，policy 输出饱和。BC + PPO 部分恢复（mean 2.4%）但 PPO 漂移会逐步覆盖 BC 知识

**对 Stage 2 的输入**：

| 原计划假设 | 实测推翻 / 修正 |
|---|---|
| "调权重 → 重训" 循环可以推到 transfer > 60% | 纯 reward shaping 在 v17 → v22 共 7 个 v 版本中只能从 1.4% 升到 2.4%，平台期是 distribution shift 不是 reward |
| Curriculum 是 stage 2 加的训练 trick | Curriculum 默认 enabled 会静默覆盖 reward；任何 stage 都需要先关 |
| Penalty 调强可避免 reward hacking | binary penalty 触发产生 bimodal advantage 反而破坏 PPO；stage 2 必须用 graduated penalty |
| Adaptive LR 适合稀疏 reward | adaptive LR 在 stage 0/1 都立刻塌到 floor；fixed 才稳 |

### Stage 1.5（新加）：BC + 强 warm-start + 粒子状态 replay

原计划没有 Stage 1.5。Stage 1 实证后插入这层是因为：纯 PPO 会卡在 transfer 2-3% 平台期，跳到 stage 2 加 curriculum/DR 解决不了 distribution shift 问题。

**目标**：让闭环 policy 接近 demo 的 22% transfer。

**实现内容**：
1. 扩大 demo 集（32 → 100+ 轨迹），加大 noise（0.02 → 0.05）覆盖更多 state
2. PPO 的 surrogate loss 加 KL anchor 项 `+ β · KL(π_current || π_BC)`，防止 PPO 把 BC 知识抹掉
3. Reverse-curriculum reset 加粒子状态 replay：在 demo 生成时记录每个 t 的粒子 positions，reset 时一并恢复
4. 训练流程：BC pretrain → 50 iter PPO (低 LR) → 解锁 KL anchor → 500 iter 标准 PPO

**验收标准**：
- [ ] BC-only deterministic transfer > 10%（解决 distribution shift）
- [ ] BC + PPO mean transfer > 10%（不被 PPO drift 抹掉）
- [ ] 多 seed 标准差 < mean（确认是 learned skill 不是 stochastic luck）

### Stage 2: 完整训练 + 奖励迭代

**目标**：在粒子版环境上完成 PPO 训练，迭代奖励函数直到 agent 行为合理。

**实现内容：**
1. 按 2.3 节的奖励函数实现，记录各项分量
2. 调参循环：观察行为 → 分析 reward 分量 → 调整权重 → 重新训练
3. 加入 curriculum learning
4. 加入 domain randomization

**每轮实验记录模板：**
```
实验 ID:     exp_007
修改内容:    增大 r_transport 权重 3.0 → 5.0
训练步数:    500k
最终 reward: 12.3 (上一轮: 8.7)
行为观察:    铲起土后会主动移向目标区，但倒土动作不准
下一步:      加入铲斗角度相关的倒土奖励
```

**验收标准：**
- [ ] Agent 能完成完整挖掘流程：接近 → 铲入 → 运输 → 倒出
- [ ] 土壤转移率 > 60%（原计划目标；Stage 1 demo 上限 22%，Stage 1.5 应能到 15-30%，Stage 2 加 curriculum/DR 后才有望接近 60%。如果含 BC + Isaac Lab 真实 bucket mesh，60% 是合理的）
- [ ] 训练曲线在 2M 步内收敛
- [ ] **graduated penalty 验证**：把 stage 1 关掉的 collision/joint_limit 改成连续梯度型（按穿透深度 / 限位余量）后重新加入，KL 不应回到 stage 1 v1-v17 的 7+ 区间

### Stage 3: 消融实验 + 结果完善

**目标**：通过消融实验证明设计决策的合理性，完善展示材料。

**验收标准：**
- [ ] 至少 3 组消融实验（见第 4 节）
- [ ] 训练曲线对比图
- [ ] 行为对比视频
- [ ] 完整 README

### 关键技术决策汇总

| 决策点 | 选项 A | 选项 B | 推荐 | 理由 | Stage 1 实证 |
|--------|--------|--------|------|------|--------------|
| 土壤表示 | Rigid body 近似 | Warp 粒子系统 | 先 A 再 B | 逐步增加复杂度 | ✓ 都跑通；粒子模式需要 containment 启发式补 AABB 桶无壁问题 |
| 动作空间 | 关节位置控制 | 关节力矩控制 | A | 更稳定易训练 | ✓ 关节位置 + 简化 FK 工作；但简化 FK + 7 关节 + 无 bucket 倾角是 transfer 上限的根因之一 |
| 观测空间 | 低维状态向量 | 点云 / 深度图 | 先低维 | 后续可扩展 | ⚠ 必须加 RunningMeanStd；不加则 KL 永远 7+。原计划没列这一项 |
| 奖励设计 | Dense shaped | Sparse (成功/失败) | Dense | 挖掘任务需持续引导 | ⚠ binary penalty (collision, joint_limit) 必须 graduated 化；binary 触发产生 bimodal advantage 破坏 PPO |
| 粒子观测 | 高度图 | PointNet 编码 | 先高度图 | 符合 Progressive-Resolution 思路 | ✓ 高度图 5×5 工作；与静态 heap_center 配合更稳（动态 centroid 会随铲入而远离 EE） |
| **PPO warm-start**（新加） | 无 / 课程学习 | BC + 演示 | **BC + reverse curriculum** | 纯 RL 在 dense reward 下卡 2-3% transfer 平台 | Stage 1 v17 → v22 实证：BC pretraining 把 KL 从 7.4 降到 0.4，但 BC distribution shift 让 deterministic 评估 = 0%；需 Stage 1.5 KL anchor 解决 |
| **铲斗物理**（新加） | AABB push kernel | 真实 mesh + tilt 关节 | 现 AABB + containment 启发式；终极 mesh | AABB 单独无法支持升降时含住粒子 | Containment 让 `reward/load` 提升 2786×；Isaac Lab + USD bucket mesh 是终极移除依赖 |

---

## 4. 验证体系

### 4.1 环境正确性验证（训练前必须全部通过）

| 检验项 | 方法 | 通过标准 |
|--------|------|---------|
| **观测范围** | 采集 10k 步 random policy 数据，打印 obs 每维的 min/max/mean/std | 数值在合理物理范围内（关节角在限位内，位置在场景范围内） |
| **奖励信号** | Random policy 下统计 reward 各分量分布 | Reward 不应全为 0 或全为常数；各分量量级可比 |
| **终止逻辑** | 手动构造边界 case（关节超限、超时、成功） | 每种条件都能正确触发终止 |
| **物理合理性** | GUI 模式下播放 random policy，肉眼检查 | 无穿模、无飞粒子、重力方向正确 |
| **可重复性** | 固定 seed 跑两次，对比 obs/reward 序列 | 完全一致（确认无未控制随机源） |
| **API 兼容性** | `env.observation_space.contains(obs)` + `env.action_space.contains(action)` | 每步都通过 |
| **粒子守恒** | 训练中监控粒子总数 | 粒子不会凭空消失或增加 |
| **Reset 一致性** | 连续 reset 100 次，检查初始 obs 分布 | 分布一致（有 DR 时在随机范围内，无 DR 时完全一致） |

### 4.2 训练过程监控指标

所有指标通过 TensorBoard 实时监控：

```
必须记录的指标:
──────────────────────────────────────────────────────
Performance:
  episodic_return          # 每回合总 reward → 应持续上升
  episodic_length          # 每回合步数 → 应趋于稳定
  soil_transfer_ratio      # 土壤转移率 (0~1) → 核心任务指标

Policy Health:
  policy_loss              # 策略损失 → 不应持续增大
  value_loss               # 价值损失 → 应持续下降
  entropy                  # 策略熵 → 应缓慢下降（非断崖式）
  clip_fraction            # PPO clip 比例 → 应在 0.05~0.20
  approx_kl                # KL 散度 → 应 < 0.05
  explained_variance       # Value 函数质量 → 应 > 0.5，理想 > 0.8

Reward Decomposition (调参关键):
  reward/transfer          # 土壤转移奖励 → 应随训练上升
  reward/approach          # 接近奖励 → 早期高，后期低
  reward/load              # 载荷奖励
  reward/transport         # 运输奖励
  reward/smooth_penalty    # 平滑惩罚 → 应逐步减小
  reward/collision_penalty # 碰撞惩罚 → 应趋近于 0

Environment:
  particle_in_target_count # 目标区粒子数
  bucket_load_avg          # 平均铲斗载荷
  contact_force_avg        # 平均接触力
──────────────────────────────────────────────────────
```

### 4.3 异常诊断表

| 现象 | 可能原因 | 排查方法 |
|------|---------|---------|
| Reward 不增长 | 奖励信号太稀疏 | 检查 random policy 下 reward 分布，加 dense 中间奖励 |
| Reward 先升后降 | Value function 过拟合或 KL 过大 | 检查 explained_variance 和 approx_kl |
| Entropy 骤降 | 策略过早收敛到局部最优 | 增大 entropy_coef (0.01 → 0.05) |
| clip_fraction > 0.3 | 步长太大 | 减小 learning rate 或增大 mini_batch_size |
| clip_fraction ≈ 0 | 几乎没有更新 | 增大 learning rate 或减少 n_epochs |
| value_loss 不降 | Value 网络容量不足或 lr 不匹配 | 增大网络或调整 value_coef |
| 碰撞惩罚持续很大 | 动作空间映射不合理 | 检查 action scaling，加关节限位保护 |
| 粒子数下降 | 仿真 bug，粒子飞出场景 | 加边界碰撞体，检查 Warp 参数 |

### 4.4 消融实验设计

每组 3 个随机种子取 mean ± std。

#### 实验 1: 奖励函数消融

| 配置 | 修改 | 验证点 |
|------|------|--------|
| `full` (baseline) | 完整 reward | 基准性能 |
| `no_approach` | 去掉 r_approach | 验证"接近引导"的必要性 |
| `no_load` | 去掉 r_load | 验证"铲斗载荷奖励"是否关键 |
| `sparse_only` | 只保留 r_transfer | 验证 dense reward 的价值 |

**预期结果**：full > no_approach ≈ no_load >> sparse_only

#### 实验 2: Domain Randomization 消融

| 配置 | 修改 | 验证点 |
|------|------|--------|
| `no_dr` | 固定物理参数 | 训练快但泛化差 |
| `soil_dr` | 只随机化土壤参数 | 土壤变化的影响 |
| `full_dr` | 所有参数随机化 | 最佳泛化 |

**验证方式**：
- 三个配置在 **固定标准参数** 下测试 → 比较 peak performance
- 三个配置在 **10 组随机参数** 下测试 → 比较泛化能力 (mean ± std)
- 画 train env performance vs test env performance 对比图

#### 实验 3: Curriculum Learning 消融

| 配置 | 修改 | 验证点 |
|------|------|--------|
| `no_curriculum` | 直接用最终难度训练 | 能否收敛 |
| `2_stage` | 简化为 2 阶段 | 必要的 curriculum 粒度 |
| `3_stage` | 完整 3 阶段 | 完整方案 |

**预期结果**：no_curriculum 可能不收敛或收敛慢；3_stage 最稳定

#### 实验 4: 观测空间对比 (NICE-TO-HAVE)

| 配置 | 修改 | 验证点 |
|------|------|--------|
| `height_map` | 5×5 高度图 | baseline |
| `height_map_10x10` | 10×10 高度图 | 分辨率影响 |
| `centroid_only` | 质心 + 方差 (6D) | 最小信息量 |

#### 实验 5: Sim-to-Real Gap 评估 (NICE-TO-HAVE)

- 改变物理参数到真实测量值范围外 ±50%
- 记录 policy 性能衰减曲线
- 目的：预估 real-world 部署风险

#### 消融实验输出格式

每组实验生成以下输出：

```
1. 训练曲线对比图 (x: timesteps, y: episodic_return)
   - 多条线在同一图中，附 mean ± std shading
   - 标注每个配置的最终 converged reward 值

2. 任务指标对比表
   ┌─────────────┬──────────────┬──────────────┬──────────────┐
   │ Config      │ Transfer Rate│ Converge Step│ Final Reward │
   ├─────────────┼──────────────┼──────────────┼──────────────┤
   │ full        │ 0.82 ± 0.03 │ 650k         │ 15.2 ± 1.1   │
   │ no_approach │ 0.61 ± 0.08 │ 900k         │ 10.3 ± 2.4   │
   │ ...         │ ...          │ ...          │ ...          │
   └─────────────┴──────────────┴──────────────┴──────────────┘

3. 行为对比视频 (每个配置录 1 段，拼在一起)
```

### 4.5 定量指标汇总

| 指标 | 定义 | 目标值 | 测量方法 |
|------|------|--------|---------|
| **挖掘体积精度** | `V_moved / V_target` | ≥ 0.85 | 粒子位移统计或 grid cell 占有率 |
| **Reward 收敛** | 最近 100 episode 平均 reward 变化率 | < 1% 持续 500K steps | TensorBoard `reward/mean` |
| **训练效率** | 达到 85% 体积精度所需总 step 数 | < 20M steps (4096 envs) | 训练 log |
| **动作平滑度** | 相邻 step 动作差 L2 范数均值 | 越小越好 | `\|a_t - a_{t-1}\|_2` |
| **能量消耗** | 关节力矩 × 角速度累积和 | 越小越好 | `Σ \|τ · ω\| dt` |
| **成功率** | Episode 结束时 V_moved/V_target ≥ 0.8 的比例 | ≥ 90% | 100 episode 评估 |
| **泛化能力** | 未见过的地形/土壤参数上的成功率 | ≥ 70% | DR 范围外测试 |

### 4.6 验证流程

```
训练完成
    │
    ▼
Step 1: 检查训练曲线 ────── reward 收敛? loss 稳定? entropy 合理衰减?
    │
    ▼
Step 2: 定量评估 ─────────── 跑 100 episodes, 计算 4.5 节所有指标
    │
    ▼
Step 3: 录制视频 ─────────── 选 best / median / worst episode 各录 1 个
    │
    ▼
Step 4: 消融实验 ─────────── 按 4.4 节逐项跑
    │
    ▼
Step 5: 整理结果 ─────────── 表格 + 曲线图 + 视频 → experiment_log.md
    │
    ▼
Step 6: 与 Terra 对比 ────── 相同场景, 对比行为和指标
```

### 4.7 最终验收 Checklist

```
═══════════════════════════════════════════════════════
         最终验收 (所有项必须通过)
═══════════════════════════════════════════════════════

□ 功能完整性
  □ 环境可创建、reset、step，无 crash
  □ 4096 envs 并行稳定运行 >30min
  □ 训练完整流程可一键启动 (python train.py)
  □ 评估脚本可加载 checkpoint 并输出指标

□ 任务性能
  □ 土壤转移率 (soil_transfer_ratio) > 0.6
  □ 训练在 2M 步内收敛
  □ 3 个随机种子结果一致 (std < 15% of mean)
  □ 明显优于 random policy baseline (至少 5x reward)

□ 分析完整性
  □ 至少 3 组消融实验，有对比图和结论
  □ 每个 reward 分量有独立的 TensorBoard 曲线
  □ 失败案例分析（什么情况下 agent 表现差，为什么）

□ 代码质量
  □ 可复现：README 中写明环境、依赖、训练命令
  □ 固定 seed 可精确复现
  □ 配置与代码分离 (yaml / dataclass)
  □ 关键模块有注释说明设计意图

□ 展示材料
  □ README：问题描述、方法、架构图、结果
  □ 训练曲线截图（含消融对比）
  □ Agent 行为 GIF / 视频（训练前 vs 训练后）
  □ 不同土壤条件下的泛化能力展示
═══════════════════════════════════════════════════════
```

---

## 5. 与 RSL 论文的对齐分析

| 维度 | 本项目方案 | RSL Soil-Adaptive (2022) | 差距与说明 |
|------|-----------|--------------------------|-----------|
| 仿真器 | Isaac Lab + Warp | 自研仿真器 (Vortex) | 仿真器不同但思路一致 |
| 土壤模型 | Warp 粒子 | FEM / 颗粒法 | 粒子更轻量，适合大规模并行 |
| 机器人 | Franka / UR10 | 真实挖掘机 | 本项目是前期验证，用标准臂即可 |
| 观测 | 高度图 + 本体感知 | 力传感 + 本体感知 | 可后期加入力反馈 |
| RL 算法 | PPO (rsl_rl) | PPO | 一致 |
| Sim-to-real | Domain randomization | DR + system ID | 本项目先做 sim，为 real 做准备 |
| 创新点 | 粒子仿真 + Isaac Lab 集成 | 力自适应策略 | 本项目侧重仿真平台能力 |

---

## 6. 风险与备选方案

| 风险 | 可能性 | 影响 | 应对方案 |
|------|--------|------|---------|
| Warp 粒子与 Isaac Lab RL 循环集成困难 | 高 | 中 | 先做独立 Warp 粒子 demo + 独立 Isaac Lab 刚体环境，两者分别展示 |
| 粒子数过多导致并行数受限 | 中 | 中 | 减少粒子数 (200~500)，用高度图聚合掩盖细节不足 |
| 铲斗-粒子交互不稳定 | 中 | 高 | 简化铲斗为平面 collider，或用 Newton 引擎替代 |
| 4096 环境并行 OOM | 中 | 中 | 减少 num_envs，或申请 Euler 集群大显存节点 |
| Reward shaping 导致局部最优 | 中 | 高 | 准备多套 reward 配置，系统做 ablation |
| Isaac Lab / Isaac Sim 版本兼容 | 高 | 低 | 锁定版本，记录完整安装步骤 |
| 训练不收敛 | 中 | 高 | 从官方 Lift-Cube 环境开始，逐步增加复杂度 |
| 奖励稀疏难以学习 | 中 | 高 | 加入 demonstration-guided reward / 人工势场引导 |

---

## 7. 技能可迁移性

即使最终未获得此 excavation 项目，各阶段技能可直接迁移：

| 内容 | 迁移率 | 适用方向 |
|------|--------|---------|
| PPO 手写 + RL 基础 | 100% | 任何 RL 相关 semester project / 实习 |
| Gymnasium 自定义环境 | 100% | 任何机器人 RL 项目 |
| Isaac Lab 实操 | 90% | RSL locomotion, manipulation; ASL, CRL 项目; NVIDIA 相关岗位 |
| Warp 粒子仿真 | 50% | 仿真物理建模方向（相对小众） |
| Terra 复现 + JAX | 60% | JAX 驱动力项目, Terra 后续研究 |
| 论文阅读 + 知识积累 | 30% | Excavation-specific，但论文阅读能力通用 |
| GitHub Portfolio | 100% | 任何申请 |

**核心结论**：Isaac Lab 环境搭建 + PPO 训练是必须完成的部分，即使项目方向调整也不会浪费。

---

## 8. 关键参考资料

### 核心论文

| 论文 | 与本项目关系 |
|------|-------------|
| Soil-Adaptive Excavation Using RL (RA-L 2022) | RSL 挖掘方向奠基论文，本项目直接前序 |
| Progressive-Resolution Policy Distillation (2024) | RSL 挖掘最新工作，多分辨率策略蒸馏 |
| RSL-RL: A Learning Library for Robotics | 训练框架设计哲学与 API |
| Learning to Walk in Minutes (RSL) | Sim-to-real 与大规模并行训练方法论 |
| Isaac Lab (arXiv 2511.04831) | 仿真平台论文 |

### 技术文档

| 资源 | 用途 |
|------|------|
| [Isaac Lab 官方文档](https://isaac-sim.github.io/IsaacLab/main/) | 环境搭建、API 参考 |
| [rsl_rl GitHub](https://github.com/leggedrobotics/rsl_rl) | 训练库安装与配置 |
| [NVIDIA Warp 文档](https://nvidia.github.io/warp/) | 粒子仿真 API |
| [Terra GitHub](https://github.com/leggedrobotics/terra) | Baseline 复现 |
| [Spinning Up (OpenAI)](https://spinningup.openai.com/) | RL 理论基础 |
| [37 PPO Implementation Details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) | PPO 手写参考 |
| [CleanRL](https://github.com/vwxyzjn/cleanrl) | 单文件 RL 实现参考 |
