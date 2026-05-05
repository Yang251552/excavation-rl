# Excavation RL 代码详解与修改记录

> 本文档记录了整个项目从零搭建到调通的完整过程，包括每个模块的设计意图、
> 遇到的 Bug 及修复思路。适合逐步学习。

---

## 目录

1. [项目整体架构](#1-项目整体架构)
2. [模块一：机器人配置 robot/](#2-模块一机器人配置-robot)
3. [模块二：土壤粒子系统 soil/](#3-模块二土壤粒子系统-soil)
4. [模块三：RL 环境核心 envs/](#4-模块三rl-环境核心-envs)
5. [模块四：PPO 训练 training/](#5-模块四ppo-训练-training)
6. [模块五：评估系统 evaluation/](#6-模块五评估系统-evaluation)
7. [Bug 修复记录](#7-bug-修复记录)
8. [数据流完整走一遍](#8-数据流完整走一遍)
9. [关键设计决策与权衡](#9-关键设计决策与权衡)
10. [下一步TODO](#10-下一步todo)

---

## 1. 项目整体架构

### 1.1 文件结构（31 个 Python 文件，~5600 行代码）

```
excavation-rl/
├── envs/                              # RL 环境（最核心的模块）
│   ├── __init__.py                    # 暴露 ExcavationEnv, ExcavationEnvCfg
│   ├── excavation_env_cfg.py    [298行] # 所有配置的单一来源
│   ├── excavation_env.py        [497行] # 主环境：reset() / step() 循环
│   ├── scene.py                 [259行] # 物理场景管理（机器人+土+地面）
│   ├── observations.py          [208行] # 观测向量的拼装
│   ├── rewards.py               [200行] # 8 项奖励分量
│   ├── terminations.py          [149行] # 终止条件判断
│   ├── curriculum.py            [123行] # 课程学习（由易到难）
│   └── events.py                [181行] # 域随机化
│
├── soil/                              # 土壤粒子仿真
│   ├── __init__.py
│   ├── particle_system.py       [344行] # Warp GPU 粒子系统
│   ├── soil_properties.py       [145行] # 5种土壤材质参数
│   └── soil_terrain.py          [206行] # 土堆形状生成 + 高度图
│
├── robot/                             # 机械臂配置
│   ├── __init__.py
│   ├── arm_cfg.py               [164行] # Franka / UR10 参数
│   └── end_effector.py          [104行] # 铲斗末端执行器
│
├── training/                          # PPO 训练管道
│   ├── __init__.py
│   ├── train.py                 [585行] # ActorCritic + RolloutBuffer + PPOTrainer
│   ├── ppo_cfg.py               [137行] # 超参数配置（3个阶段）
│   └── callbacks.py             [152行] # TensorBoard 日志 + checkpoint
│
├── evaluation/                        # 训练后分析
│   ├── __init__.py
│   ├── evaluate.py              [199行] # 定量评估脚本
│   ├── metrics.py               [165行] # 指标计算
│   ├── ablation.py              [287行] # 消融实验生成器
│   ├── visualize.py             [282行] # 训练曲线绘图
│   └── record_video.py          [245行] # 视频录制
│
├── configs/                           # YAML 配置
│   ├── base.yaml                      # 完整默认配置
│   ├── reward_ablation.yaml           # 奖励消融 4 组配置
│   └── dr_ablation.yaml              # 域随机化消融 3 组配置
│
├── tests/                             # 47 个单元测试
│   ├── test_env.py              [232行] # 环境 API、观测范围、可复现性
│   ├── test_reward.py           [229行] # 奖励各分量正确性
│   └── test_soil.py             [145行] # 土壤生成、高度图
│
├── setup.py                           # pip install -e .
├── requirements.txt                   # 依赖列表
└── README.md                          # 项目说明
```

### 1.2 数据流（一步 step 的完整路径）

```
                    ┌──────────────┐
                    │   PPO Agent  │
                    │  (train.py)  │
                    └──────┬───────┘
                           │ action (7D, [-1,1])
                           ▼
         ┌─────────────────────────────────────┐
         │        ExcavationEnv.step()          │
         │                                     │
         │  1. action × 0.1 → joint delta      │
         │  2. clip to joint limits             │
         │  3. update joint_pos                 │
         │  4. _update_ee_from_joints() → FK    │
         │  5. scene.step_physics(ee_pos, vel)  │
         │  6. 计算 soil_in_target_ratio        │
         │  7. compute_reward() → 8 项分量       │
         │  8. check_termination()              │
         │  9. build_observation() → 66D 向量    │
         └─────────────┬───────────────────────┘
                       │
              (obs, reward, done, info)
                       │
                       ▼
                    PPO 更新
```

### 1.3 模块依赖关系

```
configs (YAML)
    │
    ▼
excavation_env_cfg.py  ←── 所有模块共享的配置定义
    │
    ├── robot/arm_cfg.py + end_effector.py
    │
    ├── soil/particle_system.py + soil_properties.py + soil_terrain.py
    │
    ├── envs/scene.py     ←── 组合 robot + soil
    ├── envs/observations.py
    ├── envs/rewards.py
    ├── envs/terminations.py
    ├── envs/curriculum.py
    ├── envs/events.py
    │
    └── envs/excavation_env.py  ←── 最终环境，组合以上所有
         │
         ├── training/train.py  ←── PPO 训练
         └── evaluation/        ←── 评估分析
```

---

## 2. 模块一：机器人配置 robot/

### 2.1 arm_cfg.py — 机械臂参数

**设计目的**：把机械臂的物理参数集中定义，方便切换 Franka / UR10。

```python
@dataclass
class ArmCfg:              # 基类
    num_joints: int = 7
    joint_limits: JointLimits   # 关节角度上下限 (rad)
    default_joint_pos: list     # 默认"home"姿态
    stiffness / damping: float  # PD 控制器增益
    max_joint_delta: float      # action scale: [-1,1] × 0.1 = 最大每步 0.1 rad

class FrankaArmCfg(ArmCfg):   # Franka Panda 7DOF 具体参数
class UR10ArmCfg(ArmCfg):     # UR10 6DOF 具体参数
```

**关键修改**：

原始默认关节角 `[0, -0.569, 0, -2.810, 0, 3.037, 0.741]` 导致 EE 指向后方
（x=-0.107），而土堆在前方 (x=0.5)，永远够不到。

修改为 `[0, 0.4, 0, -1.0, 0, 1.4, 0]`，手臂前伸，初始 EE 位于 x≈0.6。

```python
# 修改前（手臂缩回）
default_joint_pos = [0, -0.569, 0, -2.810, 0, 3.037, 0.741]
# 简化FK → ee_x = -0.107  ← 指向后方！

# 修改后（手臂前伸）
default_joint_pos = [0, 0.4, 0, -1.0, 0, 1.4, 0]
# 简化FK → ee_x = 0.608   ← 指向前方，靠近土堆
```

### 2.2 end_effector.py — 铲斗

定义铲斗的几何参数和载荷检测区域：

```python
@dataclass
class BucketGeometry:
    width: 0.15m, depth: 0.12m, height: 0.08m
    → volume ≈ 0.001 m³
    → 最多装 ~115 颗粒子（半径 0.005m，随机堆积 60%）

@dataclass
class BucketEndEffectorCfg:
    geometry: BucketGeometry
    offset_pos: 铲斗相对 EE 法兰的安装偏移
    get_load_detection_bounds(): 返回铲斗内部的 AABB → 用于数粒子
```

---

## 3. 模块二：土壤粒子系统 soil/

### 3.1 soil_properties.py — 土壤材质

5 种预设土壤类型，每种有完整的物理参数：

```python
SoilType.DRY_SAND:  density=1500, friction=0.5, cohesion=0     # 干沙
SoilType.WET_SAND:  density=1900, friction=0.7, cohesion=200   # 湿沙
SoilType.CLAY:      density=2000, friction=0.8, cohesion=500   # 粘土
SoilType.GRAVEL:    density=1800, friction=0.6, cohesion=0     # 砾石
SoilType.LOAM:      density=1400, friction=0.65, cohesion=150  # 壤土

# 粒子质量从密度自动计算：
particle_mass = density × (4/3)π × radius³
```

`randomize_soil_properties()` 在给定范围内均匀采样，用于域随机化。

### 3.2 soil_terrain.py — 土堆生成

4 种初始堆形（用拒绝采样生成粒子位置）：

```
CONE (锥形)         HEMISPHERE (半球)    CYLINDER (圆柱)    FLAT (平铺)
    ▲                   ⌒                ┌──┐             ─────
   ╱ ╲                 / \               │  │
  ╱   ╲              /    \              │  │
 ╱─────╲            ╰──────╯             └──┘
```

**核心函数**：
```python
generate_particle_positions(rng) → (N, 3) numpy array   # 生成 N 个粒子的初始位置
is_in_target(positions) → bool array                     # 哪些粒子在目标区域内
compute_height_map(positions, grid_resolution=5) → (5,5) # 2D 高度图观测
```

**高度图原理**：
```
将水平面划分为 5×5 网格，记录每个格子内最高粒子的 z 值：

  ┌─────┬─────┬─────┬─────┬─────┐
  │ 0.0 │ 0.0 │ 0.0 │ 0.0 │ 0.0 │  ← 无粒子区域 z=0
  ├─────┼─────┼─────┼─────┼─────┤
  │ 0.0 │ 0.1 │ 0.2 │ 0.1 │ 0.0 │  ← 锥形边缘
  ├─────┼─────┼─────┼─────┼─────┤
  │ 0.0 │ 0.2 │ 0.25│ 0.2 │ 0.0 │  ← 锥形顶部
  ├─────┼─────┼─────┼─────┼─────┤
  │ 0.0 │ 0.1 │ 0.2 │ 0.1 │ 0.0 │
  ├─────┼─────┼─────┼─────┼─────┤
  │ 0.0 │ 0.0 │ 0.0 │ 0.0 │ 0.0 │
  └─────┴─────┴─────┴─────┴─────┘

展平为 25D 向量 → 作为观测的一部分
```

### 3.3 particle_system.py — Warp GPU 粒子仿真

基于 NVIDIA Warp 的 GPU 并行粒子系统。三个 Warp kernel：

```python
@wp.kernel _integrate_particles(...)     # 半隐式欧拉积分 + 地面碰撞
@wp.kernel _apply_bucket_force(...)      # 铲斗-粒子接触力
@wp.kernel _count_particles_in_region(...)  # 统计区域内粒子数（原子操作）
```

**积分逻辑（每个粒子独立并行）**：
```
加速度 = 重力 + 外力/质量 - 阻尼×速度
速度 += 加速度 × dt
位置 += 速度 × dt
if 位置.z < 地面:
    位置.z = 地面
    速度.z = -速度.z × 弹性系数
    速度.xy *= (1 - 摩擦)
```

> **注意**：Warp 需要 NVIDIA GPU。在无 GPU 环境下，`import warp` 会被跳过，
> 此时只能用刚体代理模式（Stage 0）。

---

## 4. 模块三：RL 环境核心 envs/

### 4.1 excavation_env_cfg.py — 全局配置中心

这是**最重要的文件之一**，所有可调参数都在这里：

```python
@dataclass
class ExcavationEnvCfg:          # 顶层配置
    seed: int = 42
    sim_dt: float = 1/60         # 仿真步长 16.7ms
    device: str = "cuda:0"

    scene: SceneCfg              # 场景布局
    observation: ObservationCfg  # 观测空间
    action: ActionCfg            # 动作空间
    reward: RewardCfg            # 奖励函数
    termination: TerminationCfg  # 终止条件
    domain_randomization: DomainRandomizationCfg  # 域随机化
    curriculum: CurriculumCfg    # 课程学习
```

**关键修改 — 场景布局调整**：

```python
# 修改前：土堆和目标分别在 x=0.4 和 x=-0.4
# 问题：Franka 臂展有限，两个位置都在工作空间边缘
soil_heap_center = (0.4, 0.0, 0.0)
target_center = (-0.4, 0.0, 0.0)

# 修改后：都放在 Franka 前方可达范围内
soil_heap_center = (0.5, 0.0, 0.05)    # 正前方，略高于地面
target_center = (0.4, 0.3, 0.05)        # 偏右侧（同样在可达范围内）
```

### 4.2 scene.py — 物理场景管理

组合机器人、土壤、地面为一个统一的场景，提供查询接口：

```python
class ExcavationScene:
    # 查询接口
    get_soil_in_target_ratio() → float     # 有多少比例的土在目标区
    get_soil_height_map() → (5,5) array    # 高度图观测
    get_soil_centroid() → (3,) array       # 土壤质心
    get_bucket_load(bucket_pos) → float    # 铲斗内的土壤质量

    # 物理步进
    step_physics(bucket_pos, bucket_vel)   # 推进一步物理仿真
```

**关键修改 — 添加刚体代理的碰撞物理**：

原始代码中，刚体代理模式（Stage 0）的 `step_particles()` 对刚体什么都不做，
导致 cube 永远不动。

```python
# 修改前
def step_particles(self, bucket_pos, bucket_vel):
    if not self._use_particles:
        return  # ← 直接返回，cube 永远不动！

# 修改后：添加简化碰撞物理
def step_physics(self, bucket_pos, bucket_vel, dt):
    if self._use_particles:
        # Warp 粒子仿真（原有逻辑）
        ...
    elif self.rigid_body_pos is not None:
        # ★ 新增：刚体代理碰撞物理
        diff = self.rigid_body_pos - bucket_pos
        dist = np.linalg.norm(diff)
        contact_radius = 0.12 + self._rigid_body_size / 2

        if dist < contact_radius and dist > 1e-6:
            # 弹簧接触力：penetration × 刚度
            normal = diff / dist
            penetration = contact_radius - dist
            push_force = normal * penetration * 80.0
            # 速度传递：铲斗速度的一部分传给 cube
            push_force += bucket_vel * 3.0
            # 加速度 → 速度
            self.rigid_body_vel += (push_force / mass) * dt

        # 重力
        self.rigid_body_vel[2] -= 9.81 * dt
        # 阻尼
        self.rigid_body_vel *= (1 - damping * dt)
        # 积分
        self.rigid_body_pos += self.rigid_body_vel * dt
        # 地面约束
        if self.rigid_body_pos[2] < ground_z:
            self.rigid_body_pos[2] = ground_z
            self.rigid_body_vel[2] = 0
```

同时还修复了高度图和 bucket load 在刚体模式下返回全零的问题。

### 4.3 observations.py — 观测向量

**观测空间组成（共 66 维）**：

```
┌──────────────────────────────────┬──────┬───────────┐
│ 分量                             │ 维度 │ 取值范围   │
├──────────────────────────────────┼──────┼───────────┤
│ joint_positions                  │  7   │ rad       │
│ joint_velocities                 │  7   │ rad/s     │
│ ee_position                      │  3   │ m         │
│ ee_orientation (quaternion)      │  4   │ [-1, 1]   │
│ ee_linear_velocity               │  3   │ m/s       │
│ ee_angular_velocity              │  3   │ rad/s     │
│ bucket_load (normalized)         │  1   │ [0, 1]    │
│ bucket_contact_force             │  3   │ N         │
│ soil_height_map (5×5 展平)        │ 25   │ m (≥0)    │
│ target_position                  │  3   │ m         │
│ previous_action                  │  7   │ [-1, 1]   │
├──────────────────────────────────┼──────┼───────────┤
│ 总计                              │ 66   │           │
└──────────────────────────────────┴──────┴───────────┘
```

**设计模式**：用独立的 dataclass 封装各部分状态：

```python
RobotState     → 关节角、速度、EE 位姿
BucketState    → 载荷、接触力
SoilObservation → 高度图 / 质心 / 方差

build_observation(robot, bucket, soil, target, prev_action) → 拼接为 1D 向量
```

### 4.4 rewards.py — 8 项奖励函数

这是训练效果的关键。每一项独立计算，独立记录到 TensorBoard：

```
┌───┬─────────────────┬────────┬──────────────────────────────────┐
│ # │ 名称             │ 权重   │ 公式                              │
├───┼─────────────────┼────────┼──────────────────────────────────┤
│ 1 │ soil_transfer   │ +10.0  │ w × Δ(soil_in_target_ratio)      │
│ 2 │ approach        │ +1.0   │ w × exp(-5 × dist(ee, soil))     │
│ 3 │ bucket_load     │ +2.0   │ w × normalized_load              │
│ 4 │ transport       │ +3.0   │ w × load × exp(-5×dist(ee,tgt))  │
│ 5 │ smoothness      │ -0.05  │ w × ‖a_t − a_{t-1}‖²            │
│ 6 │ joint_limit     │ -1.0   │ w × Σ max(0, |q_i| − limit_i)   │
│ 7 │ time_penalty    │ -0.01  │ w (常数)                          │
│ 8 │ collision       │ -5.0   │ w × has_collision                 │
│   │ success_bonus   │ +50.0  │ 一次性（土壤比例 ≥ 80%）           │
└───┴─────────────────┴────────┴──────────────────────────────────┘

总奖励 = R1 + R2 + R3 + R4 + R5 + R6 + R7 + R8 + bonus
```

**关键设计思路**：

- R1 (transfer) 是最终目标，但一开始信号太稀疏（agent 还不会碰到土）
- R2 (approach) 引导 EE 靠近土壤 → 提供早期学习信号
- R3 (load) 鼓励铲起土壤
- R4 (transport) 鼓励装满后移向目标
- R5-R8 是惩罚项，防止不良行为

**课程学习的奖励权重覆盖**：

```python
Stage 1: approach=2.0, transfer=10.0   # 重引导，轻目标
Stage 2: approach=0.5, transfer=10.0   # 减少引导
Stage 3: approach=0.1, transfer=15.0   # 几乎只看目标
```

### 4.5 terminations.py — 终止条件

```python
# 优先级从高到低检查
1. SUCCESS:            soil_ratio ≥ 0.8     → terminated=True
2. JOINT_LIMIT:        关节角超限 ±0.01 rad  → terminated=True
3. GROUND_PENETRATION: bucket_z < -0.02m    → terminated=True
4. BASE_DISPLACEMENT:  base 移动 > 0.01m    → terminated=True
5. TIMEOUT:            step ≥ 500           → truncated=True
```

**注意区分 terminated 和 truncated**：
- `terminated=True`：任务完成或失败，需要 bootstrap value = 0
- `truncated=True`：只是时间到了，需要 bootstrap value = V(s_T)

### 4.6 curriculum.py — 课程学习

三阶段递进难度：

```
Stage 1 (0~200K步):    200粒子, 近距离(0.3m), 无DR, 强引导
     ↓ 自动切换
Stage 2 (200K~600K步): 500粒子, 中距离(0.5m), 轻DR, 弱引导
     ↓ 自动切换
Stage 3 (600K+步):     1000粒子, 远距离(0.8m), 全DR, 几乎无引导
```

`CurriculumManager.update(total_steps)` 每步调用，自动判断是否切换阶段。

### 4.7 events.py — 域随机化

每次 episode reset 时，从配置范围内随机采样 13 个参数：

```python
RandomizedParams:
    soil:   density [1400, 2200], friction [0.3, 0.9], cohesion [0, 500], ...
    heap:   height [0.15, 0.35], radius [0.2, 0.4], position offset ±0.05m
    robot:  joint_friction ×[0.8, 1.2], payload [0, 0.5kg], action delay [0, 2步]
    sensor: observation noise σ = 0.01
```

域随机化只在课程 Stage 2+ 启用。

### 4.8 excavation_env.py — 主环境

**核心循环 `step(action)`**：

```python
def step(self, action):
    # 1. 域随机化处理
    action = apply_action_noise(action)       # 加动作噪声
    action = apply_action_delay(action)       # 模拟延迟

    # 2. 关节控制
    joint_delta = action * 0.1                # scale: [-1,1] → [-0.1, 0.1] rad
    target = clip(joint_pos + delta, limits)  # clip 到关节限位
    joint_pos = target                        # 瞬时位置跟踪
    _update_ee_from_joints()                  # 简化 FK → EE 位置

    # 3. 物理仿真
    scene.step_physics(ee_pos, ee_vel)        # 推进粒子/刚体

    # 4. 状态查询
    soil_ratio = scene.get_soil_in_target_ratio()
    centroid = scene.get_soil_centroid()
    load = scene.get_bucket_load(ee_pos)

    # 5. 奖励计算（8 项）
    reward_components = compute_reward(state, cfg, curriculum_overrides)

    # 6. 终止检查
    result = check_termination(state, cfg)

    # 7. 观测构建（66 维）
    obs = build_observation(robot, bucket, soil, target, prev_action)

    return obs, reward, terminated, truncated, info
```

**关键修改 — 简化 FK 改进**：

```python
# 修改前：
self._ee_pos[2] = l1 + l2*sin(q1) + l3*sin(q1+q3)
# 问题：当 q1+q3 使 sin 为负时，z 可以变成负数 → 穿地 → 提前终止

# 修改后：
self._ee_pos[2] = base_height + l2*sin(q1) + l3*sin(q1+q3)
self._ee_pos[2] = max(self._ee_pos[2], 0.005)  # ★ 地面 clamp
# 并修复 EE 速度计算：
self._ee_lin_vel = (self._ee_pos - prev_ee_pos) / dt  # 而非粗略缩放
```

**关键修改 — reset 中 EE 速度清零**：

```python
# 修改前：
self._update_ee_from_joints()   # 内部计算 vel = (pos - prev) / dt
# 问题：prev_ee_pos 是上个 episode 的残留 → 第一帧速度不为零 → 不可复现

# 修改后：
self._ee_lin_vel = np.zeros(3)  # 先清零
self._update_ee_from_joints()   # FK 计算（会覆盖 vel）
self._ee_lin_vel = np.zeros(3)  # 再清零（reset 时速度应为 0）
```

---

## 5. 模块四：PPO 训练 training/

### 5.1 ActorCritic 网络

```python
class ActorCritic(nn.Module):
    policy_net: Linear(66→256) → ELU → Linear(256→256) → ELU → Linear(256→128) → ELU → Linear(128→7)
    value_net:  Linear(66→256) → ELU → Linear(256→256) → ELU → Linear(256→128) → ELU → Linear(128→1)
    log_std:    Parameter(7,)  # 可学习的动作标准差

    # 输出高斯分布：action ~ N(mean, exp(log_std))
```

权重初始化：`orthogonal_(gain=√2)`，偏置归零。

### 5.2 RolloutBuffer

存储一轮采样的 transitions，用于 PPO 更新：

```python
observations:  (num_steps, num_envs, 66)
actions:       (num_steps, num_envs, 7)
rewards:       (num_steps, num_envs)
dones:         (num_steps, num_envs)
values:        (num_steps, num_envs)
log_probs:     (num_steps, num_envs)

# GAE 计算
compute_returns_and_advantages():
    for t in reversed(range(num_steps)):
        delta = r_t + γ·V(s_{t+1})·(1-done) - V(s_t)
        advantage_t = delta + γ·λ·(1-done)·advantage_{t+1}
    returns = advantages + values
```

### 5.3 PPO 更新

每轮迭代：
1. 收集 `num_steps_per_env × num_envs` 个 transition
2. 计算 GAE advantage
3. 分成 mini-batch，执行 `num_epochs` 轮更新

```python
# 核心 loss
ratio = exp(new_log_prob - old_log_prob)
surr1 = ratio × advantage
surr2 = clip(ratio, 1-ε, 1+ε) × advantage
policy_loss = -min(surr1, surr2).mean()

value_loss = 0.5 × max(
    (V - returns)²,
    (clip(V, old_V-ε, old_V+ε) - returns)²
).mean()

entropy_loss = -entropy.mean()

total_loss = policy_loss + 1.0 × value_loss + 0.01 × entropy_loss
```

**自适应学习率**：
```python
if approx_kl > 2 × desired_kl:   lr /= 1.5  # KL 太大，步子太大
if approx_kl < 0.5 × desired_kl: lr *= 1.5  # KL 太小，步子太小
```

### 5.4 Checkpoint 保存

**关键修改 — 保存网络结构到 checkpoint**：

```python
# 修改前：只存权重
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}

# 修改后：同时存网络结构信息
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "num_obs": 66,
    "num_actions": 7,
    "policy_hidden_dims": [128, 128],   # ★ 新增
    "value_hidden_dims": [128, 128],    # ★ 新增
    "activation": "elu",                # ★ 新增
}
```

这样评估脚本加载时就能自动用正确的网络结构。

---

## 6. 模块五：评估系统 evaluation/

### 6.1 evaluate.py — 定量评估

**关键修改 — 自动推断网络结构**：

```python
# 修改前：硬编码默认网络结构
network_cfg = NetworkCfg()  # 默认 [256, 256, 128]
model = ActorCritic(..., policy_hidden_dims=network_cfg.policy_hidden_dims)
model.load_state_dict(checkpoint["model_state_dict"])
# 错误！Stage 0 训练用的是 [128, 128]，尺寸不匹配

# 修改后：从 checkpoint 中读取
checkpoint = torch.load(path)
network_cfg = NetworkCfg(
    policy_hidden_dims=checkpoint.get("policy_hidden_dims", [256, 256, 128]),
    value_hidden_dims=checkpoint.get("value_hidden_dims", [256, 256, 128]),
    policy_activation=checkpoint.get("activation", "elu"),
)
model = ActorCritic(..., policy_hidden_dims=network_cfg.policy_hidden_dims)
model.load_state_dict(checkpoint["model_state_dict"])  # ✓ 尺寸匹配
```

### 6.2 metrics.py — 指标定义

```python
@dataclass
class EpisodeMetrics:
    total_reward: float           # 总奖励
    episode_length: int           # 步数
    soil_transfer_ratio: float    # 核心指标：多少土到了目标区
    success: bool                 # soil_ratio ≥ 0.8?
    action_smoothness: float      # mean ‖a_t - a_{t-1}‖₂
    energy_consumption: float     # Σ|torque × velocity| × dt

@dataclass
class EvaluationReport:
    mean_reward ± std             # 跨 episode 统计
    success_rate                  # 成功率
    ...
```

### 6.3 ablation.py — 消融实验

预配置 4 组消融实验，每组多个配置 × 多个 seed：

```
实验 1: 奖励消融
  full (baseline) / no_approach / no_load / sparse_only
  → 验证每项奖励分量的必要性

实验 2: 域随机化消融
  no_dr / soil_dr / full_dr
  → 验证 DR 对泛化的影响

实验 3: 课程学习消融
  no_curriculum / 2_stage / 3_stage
  → 验证渐进难度的必要性

实验 4: 观测空间对比 (可选)
  height_map_5x5 / height_map_10x10 / centroid_only
  → 验证土壤表示方式的影响
```

---

## 7. Bug 修复记录

### Bug 1: 评估脚本 state_dict 尺寸不匹配

**现象**：
```
RuntimeError: size mismatch for policy_net.0.weight:
  copying a param with shape [128, 66] from checkpoint,
  the shape in current model is [256, 66]
```

**根因**：Stage 0 用 `[128, 128]` 网络训练，评估脚本默认用 `[256, 256, 128]` 加载。

**修复**：
1. `training/train.py` — checkpoint 新增 `policy_hidden_dims` 等字段
2. `evaluation/evaluate.py` — 从 checkpoint 读取网络结构
3. `evaluation/record_video.py` — 同上

**文件改动**：`train.py` (5行), `evaluate.py` (15行), `record_video.py` (10行)

---

### Bug 2: 刚体代理不动（Soil Transfer 永远为 0）

**现象**：训练和评估中 `soil_in_target_ratio` 始终为 0.000。

**根因**：`scene.py` 的 `step_particles()` 在刚体代理模式下直接 `return`，
cube 没有任何物理交互，永远停在初始位置 `(0.5, 0, 0.05)`。

**修复**：在 `scene.py` 添加 `step_physics()` 方法，刚体模式下实现：
- 弹簧接触力模型（penetration × stiffness）
- 铲斗速度传递
- 重力、阻尼、地面约束

**文件改动**：`scene.py` (重写 30+ 行物理逻辑)

---

### Bug 3: EE 穿地导致提前终止

**现象**：Episode 在 ~119 步提前结束，原因是 `ground_penetration`。

**根因**：简化 FK 在某些关节角组合下计算出 `ee_z < 0`，触发穿地终止。

**修复**：
1. 在 FK 末尾添加 `self._ee_pos[2] = max(ee_z, 0.005)` 地面 clamp
2. 改进 EE 速度计算为 `(pos - prev_pos) / dt`（而非粗略缩放）

**文件改动**：`excavation_env.py` (_update_ee_from_joints 重写)

---

### Bug 4: 场景布局超出机械臂工作空间

**现象**：EE 初始位置在 `x=-0.107`，土堆在 `x=0.4`。距离 0.5m，而 Franka
在当前姿态的最大伸展约 0.7m，但默认关节角是缩回状态。

**根因**：默认关节角 `q1=-0.569, q3=-2.810` 使 FK 输出 `x < 0`（手臂朝后）。

**修复**：
1. 修改默认关节角为前伸姿态：`[0, 0.4, 0, -1.0, 0, 1.4, 0]`
2. 调整场景布局：
   - 土堆 `(0.4, 0, 0)` → `(0.5, 0, 0.05)`
   - 目标 `(-0.4, 0, 0)` → `(0.4, 0.3, 0.05)`（同侧偏横）

**文件改动**：`arm_cfg.py` (1处), `excavation_env_cfg.py` (2处)

---

### Bug 5: Reset 后首帧 EE 速度不为零

**现象**：`test_deterministic_reset` 和 `test_reproducible_trajectory` 失败。

**根因**：`_update_ee_from_joints()` 计算 `vel = (pos - prev_pos) / dt`，
reset 时 `prev_pos` 是上个 episode 的残留值。

**修复**：reset 中在调用 FK 前后都将 `_ee_lin_vel` 清零。

**文件改动**：`excavation_env.py` (3行)

---

### Bug 6: 测试参数名不匹配

**现象**：`test_height_map_shape` 报 `got an unexpected keyword argument 'resolution'`。

**根因**：`SoilTerrain.compute_height_map()` 的参数名是 `grid_resolution`，
测试中写的是 `resolution`。

**修复**：测试中改为 `grid_resolution=5`。

**文件改动**：`tests/test_soil.py` (2处)

---

## 8. 数据流完整走一遍

以一个完整 episode 为例，追踪数据在各模块间的流动：

### 8.1 Reset 阶段

```
train.py: env.reset(seed=42)
    │
    ▼
excavation_env.py: reset()
    ├── curriculum.update(total_steps)  → 判断当前 stage
    ├── event_manager.sample()          → 采样域随机化参数
    ├── joint_pos = default_joint_pos   → [0, 0.4, 0, -1.0, 0, 1.4, 0]
    ├── _update_ee_from_joints()        → ee_pos = [0.608, 0, 0.239]
    ├── scene.reset(rng)                → 生成粒子/重置刚体
    └── _get_observation()
            ├── RobotState(joint_pos, joint_vel, ee_pos, ...)
            ├── BucketState(load=0, force=[0,0,0])
            ├── SoilObservation(height_map=5×5)
            └── build_observation() → 66D numpy array
```

### 8.2 Step 阶段

```
train.py: actor_critic.get_action(obs) → action (7D, 高斯采样)
    │
    ▼
excavation_env.py: step(action)
    ├── action_noise (如果 DR 启用)
    ├── action × 0.1 → joint_delta
    ├── joint_pos += delta, clip to limits
    ├── FK → ee_pos
    │
    ├── scene.step_physics(ee_pos, ee_vel)
    │       └── 刚体模式: 弹簧接触力 + 重力 + 积分
    │       └── 粒子模式: Warp kernel 更新
    │
    ├── soil_ratio = scene.get_soil_in_target_ratio()
    ├── centroid = scene.get_soil_centroid()
    ├── load = scene.get_bucket_load(ee_pos)
    │
    ├── compute_reward(RewardState{...}) → RewardComponents
    │       R1: transfer = 10.0 × Δsoil_ratio
    │       R2: approach = 1.0 × exp(-5 × dist(ee, soil))
    │       R3: load = 2.0 × normalized_load
    │       R4: transport = 3.0 × load × exp(-5 × dist(ee, target))
    │       R5: smooth = -0.05 × ‖a_t - a_{t-1}‖²
    │       R6: limit = -1.0 × joint_violation
    │       R7: time = -0.01
    │       R8: collision = -5.0 × (ee_z < 0.01)
    │       total = sum(R1..R8)
    │
    ├── check_termination(TerminationState{...})
    │       → SUCCESS / TIMEOUT / JOINT_LIMIT / GROUND_PENETRATION
    │
    └── build_observation() → 66D numpy array
            │
            ▼
        return (obs, reward=total, terminated, truncated, info)
```

### 8.3 PPO 更新阶段

```
PPOTrainer.train():
    for iteration in range(max_iterations):
        # 1. 收集 rollout
        for step in range(24):                    # 24 步/env/update
            action = actor_critic.get_action(obs)  # 采样
            obs, rew, done, _, info = env.step(action)
            buffer.add(obs, action, rew, done, value, log_prob)

        # 2. 计算 advantage
        buffer.compute_returns_and_advantages(last_values, γ=0.99, λ=0.95)

        # 3. PPO 更新（5 epochs × 4 mini-batches）
        for epoch in range(5):
            for batch in buffer.get_batches(4):
                log_prob, value, entropy = evaluate_actions(batch.obs, batch.actions)
                ratio = exp(log_prob - old_log_prob)
                policy_loss = -min(ratio × adv, clip(ratio) × adv)
                value_loss = max(clipped, unclipped MSE)
                loss = policy_loss + value_loss - entropy_bonus
                loss.backward()
                clip_grad_norm_(1.0)
                optimizer.step()

        # 4. 自适应 lr
        if kl > 0.02: lr /= 1.5
        if kl < 0.005: lr *= 1.5
```

---

## 9. 关键设计决策与权衡

### 9.1 为什么用关节位置控制而非力矩控制？

```
关节位置控制:  action → delta_q → PD 控制器跟踪
关节力矩控制:  action → τ → 直接施加力矩

选位置控制的原因：
✓ 更稳定（PD 控制器兜底）
✓ 更容易 sim-to-real（实际机器人也常用位置控制）
✓ 训练更快收敛
✗ 不够灵活（不能做力控任务）
```

### 9.2 为什么用高度图而非全粒子位置？

```
全粒子:   (3 × 1000) = 3000D  → 维度爆炸，训练困难
质心+方差: (3 + 3 + 1) = 7D   → 太简化，丢失空间信息
高度图:    (5 × 5) = 25D      → 紧凑，保留空间结构 ← 推荐
PointNet:  encoder → 64D      → 灵活但增加网络复杂度（Stage 3 可尝试）
```

### 9.3 为什么 PPO 而非 SAC？

```
PPO:  on-policy, 适合大规模并行环境 (4096 envs)
SAC:  off-policy, replay buffer 占用大量 GPU 显存

Isaac Lab 的 GPU 并行设计天然适合 on-policy 算法。
RSL 的所有项目 (ANYmal locomotion 等) 都用 PPO。
```

### 9.4 Standalone 模式 vs Isaac Lab 模式

```
Standalone (当前):
  - 简化 FK (非精确)
  - CPU 上 32 个 env
  - 用于验证代码管道
  - 训练效果有限

Isaac Lab (目标):
  - GPU PhysX 精确碰撞
  - 4096 个并行 env
  - Warp 粒子实时仿真
  - 训练出真正可用的策略
```

---

## 10. 下一步 TODO

### 在 ETH Euler 集群上跑的步骤

```bash
# 1. 安装 Isaac Sim + Isaac Lab
# (按官方文档: https://isaac-sim.github.io/IsaacLab/main/)

# 2. 安装项目
cd excavation-rl && pip install -e .

# 3. Stage 0 验证 (GPU, 4096 envs)
python -m training.train --stage 0 --num-envs 4096 --max-iterations 500

# 4. Stage 1: 引入 Warp 粒子
python -m training.train --stage 1 --max-iterations 1000

# 5. Stage 2: 完整训练 + 课程学习
python -m training.train --stage 2 --max-iterations 2000

# 6. 评估
python -m evaluation.evaluate --checkpoint results/checkpoints/best_model.pt

# 7. 消融实验
python -m evaluation.ablation --experiment all --seeds 3
```

### 需要进一步开发的部分

- [ ] 将 `ExcavationEnv` 注册为 Isaac Lab 的 `ManagerBasedRLEnv`
- [ ] 替换简化 FK 为 Isaac Lab 的精确运动学
- [ ] 集成 Warp 粒子系统到 Isaac Lab 场景
- [ ] 接入 `rsl_rl` 官方训练管道（替换自定义 PPO）
- [ ] 加入 USD 铲斗资产
- [ ] 实现 TensorBoard 实时 reward 分量监控
