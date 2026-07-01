# MoE Highway — 融合博弈论与强化学习的高速公路自动驾驶安全换道决策

面向 `highway-env` 的混合专家（Mixture-of-Experts）自动驾驶决策框架。两个专家模块——Stackelberg 博弈和 PPO+RSS 强化学习——提供互补决策能力，通过场景门控网络自适应融合，支持对照实验与论文级可视化。

## 研究内容

本课题针对高速公路混合交通流，开展融合博弈论与强化学习的自动驾驶安全换道决策方法研究，包含三个递进的研究内容：

| 研究内容 | 模块 | 方法 | 解决什么问题 |
| --- | --- | --- | --- |
| **研究内容一** | `stackelberg/` | 主从博弈 + 有限状态机 | 交互式换道决策，抑制高频指令跳变 |
| **研究内容二** | `ppo/` + `rss/` | PPO 强化学习 + RSS 安全屏蔽 | 数据驱动速度优化，底线安全兜底 |
| **研究内容三** | `moe_hybrid.py` | 混合专家自适应门控 | 全工况场景适应性，动态仲裁与平滑融合 |

## 双专家架构

```
                      ┌─────────────────────┐
                      │   场景感知 / 门控    │
                      │   MoE Gate          │
                      └──────┬──────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
   ┌──────────────────┐          ┌──────────────────┐
   │ Expert 1         │          │ Expert 2         │
   │ Stackelberg+FSM  │          │ PPO+RSS          │
   │ 博弈论换道决策    │          │ 强化学习+安全屏蔽 │
   └──────────────────┘          └──────────────────┘
         │                                │
         │  博弈求解 → 状态机治理          │  PPO推理 → RSS校验
         │  换道意图（左/右/保持）          │  速度控制（加速/减速）
         │                                │
         └──────────────┬────────────────┘
                        ▼
                 ┌──────────────┐
                 │  最终动作     │
                 │  (5 个离散动作) │
                 └──────────────┘
```

- **Expert 1（Stackelberg+FSM）**：自车为领导者，邻车为跟随者，通过短时域轨迹预测和多目标效用函数求解 Stackelberg 均衡。FSM 四状态机（车道保持 → 换道准备 → 换道执行 → 状态恢复）提供状态锁定、冷却延时和安全门控，抑制决策抖动。
- **Expert 2（PPO+RSS）**：PPO 深度强化学习策略负责速度优化，RSS（责任敏感安全）模型作为安全屏蔽层，对危险动作实时拦截与修正。RSS 紧急制动（TTC < 3s）是该专家内部的安全机制。
- **MoE Gate**：规则门控，三层优先级——TTC < 3s 触发紧急安全 → 换道有益时选 Stackelberg → 其余默认 PPO+RSS 速度优化。

## 项目结构

```text
PPO_SB3/
├── config.py              # 中心配置：环境、PPO超参、RSS参数、实验组定义
├── train.py               # PPO 训练函数：建环境、课程学习、训练、保存模型
├── evaluate.py            # 模型评估：加载模型、多回合评估、输出指标
├── metrics.py             # 训练指标采集：MetricsCollector 回调 + TrainingMetrics
├── experiment.py          # 实验编排入口：多实验组×多种子、保存JSON、生成绑图
├── plotting.py            # 论文级 matplotlib 绑图（6 张对比图）
├── moe_hybrid.py          # 混合专家顶层整合：门控 + 三专家决策（研究内容三）
├── sweep_lc_reward.py     # 换道奖励超参扫描
│
├── ppo/                   # PPO 可视化模块
│   ├── __init__.py
│   └── visualize.py       # pygame/console 可视化已训练 PPO+RSS 策略
│
├── rss/                   # RSS 安全屏蔽模块
│   ├── __init__.py
│   └── shield.py          # RSSSafetyWrapper：动作级安全校验与替换
│
├── stackelberg/           # Stackelberg 博弈专家模块（研究内容一）
│   ├── __init__.py
│   ├── config.py          # GameConfig + 驾驶风格权重表
│   ├── trajectory_predictor.py  # 线性衰减加速度轨迹预测
│   ├── utility_functions.py     # HV 效用函数 + EV 代价函数
│   ├── game_solver.py           # Stackelberg 均衡求解器
│   ├── fsm_executor.py          # 4 状态 FSM + 安全门控 + 冷却延时
│   ├── expert.py                # 顶层专家：Game → FSM → Action
│   ├── visualize.py             # Stackelberg 决策可视化
│   ├── test_expert.py           # 冒烟测试
│   └── test_units.py            # 单元测试
│
├── tools/                 # 工具脚本集
│   ├── compare_experts.py      # 多专家对比（Stackelberg vs IDM vs Random vs PPO+RSS）
│   ├── ablation_threshold.py   # 阈值消融实验（FSM vs RSS 参数敏感性）
│   ├── plot_ablation.py        # 消融实验绑图
│   ├── plot_moe_results.py     # MoE 结果 SCI 级绑图
│   ├── diagnose_blocked.py     # Blocked 状态诊断
│   ├── test_lc_training.py     # 换道训练验证
│   ├── test_wrapper_debug.py   # ForceExplore wrapper 调试
│   ├── _debug_trace.py         # 单种子决策时间线调试
│   └── _test_collision.py      # 多种子碰撞测试
│
├── runs/                  # 实验输出（时间戳命名）
├── results/               # 历史实验结果
├── results_v1_30k/        # v1 版本实验结果
├── README.md
└── CLAUDE.md
```

## 运行环境

Python 3.10+，推荐使用 Anaconda 环境：

```bash
conda create -n ppo_main python=3.10
conda activate ppo_main
pip install stable-baselines3 gymnasium highway-env numpy matplotlib torch
pip install tensorboard  # 可选，用于查看训练日志
```

## 快速开始

### 运行完整实验

```bash
D:\anaconda\envs\ppo_main\python.exe experiment.py
```

默认配置（`config.py`）：
- 环境：`highway-fast-v0`，4 车道，20 辆车
- 训练步数：200,000（3 阶段课程学习）
- 4 个并行环境
- 5 个随机种子（42, 123, 456, 789, 1011）
- 4 组实验（baseline, our_method, ablation_no_curriculum, ablation_no_rss）

### 快速测试

```bash
D:\anaconda\envs\ppo_main\python.exe experiment.py --experiments our_method --seeds 42 --timesteps 5000
```

### 可视化已训练模型

```bash
# pygame 窗口（默认）
D:\anaconda\envs\ppo_main\python.exe ppo/visualize.py --seed 42 --vehicles 20

# 控制台诊断模式
D:\anaconda\envs\ppo_main\python.exe ppo/visualize.py --render console
```

### 运行 MoE 混合专家

```bash
D:\anaconda\envs\ppo_main\python.exe moe_hybrid.py --seed 42 --vehicles 20 --duration 30
```

### 运行 Stackelberg 专家测试

```bash
D:\anaconda\envs\ppo_main\python.exe -m stackelberg.test_expert
```

## 各模块说明

### ppo/ — PPO 强化学习 + 可视化

PPO 算法训练的驾驶策略，使用 DiscreteMetaAction（5 个离散动作：左转、保持、右转、加速、减速）。训练时可选 RSS 安全屏蔽 + 课程学习 + Blocked 惩罚 + 强制探索。

- `ppo/visualize.py` — 加载训练好的模型，通过 pygame 窗口或终端模式观察决策行为，实时显示速度、TTC、车距、RSS 干预状态。

### rss/ — RSS 安全屏蔽

RSS（Responsibility-Sensitive Safety）是 Mobileye 提出的数学安全模型。`RSSSafetyWrapper` 作为 Gym Wrapper，PPO 输出动作后，先判断是否安全：

- 检查目标车道是否存在、前后车 TTC 和车距是否满足 RSS 安全距离
- 若动作危险且 `enable_shield=True`，替换为保守动作（IDLE 或 SLOWER）
- 每步写入 `info`：`rss_intervened`, `rss_min_ttc`, `rss_min_distance`, `rss_reason` 等

详见 [RSS 安全层](#rss-安全层) 章节。

### stackelberg/ — Stackelberg 博弈专家

主从博弈换道决策，5 步决策流水线：

1. **轨迹预测** — 线性衰减加速度模型，预测邻车短时域轨迹
2. **博弈求解** — 自车为领导者，枚举 18 个候选策略；邻车为跟随者，最优响应
3. **FSM 执行** — 4 状态机（LANE_KEEPING → LC_PREPARATION → LC_EXECUTION → STATE_RECOVERY），带冷却延时和迟滞阈值
4. **安全门控** — TTC/间距校验，必要时否决博弈输出
5. **指令平滑** — EMA 滤波 + 变化率限制，消除高频抖动

测试：`D:\anaconda\envs\ppo_main\python.exe -m stackelberg.test_expert`

### tools/ — 工具脚本

| 脚本 | 用途 |
| --- | --- |
| `compare_experts.py` | 四种专家（Stackelberg / IDM / Random / PPO+RSS）多密度对比 |
| `ablation_threshold.py` | 阈值消融：FSM 放松 vs RSS 收紧，验证 MoE 优势来自架构还是参数 |
| `plot_ablation.py` | 消融实验 4 张对比图 |
| `plot_moe_results.py` | MoE 混合专家 SCI 论文级绑图 |
| `diagnose_blocked.py` | 诊断 Blocked 状态触发频率 |
| `test_lc_training.py` | 换道训练参数验证 |
| `test_wrapper_debug.py` | ForceExplore wrapper 行为调试 |
| `_debug_trace.py` | 单种子决策时间线逐帧追踪 |
| `_test_collision.py` | 多种子碰撞率批量测试 |

## 实验设计

| 实验组 | PPO | RSS | Curriculum | 说明 |
| --- | --- | --- | --- | --- |
| `baseline` | ✓ | ✗ | ✗ | 纯 PPO 基线 |
| `our_method` | ✓ | ✓ | ✓ | 完整方法（PPO+RSS+课程学习） |
| `ablation_no_curriculum` | ✓ | ✓ | ✗ | 消融：去掉课程学习 |
| `ablation_no_rss` | ✓ | ✗ | ✓ | 消融：去掉 RSS |
| `ppo_lc` | ✓ | 仅评估 | ✗ | PPO 换道专项（训练无RSS，评估有RSS） |

实验定义在 `config.py:EXPERIMENTS`，通过 `ACTIVE_EXPERIMENTS` 控制激活哪些实验组。

## 课程学习

3 阶段渐进式训练（`config.py:CURRICULUM_PHASES`）：

| 阶段 | 占比 | 车辆数 | 密度 | 持续时间 |
| --- | --- | --- | --- | --- |
| Phase 1: Light | 25% | 10 | 0.45 | 45s |
| Phase 2: Medium | 35% | 15 | 0.70 | 55s |
| Phase 3: Target | 40% | 20 | 1.0 | 60s |

## RSS 安全层

`RSSSafetyWrapper` 位于 PPO 策略和环境之间，对 5 个动作进行安全校验：

| 动作 | RSS 检查内容 |
| --- | --- |
| LEFT (0) / RIGHT (2) | 目标车道是否存在、前后车 RSS 安全距离、侧向间隙 |
| FASTER (3) | 前车纵向 RSS 安全距离 |
| IDLE (1) / SLOWER (4) | 通常放行（减速和保持总是安全的） |

每一步 `info` 中包含的 RSS 字段：

```text
rss_enabled, rss_intervened, rss_original_action, rss_final_action,
rss_reason, rss_penalty, rss_min_ttc, rss_min_distance,
rss_front_gap, rss_front_ttc, rss_safe_front_distance, rss_safe_rear_distance
```

## 训练指标

| 指标 | 含义 |
| --- | --- |
| `reward_curve_y` | 训练过程滑动平均回合奖励 |
| `loss_curve_y` | PPO 训练 loss |
| `collision_curve_y` | 训练过程滑动碰撞率 |
| `eval_reward_mean` | 定期评估平均奖励 |
| `eval_collision_rate` | 定期评估碰撞率 |
| `eval_intervention_rate` | RSS 干预步数比例 |
| `eval_min_ttc` | 评估过程最小 TTC 均值 |
| `eval_min_distance` | 评估过程最小车距均值 |
| `final_reward_mean` | 最终评估平均奖励 |
| `final_collision_rate` | 最终评估碰撞率 |
| `final_intervention_rate` | 最终评估 RSS 干预率 |
| `wall_time_seconds` | 单次训练耗时 |

## 输出文件

```text
runs/<timestamp>/
├── models/
│   └── <experiment>_seed<seed>/
│       └── final_model.zip
├── logs/
│   └── PPO_0/
├── data/
│   └── results.json
└── plots/
    ├── 01_reward_comparison.png      # 各实验组奖励对比
    ├── 02_collision_comparison.png   # 各实验组碰撞率对比
    ├── 03_loss_comparison.png        # 各实验组 Loss 对比
    ├── 04_safety_metrics.png         # 安全指标 2×2 面板
    ├── 05_final_performance.png      # 最终性能柱状图
    └── 06_training_reward.png        # 训练奖励曲线
```

## 修改配置

所有主要配置在 `config.py`：

```python
TOTAL_TIMESTEPS = 200_000        # 训练步数
SEEDS = [42, 123, 456, 789, 1011]  # 随机种子
N_ENVS = 4                       # 并行环境数
```

RSS 参数：

```python
RSS_CONFIG = {
    "ttc_threshold": 3.0,        # TTC 安全阈值
    "min_distance": 8.0,         # 最小安全距离
    "enable_shield": True,       # 是否启用安全屏蔽
    "intervention_penalty": -0.5, # 干预惩罚
}
```

添加新实验组：

```python
EXPERIMENTS["my_exp"] = {
    "name": "my_exp",
    "label": "My Experiment",
    "use_rss": True,
    "use_curriculum": True,
    "rss_overrides": {},
    "color": "#9467bd",
    "linestyle": "-.",
    "marker": "d",
}
ACTIVE_EXPERIMENTS = ["baseline", "our_method", "my_exp"]
```

## 查看 TensorBoard

```bash
tensorboard --logdir runs/<timestamp>/logs
```

## 注意事项

- 完整实验（4 组 × 5 种子 × 200k 步）耗时较长，建议先跑 smoke test：`--experiments our_method --seeds 42 --timesteps 5000`
- `results/` 和 `results_v1_30k/` 是历史输出目录，当前训练默认输出到 `runs/<timestamp>/`
- `ppo/visualize.py` 和 `moe_hybrid.py` 依赖已训练的 PPO 模型，默认路径为 `runs/20260615_163841/models/our_method_seed42/final_model.zip`
- 所有 tools/ 脚本从项目根目录运行，自动通过 `sys.path` 定位 `config.py` 和各模块
