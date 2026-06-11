# PPO + RSS 安全强化学习实验

本项目基于 Stable-Baselines3 的 PPO 算法，在 `highway-env` 高速公路场景中训练自动驾驶决策智能体，并加入 RSS 安全屏蔽层，用于比较纯 PPO 与 PPO+RSS 方法在奖励、碰撞率和安全指标上的表现。

当前代码以 `config.py` 中的配置为准，主要运行 3 组实验：

| 实验名 | 方法 | RSS | 说明 |
| --- | --- | --- | --- |
| `baseline` | Pure PPO | 否 | 纯 PPO 基线 |
| `our_method` | PPO + RSS | 是 | RSS 干预惩罚为 `-1.0` |
| `ablation_rss_harsh` | PPO + RSS | 是 | RSS 干预惩罚为 `-2.5`，用于对比更强惩罚 |

## 项目结构

```text
PPO_SB3/
├─ config.py              # 全局配置：环境参数、PPO超参数、RSS参数、实验组、输出路径
├─ train.py               # 单次训练入口：创建环境、训练PPO、保存模型、返回指标
├─ rss_safety.py          # RSS安全包裹器：动作风险评估、安全动作替换、RSS指标记录
├─ metrics.py             # 训练和评估指标采集：奖励、碰撞率、loss、TTC、干预率等
├─ evaluate.py            # 加载已训练模型并单独评估
├─ experiment.py          # 批量实验入口：多实验组、多随机种子、保存结果、生成图表
├─ plotting.py            # 根据 results.json 生成对比图
├─ architecture_flowchart.md
├─ results/               # 当前实验输出目录
└─ results_v1_30k/        # 历史实验结果
```

## 运行环境

建议使用 Python 3.10 或相近版本。

安装依赖：

```bash
pip install stable-baselines3 gymnasium highway-env numpy matplotlib torch
```

如果需要查看 TensorBoard 日志，可额外安装：

```bash
pip install tensorboard
```

## 快速开始

运行默认完整实验：

```bash
python experiment.py
```

默认配置来自 `config.py`：

- 环境：`highway-fast-v0`
- 训练步数：`50_000`
- 并行环境数：`4`
- 随机种子：`42, 123, 456, 789, 1011`
- 实验组：`baseline, our_method, ablation_rss_harsh`

快速测试单个实验：

```bash
python experiment.py --experiments baseline --seeds 42 --timesteps 5000
```

只训练并保存数据，不生成图表：

```bash
python experiment.py --skip-plots
```

指定设备：

```bash
python experiment.py --device cpu
python experiment.py --device cuda
python experiment.py --device auto
```

## 单独评估模型

评估纯 PPO 模型：

```bash
python evaluate.py --model-path results/models/baseline_seed42/final_model.zip
```

评估时启用 RSS 安全屏蔽：

```bash
python evaluate.py --model-path results/models/our_method_seed42/final_model.zip --rss
```

指定评估回合数：

```bash
python evaluate.py --model-path results/models/our_method_seed42/final_model.zip --rss --episodes 20
```

## 核心流程

```text
experiment.py
  └─ run_experiments()
      └─ train.py / run_training()
          ├─ 读取 config.py
          ├─ 创建 highway-fast-v0 环境
          ├─ 可选套用 RSSSafetyWrapper
          ├─ 使用 FlattenObservation 展平观测
          ├─ 创建 PPO(MlpPolicy)
          ├─ 通过 MetricsCollector 采集指标
          ├─ 保存模型到 results/models/
          └─ 返回训练和评估指标

experiment.py
  ├─ save_results()       -> results/data/results.json
  └─ generate_plots()     -> results/plots/*.png
```

## RSS 安全层

`rss_safety.py` 中的 `RSSSafetyWrapper` 是一个 Gym Wrapper，位于智能体和环境之间。PPO 输出动作后，RSS 层先判断该动作是否安全，再决定是否放行或替换动作。

动作编号如下：

| 编号 | 动作 |
| --- | --- |
| `0` | `LANE_LEFT` |
| `1` | `IDLE` |
| `2` | `LANE_RIGHT` |
| `3` | `FASTER` |
| `4` | `SLOWER` |

RSS 层主要检查：

- 目标车道是否存在
- 前车距离和前向 TTC
- 后车距离和后向 TTC
- RSS 安全距离
- 加速、保持、变道、减速动作是否存在风险

如果动作被判定为危险，并且 `enable_shield=True`，RSS 会替换为更保守的动作，例如 `IDLE` 或 `SLOWER`。

每一步环境交互都会在 `info` 中写入 RSS 相关信息：

```text
rss_enabled
rss_intervened
rss_original_action
rss_final_action
rss_reason
rss_penalty
rss_min_ttc
rss_min_distance
rss_front_gap
rss_front_ttc
rss_safe_front_distance
rss_safe_rear_distance
```

## 指标说明

训练和评估过程中会记录以下指标：

| 指标 | 含义 |
| --- | --- |
| `reward_curve_y` | 训练过程中的滑动平均回合奖励 |
| `loss_curve_y` | PPO 训练 loss |
| `collision_curve_y` | 训练过程中的滑动碰撞率 |
| `eval_reward_mean` | 定期评估平均奖励 |
| `eval_collision_rate` | 定期评估碰撞率 |
| `eval_intervention_rate` | RSS 干预步数比例 |
| `eval_min_ttc` | 评估过程中的最小 TTC 均值 |
| `eval_min_distance` | 评估过程中的最小车距均值 |
| `final_reward_mean` | 最终评估平均奖励 |
| `final_collision_rate` | 最终评估碰撞率 |
| `final_intervention_rate` | 最终评估 RSS 干预率 |
| `wall_time_seconds` | 单次训练耗时 |

## 输出文件

当前实验输出到 `results/`：

```text
results/
├─ models/
│  └─ <experiment>_seed<seed>/final_model.zip
├─ logs/
│  ├─ PPO_0/
│  └─ monitor_seed*/
├─ data/
│  └─ results.json
└─ plots/
   ├─ 01_reward_comparison.png
   ├─ 02_collision_comparison.png
   ├─ 03_loss_comparison.png
   ├─ 04_safety_metrics.png
   ├─ 05_final_performance.png
   └─ 06_training_reward.png
```

`results_v1_30k/` 是历史实验结果目录，当前训练默认不会写入该目录。

## 修改实验配置

所有主要配置都在 `config.py`。

修改训练步数：

```python
TOTAL_TIMESTEPS = 50_000
```

修改随机种子：

```python
SEEDS = [42, 123, 456, 789, 1011]
```

修改 PPO 超参数：

```python
PPO_PARAMS = {
    "learning_rate": 3.0e-4,
    "n_steps": 256,
    "batch_size": 256,
    "n_epochs": 6,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.005,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": 0.02,
}
```

修改 RSS 参数：

```python
RSS_CONFIG = {
    "response_time": 0.8,
    "rear_response_time": 0.6,
    "min_distance": 5.0,
    "max_brake": 6.0,
    "ttc_threshold": 2.0,
    "intervention_penalty": -1.0,
    "enable_shield": True,
}
```

添加新实验组：

```python
EXPERIMENTS["my_experiment"] = {
    "name": "my_experiment",
    "label": "My PPO + RSS",
    "use_rss": True,
    "use_curriculum": False,
    "rss_overrides": {"intervention_penalty": -0.5},
    "color": "#9467bd",
    "linestyle": "-.",
    "marker": "d",
}
```

然后运行：

```bash
python experiment.py --experiments my_experiment
```

## 查看 TensorBoard

训练日志保存在 `results/logs/`，可以使用：

```bash
tensorboard --logdir results/logs
```

## 注意事项

- 当前 `config.py` 中虽然保留了 `CURRICULUM_PHASES`，但默认启用的实验组都设置了 `use_curriculum=False`。
- 如果要重新启用课程学习，需要在对应实验配置中设置 `use_curriculum=True`。
- `README.md` 描述的是当前代码版本；历史目录 `results_v1_30k/` 中的实验组可能与当前配置不同。
