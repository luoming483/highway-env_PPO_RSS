# PPO + RSS 双层次安全框架 — 文件结构与数据流图

```mermaid
flowchart TB
    subgraph entry["🚀 入口层 Entry"]
        exp["experiment.py<br/>实验编排器"]
    end

    subgraph config_layer["⚙️ 配置层 Config"]
        cfg["config.py<br/>统一配置中心"]
    end

    subgraph core["🧠 核心训练层 Core Training"]
        train["train.py<br/>训练引擎"]
        rss["rss_safety.py<br/>RSS 安全包装器"]
        metrics["metrics.py<br/>MetricsCollector<br/>+ TrainingMetrics"]
    end

    subgraph eval_layer["📊 评估层 Evaluation"]
        evaluate["evaluate.py<br/>独立模型评估"]
    end

    subgraph vis["📈 可视化层 Visualization"]
        plot["plotting.py<br/>论文级图表生成"]
    end

    subgraph output["📁 输出层 Output"]
        direction LR
        models["results/models/<br/>训练模型 .zip"]
        plots["results/plots/<br/>*.png 图表"]
        data["results/data/<br/>results.json"]
        logs["results/logs/<br/>TensorBoard 日志"]
    end

    %% ===== 主流程 =====
    entry -->|"import & 读取实验定义"| cfg
    entry -->|"for each experiment × seed"| train
    entry -->|"save_results()"| data
    entry -->|"generate_plots()"| plot

    %% ===== train.py 内部 =====
    train -->|"导入 PPO_PARAMS, ENV_CONFIG, CURRICULUM_PHASES"| cfg
    train -->|"RSSConfig + RSSSafetyWrapper"| rss
    train -->|"MetricsCollector 回调"| metrics
    train -->|"save model"| models

    %% ===== RSS 安全层 =====
    rss -->|"RSSConfig 参数"| cfg

    %% ===== Metrics 数据流 =====
    metrics -->|"reward/loss/collision curves"| train
    metrics -->|"定期 eval + 最终 eval"| train

    %% ===== 评估链路 =====
    evaluate -->|"加载 PPO + RSS 配置"| cfg
    evaluate -->|"RSSSafetyWrapper"| rss
    evaluate -->|"加载模型"| models

    %% ===== 可视化链路 =====
    plot -->|"读取聚合数据"| data
    plot -->|"实验配置（颜色/线型）"| cfg
    plot -->|"6 张图表"| plots

    %% ===== 样式 =====
    style entry fill:#1f77b4,color:#fff,stroke:#0d3b66
    style config_layer fill:#ff7f0e,color:#fff,stroke:#b85d00
    style core fill:#2ca02c,color:#fff,stroke:#1b5e1b
    style eval_layer fill:#d62728,color:#fff,stroke:#8b0000
    style vis fill:#9467bd,color:#fff,stroke:#5b2c8e
    style output fill:#7f7f7f,color:#fff,stroke:#4d4d4d
```

---

## 模块依赖关系图

```mermaid
graph LR
    subgraph 依赖链
        A[experiment.py] -->|调用| B[train.py]
        A -->|调用| C[plotting.py]
        B -->|导入| D[rss_safety.py]
        B -->|导入| E[metrics.py]
        A -->|读取| F[config.py]
        B -->|读取| F
        C -->|读取| F
        G[evaluate.py] -->|读取| F
        G -->|导入| D
        G -->|加载模型| H[results/models/]
    end
```

---

## train.py 训练循环详解

```mermaid
flowchart TD
    subgraph train_loop["run_training() 训练循环"]
        s0["set_seed()<br/>设置随机种子(random,numpy,torch)"] --> s1

        s1["构建 phase_plan<br/>使用课程 or 单阶段"] --> s2

        s2["创建 MetricsCollector<br/>eval_env_builder → target_env"] --> s3

        s3["Phase 循环开始"] --> s4

        s4["_train_phase()"] --> s4a

        subgraph phase_detail["单阶段训练"]
            s4a["_build_vec_env()<br/>make_vec_env + FlattenObservation + RSSSafetyWrapper"] --> s4b
            s4b["新建 PPO 或 model.set_env()<br/>PPO(MlpPolicy, policy_kwargs, tensorboard_log)"] --> s4c
            s4c["model.learn()<br/>total_timesteps=phase_steps<br/>callback=MetricsCollector"] --> s4d
            s4d["train_env.close()"]
        end

        s4 --> s5{"还有下一阶段?"}
        s5 -->|Yes| s4
        s5 -->|No| s6

        s6["run_final_evaluation()<br/>20+ episodes 全面评估"] --> s7
        s7["model.save() → results/models/"] --> s8
        s8["返回 metrics.to_dict()"]
    end
```

---

## RSS 安全包装器 — 动作级安全盾

```mermaid
flowchart TD
    subgraph rss_step["RSSSafetyWrapper.step(action)"]
        a0["原始 action ∈ {LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER}"] --> a1

        a1["_assess_action_risk(action)"]

        subgraph risk_check["风险评估"]
            a1a["_candidate_lane() 计算目标车道"] --> a1b
            a1b["_front_rear_on_lane() 获取前后车"] --> a1c
            a1c["_front_gap_and_ttc() / _rear_gap_and_ttc()<br/>计算 gap, TTC"] --> a1d
            a1d["_safe_distance_front() / _safe_distance_rear()<br/>计算 RSS 安全距离"] --> a1e
            a1e{"危险?"}
            a1e -->|"gap < safe_dist<br/>OR ttc < threshold"| a1f["标记 unsafe, 记录 reason"]
            a1e -->|"安全"| a1g["标记 safe"]
        end

        a1 --> a2

        a2{"enable_shield AND unsafe?"} -->|"Yes → 干预"| a3
        a2 -->|"No → 放行"| a4

        a3["_choose_safe_action()<br/>多数情况→SLOWER<br/>已SLOWER→IDLE"] --> a5
        a4["final_action = original_action"] --> a5

        a5["env.step(final_action)"] --> a6

        a6{"intervened?"} -->|Yes| a7["reward += intervention_penalty (-2.5)"]
        a6 -->|No| a8["保持原 reward"]

        a7 --> a9
        a8 --> a9

        a9["注入 info 字段:<br/>rss_intervened, rss_original_action,<br/>rss_final_action, rss_reason,<br/>rss_min_ttc, rss_min_distance,<br/>rss_front_gap, rss_front_ttc,<br/>rss_safe_front_distance 等"]
    end
```

---

## MetricsCollector 回调 — 数据采集

```mermaid
flowchart TD
    subgraph callback["MetricsCollector (BaseCallback)"]
        direction TB

        s1["_on_training_start()<br/>初始化 running_returns, eval_env"] --> s2

        s2["每个 step: _on_step()"] --> s2a["累积 rewards → running_returns"]

        s2a --> s2b{"done?"}
        s2b -->|"是"| s2c["记录 episode reward<br/>记录 collision (crashed)<br/>计算 moving average<br/>写入 reward/collision curves"]
        s2b -->|"否"| s2d["继续"]

        s2c --> s2e{"到 eval 频率?<br/>(每 6000 steps)"}
        s2e -->|"是"| s2f["_run_periodic_eval()<br/>6 episodes, deterministic<br/>记录: reward_mean/std, collision_rate,<br/>intervention_rate, min_ttc, min_distance"]
        s2e -->|"否"| s2d

        s3["_on_rollout_start()"] --> s3a["_capture_loss()<br/>从 logger 读取 train/loss"]

        s4["_on_training_end()"] --> s4a["最终 eval + wall_time"]

        s5["run_final_evaluation()（外部调用）"] --> s5a["20+ episodes 全面评估<br/>→ final_reward_mean/std<br/>→ final_collision_rate<br/>→ final_intervention_rate<br/>→ final_min_ttc/min_distance"]
    end
```

---

## 实验定义 → 输出全流程

```mermaid
flowchart LR
    subgraph experiments["EXPERIMENTS 定义 (config.py)"]
        e1["baseline<br/>PPO only"]
        e2["our_method<br/>PPO + RSS + Curriculum"]
        e3["ablation_no_curriculum<br/>PPO + RSS"]
        e4["ablation_no_rss<br/>PPO + Curriculum"]
    end

    subgraph run["experiment.py 主流程"]
        r1["parse_args()"]
        r2["run_experiments()"]
        r3["save_results()"]
        r4["generate_plots()"]
    end

    subgraph plots["6 张输出图表"]
        p1["01_reward_comparison.png<br/>奖励收敛曲线"]
        p2["02_collision_comparison.png<br/>碰撞率对比"]
        p3["03_loss_comparison.png<br/>损失曲线对比"]
        p4["04_safety_metrics.png<br/>2x2 安全指标面板"]
        p5["05_final_performance.png<br/>最终性能柱状图"]
        p6["06_training_reward.png<br/>训练奖励曲线"]
    end

    e1 & e2 & e3 & e4 -->|"3 seeds × 30k steps"| r2
    r2 -->|"{exp: {seed: metrics}}"| r3
    r3 -->|"results/data/results.json"| r4
    r4 --> p1 & p2 & p3 & p4 & p5 & p6

    style experiments fill:#ff7f0e,color:#fff
    style run fill:#1f77b4,color:#fff
    style plots fill:#2ca02c,color:#fff
```

---

## 文件总览

| 文件 | 职责 | 行数 |
|------|------|------|
| `config.py` | 统一配置中心：env、PPO参数、RSS参数、实验定义、种子 | 129 |
| `train.py` | 训练引擎：run_training()，课程学习 + RSS 包装 | 217 |
| `rss_safety.py` | RSS 安全包装器：动作级安全盾，TTC/安全距离计算 | 217 |
| `metrics.py` | MetricsCollector 回调 + TrainingMetrics 数据类 | 277 |
| `plotting.py` | 论文级 matplotlib 图表：6 张对比图 | 377 |
| `experiment.py` | 实验编排器：遍历实验×种子，保存结果，生成图表 | 183 |
| `evaluate.py` | 独立模型评估：加载模型 + 可选 RSS 进行评估 | 120 |

---

## 使用说明

将此 Mermaid 代码复制到支持 Mermaid 渲染的工具中即可查看流程图：

- **GitHub**: 直接粘贴到 `.md` 文件中，GitHub 原生支持 Mermaid 渲染
- **VS Code**: 安装 "Markdown Preview Mermaid Support" 插件
- **在线工具**: https://mermaid.live/
