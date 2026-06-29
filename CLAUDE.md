# PPO + RSS Dual-Layer Safety Framework — Team Agents

## Project Overview

PPO reinforcement learning + RSS safety shield for autonomous driving on `highway-env`. The dual-layer architecture: PPO proposes actions, RSS validates and overrides unsafe ones. Supports controlled experiments with paper-quality plotting.

**Key files:**
- `config.py` — centralized config: env, PPO params, RSS params, experiment definitions, seeds
- `train.py` — single `run_training()` function, returns structured metrics dict
- `evaluate.py` — standalone model evaluation with RSS support
- `rss_safety.py` — RSS safety wrapper (action-level shield, unchanged core logic)
- `metrics.py` — `MetricsCollector` callback + `TrainingMetrics` dataclass
- `plotting.py` — paper-quality matplotlib plots (reward, collision, loss, safety, bar charts)
- `experiment.py` — experiment orchestration: run all experiments × seeds, save JSON, generate plots
- `README.md` — comprehensive beginner-friendly documentation
- `stackelberg/` — Stackelberg game + FSM expert module (研究内容一)

**Output structure:** `results/models/`, `results/plots/`, `results/data/`

**Env:** `highway-fast-v0` with DiscreteMetaAction (5 actions: LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER)

---

## Three-Expert Architecture (Thesis Plan)

The complete thesis targets a Mixture-of-Experts (MoE) framework with three decision modules:

| Expert | Module | Status | Description |
|--------|--------|--------|-------------|
| Game Expert | `stackelberg/` | Implemented | Stackelberg leader-follower game + FSM governance |
| RL Expert | `train.py` + PPO | Implemented | PPO-trained policy (currently discrete, needs continuous action upgrade) |
| RSS Shield | `rss_safety.py` | Implemented | RSS safety envelope for action-level override |
| MoE Gating | (future) | Not started | Scene-adaptive gating network fusing experts |

### stackelberg/ module

Stackelberg game based on trajectory prediction for lane change in mixed traffic.
Reference: Shi B, Zhai L, Liu C. IEEE Access.

```
stackelberg/
    __init__.py              # Package exports
    config.py                # GameConfig + driving style weight table (paper Table 1)
    trajectory_predictor.py  # Linear decay acceleration model (paper eq.4-2~4-4)
    utility_functions.py     # HV utility (eq.7-11) + EV cost (eq.13-18)
    game_solver.py           # Stackelberg equilibrium solver (simplified Algorithm 1)
    fsm_executor.py          # 4-state FSM + safety gating + rate limiting (tech roadmap 2.1.2)
    stackelberg_expert.py    # Top-level expert: Game → FSM → Action
    test_expert.py           # Smoke tests
```

**Decision pipeline:** Perception → Stackelberg Game Solver → FSM Governance → Action

**Test:** `D:\anaconda\envs\ppo_main\python.exe -m stackelberg.test_expert`

---

## Experiment Design

| Experiment | PPO | RSS | Curriculum | Purpose |
|------------|-----|-----|-----------|---------|
| baseline | Yes | No | No | Pure PPO baseline |
| our_method | Yes | Yes | Yes | Our complete method |
| ablation_no_curriculum | Yes | Yes | No | Ablation: remove curriculum |
| ablation_no_rss | Yes | No | Yes | Ablation: remove RSS |

Experiments defined in `config.py:EXPERIMENTS`. 3 seeds each (42, 123, 456). 30k timesteps total.

---

## Agents

### rl-researcher
Reinforcement Learning researcher specialized in PPO, curriculum learning, reward shaping, and hyperparameter tuning.

**When to use:** designing reward functions, tuning PPO hyperparameters, analyzing convergence, designing curriculum phases, interpreting experiment results.

**Tools:** Read, Grep, Glob, Bash, WebSearch, WebFetch

**Instructions:**
- You are an RL researcher focused on PPO and autonomous driving safety.
- Check `results/plots/` for training curves and `results/data/results.json` for raw metrics.
- Reference PPO hyperparams in `config.py` (PPO_PARAMS, POLICY_NET_ARCH, CURRICULUM_PHASES).
- RSS params are in `config.py:RSS_CONFIG` — understand RSS theory before suggesting parameter changes.
- The RSS safety wrapper (`rss_safety.py`) is the safety layer — consider its impact on exploration trade-off.
- When analyzing experiment results, compare across the 4 experiment groups defined in EXPERIMENTS.
- Always read `config.py` first to understand current settings before making suggestions.
- Explain RL theory behind each recommendation.

---

### code-reviewer
Code reviewer for this PPO+RSS project. Reviews for correctness, best practices, and potential bugs.

**When to use:** reviewing code changes, checking callback logic, verifying wrapper correctness, spotting numerical issues.

**Tools:** Read, Grep, Glob, Bash

**Instructions:**
- Review for: numerical stability (NaN/Inf handling), correct SB3 API usage, proper env lifecycle (close/seed), callback correctness.
- Key patterns to verify: gymnasium API compliance (obs, info tuple returns), proper use of `unwrapped` in wrappers, callback `_on_step` return values.
- The `rss_safety.py` wrapper must correctly handle all 5 discrete actions and properly compute RSS distances.
- In `metrics.py`, check that episode boundary detection is correct and eval metrics properly aggregate.
- In `train.py`, verify seed setting covers random, numpy, and torch.
- After reviewing: concise summary — what looks good, what's risky, concrete fix suggestions.

---

### rl-debugger
Debugging specialist for RL training issues. Diagnoses convergence problems, high collision rates, unexpected behavior.

**When to use:** training not converging, collision rate too high, reward collapse, NaN losses, slow training, RSS not intervening.

**Tools:** Read, Grep, Glob, Bash, WebSearch

**Instructions:**
- First check `results/plots/` for training curves, then `results/data/results.json` for raw metrics.
- Common issues to diagnose:
  - Reward collapse → check `ent_coef` is not too low, `learning_rate` not too high.
  - High collision rate → check RSS config (ttc_threshold, min_distance), collision_reward magnitude.
  - NaN loss → check `max_grad_norm` is set, verify gradient stability.
  - RSS not intervening → check `enable_shield=True`, verify intervention_rate > 0 in metrics.
  - Slow training → check `n_steps`, `batch_size`, number of envs, device.
- Compare config.py hyperparams against known good ranges for highway-env.
- Use `WebSearch` to look up similar issues with stable-baselines3 PPO on highway-env.
- End with: root cause hypothesis, evidence, concrete fix.

---

### rl-architect
Architecture and design agent. Plans refactoring, new features, and structural improvements.

**When to use:** adding new metrics, refactoring modules, adding new safety mechanisms, restructuring config, planning new experiment types.

**Tools:** Read, Grep, Glob, Bash

**Instructions:**
- Current architecture: config.py is single source of truth → train.py orchestrates training → metrics.py collects data → plotting.py visualizes → experiment.py ties them together.
- When designing new features:
  - Keep config.py as the central config hub.
  - New callbacks should extend `MetricsCollector` pattern in `metrics.py`.
  - New wrappers follow `RSSSafetyWrapper` pattern in `rss_safety.py`.
  - New plot types go in `plotting.py` and are called from `experiment.py:generate_plots()`.
  - New experiments go in `config.py:EXPERIMENTS` dict.
- Consider: reproducibility (seed management), Windows compatibility, clean separation of concerns.
- Propose changes as: which files, what changes, migration steps.

---

## Commands

### Run all experiments
```bash
D:\anaconda\envs\ppo_main\python.exe experiment.py
```

### Quick test
```bash
D:\anaconda\envs\ppo_main\python.exe experiment.py --experiments baseline --seeds 42 --timesteps 5000
```

### Evaluate a model
```bash
D:\anaconda\envs\ppo_main\python.exe evaluate.py --model-path results/models/our_method_seed42/final_model.zip --rss
```

### TensorBoard
```bash
tensorboard --logdir results/logs
```
