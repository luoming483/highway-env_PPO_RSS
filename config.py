"""Central configuration for PPO + RSS dual-layer experiments."""

from pathlib import Path
from typing import Dict, List

# ---- Paths ----
PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
RUNS_DIR = PROJECT_DIR / "runs"
MODEL_DIR = RESULTS_DIR / "models"
PLOT_DIR = RESULTS_DIR / "plots"
DATA_DIR = RESULTS_DIR / "data"
LOG_DIR = RESULTS_DIR / "logs"

# ---- Environment ----
ENV_ID = "highway-fast-v0"

ENV_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 20,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "absolute": False,
        "order": "sorted",
    },
    "action": {"type": "DiscreteMetaAction"},
    "lanes_count": 4,
    "vehicles_count": 20,
    "vehicles_density": 1.0,
    "duration": 60,
    "simulation_frequency": 8,
    "policy_frequency": 4,
    "collision_reward": -10.0,
    "right_lane_reward": 0.02,
    "high_speed_reward": 0.70,
    "lane_change_reward": 0.12,
    "reward_speed_range": [20, 30],
    "normalize_reward": False,
    "offroad_terminal": True,
}

# ---- Curriculum Phases ----
CURRICULUM_PHASES = [
    {"name": "phase1_light", "ratio": 0.25, "overrides": {"vehicles_count": 10, "vehicles_density": 0.45, "duration": 45}},
    {"name": "phase2_medium", "ratio": 0.35, "overrides": {"vehicles_count": 15, "vehicles_density": 0.70, "duration": 55}},
    {"name": "phase3_target", "ratio": 0.40, "overrides": {}},
]

# ---- PPO Hyperparameters ----
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

POLICY_NET_ARCH = [128, 128]

# ---- Training Settings ----
TOTAL_TIMESTEPS = 200_000
N_ENVS = 4
EVAL_FREQ_STEPS = 20_000
N_EVAL_EPISODES = 10
REWARD_WINDOW = 10

# ---- RSS Parameters ----
RSS_CONFIG = {
    "response_time": 1.0,
    "rear_response_time": 0.8,
    "min_distance": 8.0,
    "max_brake": 6.0,
    "ttc_threshold": 3.0,
    "intervention_penalty": -0.5,
    "nearby_vehicle_horizon": 45.0,
    "lane_change_side_gap": 8.0,
    "enable_shield": True,
}

# ---- Action Labels ----
ACTION_LABELS = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}

# ---- Experiment Definitions ----
SEEDS: List[int] = [42, 123, 456, 789, 1011]

EXPERIMENTS: Dict[str, Dict] = {
    "baseline": {
        "name": "baseline",
        "label": "Pure PPO (Baseline)",
        "use_rss": False,
        "use_curriculum": False,
        "rss_overrides": {},
        "color": "#1f77b4",
        "linestyle": "-",
        "marker": "o",
    },
    "our_method": {
        "name": "our_method",
        "label": "PPO + RSS + Curriculum",
        "use_rss": True,
        "use_curriculum": True,
        "rss_overrides": {"intervention_penalty": -0.5},
        "color": "#2ca02c",
        "linestyle": "-",
        "marker": "s",
    },
    "ablation_no_curriculum": {
        "name": "ablation_no_curriculum",
        "label": "PPO + RSS (No Curriculum)",
        "use_rss": True,
        "use_curriculum": False,
        "rss_overrides": {"intervention_penalty": -0.5},
        "color": "#ff7f0e",
        "linestyle": "--",
        "marker": "^",
    },
    "ablation_no_rss": {
        "name": "ablation_no_rss",
        "label": "PPO + Curriculum (No RSS)",
        "use_rss": False,
        "use_curriculum": True,
        "rss_overrides": {},
        "color": "#d62728",
        "linestyle": ":",
        "marker": "D",
    },
}

ACTIVE_EXPERIMENTS = ["baseline", "our_method", "ablation_no_curriculum", "ablation_no_rss"]
