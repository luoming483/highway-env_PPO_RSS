"""Central configuration for PPO + RSS dual-layer experiments."""

from pathlib import Path
from typing import Dict, List

# ---- Paths ----
PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
MODEL_DIR = RESULTS_DIR / "models"
PLOT_DIR = RESULTS_DIR / "plots"
DATA_DIR = RESULTS_DIR / "data"
LOG_DIR = RESULTS_DIR / "logs"

# ---- Environment (reduced difficulty) ----
ENV_ID = "highway-fast-v0"

ENV_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "absolute": False,
        "order": "sorted",
    },
    "action": {"type": "DiscreteMetaAction"},
    "lanes_count": 4,
    "vehicles_count": 25,
    "vehicles_density": 1.2,
    "duration": 70,
    "simulation_frequency": 8,
    "policy_frequency": 4,
    "collision_reward": -5.0,
    "right_lane_reward": 0.02,
    "high_speed_reward": 0.70,
    "lane_change_reward": 0.12,
    "reward_speed_range": [20, 30],
    "normalize_reward": True,
    "offroad_terminal": True,
}

# ---- Curriculum Phases (kept for reference, not used in experiments) ----
CURRICULUM_PHASES = [
    {"name": "phase1_light", "ratio": 0.20, "overrides": {"vehicles_count": 14, "vehicles_density": 0.75, "duration": 55}},
    {"name": "phase2_medium", "ratio": 0.35, "overrides": {"vehicles_count": 24, "vehicles_density": 1.05, "duration": 62}},
    {"name": "phase3_target", "ratio": 0.45, "overrides": {}},
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
TOTAL_TIMESTEPS = 50_000
N_ENVS = 4
EVAL_FREQ_STEPS = 8_000
N_EVAL_EPISODES = 10
REWARD_WINDOW = 10

# ---- RSS Parameters (adjusted penalty) ----
RSS_CONFIG = {
    "response_time": 0.8,
    "rear_response_time": 0.6,
    "min_distance": 5.0,
    "max_brake": 6.0,
    "ttc_threshold": 2.0,
    "intervention_penalty": -1.0,
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
        "label": "PPO + RSS (Ours, penalty=-1.0)",
        "use_rss": True,
        "use_curriculum": False,
        "rss_overrides": {"intervention_penalty": -1.0},
        "color": "#2ca02c",
        "linestyle": "-",
        "marker": "s",
    },
    "ablation_rss_harsh": {
        "name": "ablation_rss_harsh",
        "label": "PPO + RSS (Harsh penalty=-2.5)",
        "use_rss": True,
        "use_curriculum": False,
        "rss_overrides": {"intervention_penalty": -2.5},
        "color": "#ff7f0e",
        "linestyle": "--",
        "marker": "^",
    },
}

ACTIVE_EXPERIMENTS = ["baseline", "our_method", "ablation_rss_harsh"]
