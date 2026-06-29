"""Game configuration and driving style weights for Stackelberg lane-change model.

References:
    Shi B, Zhai L, Liu C. Stackelberg Game Based on Trajectory Prediction
    for Lane Change in Mixed Traffic. IEEE Access.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class GameConfig:
    """Hyperparameters for the Stackelberg game solver and FSM executor."""

    # ---- Vehicle physics ----
    max_accel: float = 3.0
    max_brake: float = 6.0
    min_safe_distance: float = 5.0
    lc_duration: float = 3.0
    tau_system: float = 0.5       # τ1: driving system response time (s)
    tau_driver: float = 0.6       # τ2: driver reaction time (s)
    desired_speed: float = 25.0   # v_desire (m/s)
    max_speed_deviation: float = 10.0

    # ---- EV cost weights (paper: ω_EV_safe, ω_EV_com, ω_EV_eff) ----
    # Safety weights raised to dominate comfort/efficiency when TTC is low.
    # With C_safe ≈ 200 (TTC=1s), C_com ≈ 18, C_eff ≈ 225 (hard brake):
    #   total_safe = 10 * 200 = 2000  >>  0.3*18 + 0.1*225 = 27.9
    w_ev_safe_long: float = 2.0
    w_ev_safe_lat: float = 2.0
    w_ev_com: float = 0.3
    w_ev_eff: float = 0.5

    # ---- Cost internal weights (paper eq.13-14) ----
    k_v_long: float = 0.5         # velocity weight in longitudinal safety
    k_s_long: float = 5.0         # distance weight in longitudinal safety
    k_v_lat: float = 0.8          # velocity weight in lateral safety
    k_s_lat: float = 15.0         # distance weight in lateral safety
    vehicle_length: float = 5.0   # l, for safety factor v_l
    safety_eps: float = 1e-3      # ε, prevent division by zero

    # ---- Game search ----
    accel_candidates: Tuple[float, ...] = (-3.0, -1.5, 0.0, 1.0, 2.0, 3.0)
    prediction_horizon: float = 5.0
    dt: float = 0.1
    attenuation_factor: float = 0.85  # ξ: future payoff decay

    # ---- FSM (tech roadmap Section 2.1.2) ----
    lc_cooldown: float = 2.0
    lc_prep_timeout: float = 1.5      # max time in LC_PREPARATION
    lc_exec_timeout: float = 5.0      # max time in LC_EXECUTION
    cost_improvement_threshold: float = 0.10  # ε in formula (4-7)
    ttc_safe_threshold: float = 5.0   # minimum safe TTC for FSM gating
    gap_safety_margin: float = 1.2    # multiplicative safety margin on RSS distance
    max_jerk: float = 2.5             # max jerk for rate limiting (m/s³)
    lateral_speed_threshold: float = 0.5  # lane-change detection threshold
    fsm_safety_horizon: float = 3.0   # FSM safety gate prediction horizon (s)
    min_cruise_speed: float = 15.0    # minimum highway cruising speed (m/s ≈ 54 km/h)
    rear_ttc_warning: float = 6.0     # rear TTC threshold: limit braking below this
    rear_ttc_critical: float = 3.0    # rear TTC threshold: abort braking below this
    rear_gap_warning: float = 30.0    # rear gap threshold (m): warn when closer than this


# ---- Driving Style Weights (paper Table 1) ----
# HV utility: J_HV = ω_safe·J_safe + ω_space·J_space + ω_com·J_com + ω_eff·J_eff

DRIVING_STYLE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "aggressive": {
        "w_safe": 0.5,
        "w_space": 0.5,
        "w_com": 0.1,
        "w_eff": 0.8,
    },
    "normal": {
        "w_safe": 1.0,
        "w_space": 1.0,
        "w_com": 0.3,
        "w_eff": 0.2,
    },
    "conservative": {
        "w_safe": 2.0,
        "w_space": 2.0,
        "w_com": 0.5,
        "w_eff": 0.1,
    },
}
