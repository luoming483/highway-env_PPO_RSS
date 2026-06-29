"""HV utility (payoff) and EV cost functions for Stackelberg lane-change game.

Implements paper formulas:

HV utility (eq.7-11):
    J_HV = ω_safe·J_safe + ω_space·J_space + ω_com·J_com + ω_eff·J_eff

    J_safe  = τ / (Th(t_lc) + a*·TTC(t_lc)/v_rel(t_lc))
    J_space = { -1        if Δx ≤ -Δx_lim
              {  Δx/Δx_lim if -Δx_lim < Δx < Δx_lim
              {  1         if Δx ≥ Δx_lim
    J_com   = -max|a_HV| / a_max
    J_eff   = -max|v_HV(t) - v_target| / Δv_max

EV cost (eq.13-18):
    C_EV = ω_safe·C_safe + ω_com·C_com + ω_eff·C_eff

    C_safe = C_safe_long + C_safe_lat
    C_com  = a_EV² + Δa_EV²
    C_eff  = (v_EV - v_desire)²

Prediction-weighted (eq.19):
    J(t) = Σ ξ^k · J(state_t, s_EV, s_HV_k)
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .config import DRIVING_STYLE_WEIGHTS, GameConfig
from .trajectory_predictor import TrajectoryPoint, VehicleState, compute_ttc_gap


@dataclass
class UtilityResult:
    """Breakdown of HV utility or EV cost."""
    total: float
    safe: float
    space_or_lat: float  # J_space for HV, C_safe_lat for EV
    com: float
    eff: float
    details: dict = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


def hv_safety_payoff(
    ttc_at_lc_end: float,
    headway_at_lc_end: float,
    v_rel_at_lc_end: float,
    tau: float = 1.1,
    a_brake: float = 6.0,
) -> float:
    """Compute HV safety payoff J_safe — paper formula (7).

    J_safe = τ / (Th(t_lc) + a* · TTC(t_lc) / v_rel(t_lc))

    Where:
        τ = τ1 + τ2 (system + driver reaction time)
        Th(t_lc) = headway time at end of lane change
        TTC(t_lc) = time-to-collision at end of lane change
        a* = braking deceleration (6 m/s²)
        v_rel = relative speed at end of lane change

    Higher J_safe = safer situation.
    """
    denominator = headway_at_lc_end + a_brake * max(ttc_at_lc_end, 0.0) / max(abs(v_rel_at_lc_end), 1e-3)
    if denominator <= 1e-6:
        return 10.0  # very safe (denominator near zero = huge safety margin)
    return float(tau / denominator)


def hv_space_payoff(dx: float, dx_lim: float) -> float:
    """Compute HV positional advantage J_space — paper formula (8).

    J_space = {
        -1          if Δx ≤ -Δx_lim       (EV is behind, much space for HV)
        Δx / Δx_lim  if -Δx_lim < Δx < Δx_lim  (linear region)
        1           if Δx ≥ Δx_lim        (EV is ahead, little space)
    }

    Δx = x_EV - x_HV (positive = EV ahead of HV).
    Higher J_space = more space for HV (EV further ahead).
    """
    ratio = dx / max(dx_lim, 1e-3)
    return float(np.clip(ratio, -1.0, 1.0))


def hv_comfort_payoff(
    a_history: List[float],
    max_accel: float = 3.0,
) -> float:
    """Compute HV comfort payoff J_com — paper formula (9).

    J_com = -max|a_HV(t)| / a_max

    Closer to 0 = more comfortable. -1 = maximum discomfort.
    """
    if not a_history:
        return 0.0
    max_abs_a = max(abs(a) for a in a_history)
    return float(-max_abs_a / max(max_accel, 1e-3))


def hv_efficiency_payoff(
    v_actual: float,
    v_target: float,
    dv_max: float = 10.0,
) -> float:
    """Compute HV efficiency payoff J_eff — paper formula (10).

    J_eff = -max|v_HV(t) - v_target| / Δv_max

    Closer to 0 = better speed tracking.
    """
    return float(-abs(v_actual - v_target) / max(dv_max, 1e-3))


def compute_hv_utility(
    ev_traj: List[TrajectoryPoint],
    hv_traj: List[TrajectoryPoint],
    hv_target_speed: float,
    style_weights: Dict[str, float],
    tau: float = 1.1,
    a_brake: float = 6.0,
    dx_lim: float = 100.0,
    max_accel: float = 3.0,
    dv_max: float = 10.0,
) -> UtilityResult:
    """Compute total HV utility J_HV — paper formula (11).

    J_HV = ω_safe·J_safe + ω_space·J_space + ω_com·J_com + ω_eff·J_eff
    """
    # Safety: from trajectory end state
    _, _, ttc_end, gap_end = compute_ttc_gap(ev_traj, hv_traj)
    ev_end, hv_end = ev_traj[-1], hv_traj[-1]
    v_rel_end = ev_end.v - hv_end.v
    headway_end = gap_end / max(hv_end.v, 1e-3)

    j_safe = hv_safety_payoff(ttc_end, headway_end, v_rel_end, tau, a_brake)

    # Space: position gap at end of lane change
    dx_end = ev_end.x - hv_end.x
    j_space = hv_space_payoff(dx_end, dx_lim)

    # Comfort: max acceleration during response
    a_hist = [p.a for p in hv_traj]
    j_com = hv_comfort_payoff(a_hist, max_accel)

    # Efficiency: speed deviation at end
    j_eff = hv_efficiency_payoff(hv_end.v, hv_target_speed, dv_max)

    w = style_weights
    total = (
        w["w_safe"] * j_safe
        + w["w_space"] * j_space
        + w["w_com"] * j_com
        + w["w_eff"] * j_eff
    )

    return UtilityResult(
        total=float(total),
        safe=float(j_safe),
        space_or_lat=float(j_space),
        com=float(j_com),
        eff=float(j_eff),
        details={
            "ttc_end": float(ttc_end),
            "headway_end": float(headway_end),
            "dx_end": float(dx_end),
            "v_rel_end": float(v_rel_end),
        },
    )


def ev_safety_cost_longitudinal(
    ev_state: VehicleState,
    fv_state: VehicleState,
    config: GameConfig,
    min_ttc: float = None,
    min_gap: float = None,
) -> float:
    """Compute EV longitudinal safety cost — paper formula (13).

    C_safe_long = k_v·Δv² + k_s·(1/Δs²) * λ_v

    where:
        Δv = v_FV - v_EV
        Δs = distance between EV and FV (minus vehicle length)
        λ_v = 1 if Δv < 0 (closing), 0 otherwise

    If min_ttc/min_gap are provided (pre-computed along the trajectory),
    they override the endpoint-only calculation for more accurate danger assessment.
    """
    dx = fv_state.x - ev_state.x
    dv = fv_state.vx - ev_state.vx
    ds = abs(dx) - config.vehicle_length

    # Use trajectory-minimum gap if provided
    effective_ds = min(ds, min_gap) if min_gap is not None else ds

    if effective_ds <= 0.0:
        return 500.0  # collision

    # Use pre-computed min TTC if available, otherwise compute from endpoint
    if min_ttc is not None:
        ttc = min_ttc
    else:
        rel_speed = ev_state.vx - fv_state.vx  # positive = ego faster (closing)
        if effective_ds > 0.0 and rel_speed > 1e-6:
            ttc = effective_ds / rel_speed
        else:
            ttc = float('inf')

    # TTC-based exponential penalty: dominates when TTC < 3s
    ttc_penalty = 0.0
    if ttc < 3.0:
        ttc_penalty = 50.0 * np.exp(-ttc)

    lambda_v = 1.0 if dv < 0 else 0.0
    cost_v = config.k_v_long * dv * dv
    cost_s = config.k_s_long * (1.0 / max(effective_ds * effective_ds, config.safety_eps)) * lambda_v
    return float(cost_v + cost_s + ttc_penalty)


def ev_safety_cost_lateral(
    ev_state: VehicleState,
    rv_state: VehicleState,
    config: GameConfig,
) -> float:
    """Compute EV lateral safety cost — paper formula (14).

    C_safe_lat = k_v_lat·Δv_lat² + k_s_lat·(1/Δs_lat²) * λ_v_lat

    where:
        Δv_lat = v_RV_lat - v_EV_lat
        Δs_lat = lateral distance (minus vehicle width)
        λ_v_lat = 1 if Δv_lat pulls vehicles closer, 0 otherwise

    Note: when dy ≈ 0 (same lane), no lateral collision risk — longitudinal
    safety handles that case. The penalty only applies for partial lateral overlap.
    """
    dy = rv_state.y - ev_state.y
    dv_lat = rv_state.vy - ev_state.vy
    ds_lat = abs(dy) - config.vehicle_length / 2.0  # lateral

    # Only penalize partial lateral overlap (different lanes, close laterally).
    # dy ≈ 0 means same lane — longitudinal safety covers this.
    if ds_lat <= 0.0 and abs(dy) > 0.1:
        return 50.0  # side collision risk

    if abs(dy) <= 0.1:
        return 0.0  # same lane, no lateral risk

    lambda_v_lat = 1.0 if dv_lat * dy < 0 else 0.0  # moving toward each other
    cost_v = config.k_v_lat * dv_lat * dv_lat
    cost_s = config.k_s_lat * (1.0 / max(ds_lat * ds_lat, config.safety_eps)) * lambda_v_lat
    return float(cost_v + cost_s)


def ev_comfort_cost(
    a_current: float,
    a_previous: float,
) -> float:
    """Compute EV ride comfort cost — paper formula (15).

    C_com = a_EV² + Δa_EV²

    This penalizes both the magnitude of acceleration AND jerk.
    """
    da = a_current - a_previous
    return float(a_current * a_current + da * da)


def ev_efficiency_cost(
    v_current: float,
    v_desire: float,
) -> float:
    """Compute EV driving efficiency cost — paper formula (16).

    C_eff = (v_EV - v_desire)²
    """
    return float((v_current - v_desire) ** 2)


def compute_ev_cost(
    ev_traj: List[TrajectoryPoint],
    fv_state: VehicleState,
    rv_state: VehicleState,
    config: GameConfig,
    prev_accel: float = 0.0,
    horizon: float = None,
) -> UtilityResult:
    """Compute total EV cost — paper formula (17,18).

    C_EV = ω_safe·(C_safe_long + C_safe_lat) + ω_com·C_com + ω_eff·C_eff

    The FV and RV states are propagated forward over the prediction horizon
    so that safety is evaluated at the same future timestep as the EV trajectory.
    """
    ev_end = ev_traj[-1]

    # Use provided horizon or infer from trajectory
    if horizon is None:
        horizon = ev_end.t

    # ---- Propagate FV forward to match prediction horizon ----
    fv_future = VehicleState(
        x=fv_state.x + fv_state.vx * horizon,
        y=fv_state.y,
        vx=fv_state.vx,
        vy=fv_state.vy,
    )

    # ---- Compute min TTC and min gap along the trajectory ----
    min_ttc = float('inf')
    min_gap = float('inf')
    for ev_pt in ev_traj:
        fv_x_t = fv_state.x + fv_state.vx * ev_pt.t
        gap = abs(fv_x_t - ev_pt.x) - config.vehicle_length
        if gap < min_gap:
            min_gap = gap
        rel_speed = ev_pt.v - fv_state.vx  # positive = EV faster (closing)
        if gap <= 0.0:
            min_ttc = 0.0
        elif rel_speed > 1e-6:
            ttc = gap / rel_speed
            if ttc < min_ttc:
                min_ttc = ttc

    # ---- Safety costs ----
    c_safe_long = ev_safety_cost_longitudinal(
        VehicleState(x=ev_end.x, y=ev_end.y, vx=ev_end.v, vy=0.0),
        fv_future,
        config,
        min_ttc=min_ttc,
        min_gap=min_gap,
    )

    c_safe_lat = 0.0
    if rv_state is not None:
        rv_future = VehicleState(
            x=rv_state.x + rv_state.vx * horizon,
            y=rv_state.y,
            vx=rv_state.vx,
            vy=rv_state.vy,
        )
        c_safe_lat = ev_safety_cost_lateral(
            VehicleState(x=ev_end.x, y=ev_end.y, vx=ev_end.v, vy=0.0),
            rv_future,
            config,
        )

    # ---- Comfort: max acceleration + jerk over trajectory ----
    a_vals = [p.a for p in ev_traj]
    max_com = 0.0
    for i, a in enumerate(a_vals):
        prev = a_vals[i - 1] if i > 0 else prev_accel
        max_com = max(max_com, ev_comfort_cost(a, prev))

    # ---- Efficiency: final speed deviation ----
    c_eff = ev_efficiency_cost(ev_end.v, config.desired_speed)

    c_safe = c_safe_long + c_safe_lat
    total = (
        (config.w_ev_safe_long + config.w_ev_safe_lat) * c_safe
        + config.w_ev_com * max_com
        + config.w_ev_eff * c_eff
    )

    return UtilityResult(
        total=float(total),
        safe=float(c_safe),
        space_or_lat=float(c_safe_lat),
        com=float(max_com),
        eff=float(c_eff),
        details={
            "safe_long": float(c_safe_long),
            "safe_lat": float(c_safe_lat),
            "min_ttc": float(min_ttc),
            "min_gap": float(min_gap),
        },
    )


def compute_prediction_weighted_payoff(
    traj_list: List[List[TrajectoryPoint]],
    payoff_fn,
    attenuation: float = 0.85,
) -> float:
    """Compute prediction-weighted payoff — paper formula (19).

    J(t) = Σ_{k=0}^{N} ξ^k · J(state_t, s_EV, s_HV_k)

    Future payoffs are discounted by ξ^k, giving more weight to
    near-term outcomes.
    """
    total = 0.0
    for k, traj in enumerate(traj_list):
        weight = attenuation ** k
        total += weight * payoff_fn(traj)
    return float(total)
