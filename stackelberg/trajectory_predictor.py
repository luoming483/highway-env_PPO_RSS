"""Trajectory prediction using linear decay acceleration model.

Implements paper formulas (4-2) ~ (4-4):
    a(t) = a0 - k * t
    v(t) = v0 + a0*t - 0.5*k*t^2
    x(t) = x0 + v0*t + 0.5*a0*t^2 - (1/6)*k*t^3

The decay rate k captures the fact that real drivers gradually reduce
acceleration/braking intensity rather than maintaining constant values.
k > 0 means acceleration decays (throttle fade), k < 0 means braking
decays (brake release).
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class TrajectoryPoint:
    t: float
    x: float
    y: float
    v: float
    a: float


@dataclass
class VehicleState:
    x: float
    y: float
    vx: float
    vy: float
    ax: float = 0.0
    ay: float = 0.0
    heading: float = 0.0


def estimate_decay_rate(a0: float, max_accel: float, horizon: float) -> float:
    """Estimate k so that acceleration decays toward 0 over horizon.

    For throttle (a0 > 0):  k > 0,  a(t) = a0 - k*t → k = a0 / horizon
    For brake   (a0 < 0):  k < 0,  a(t) = a0 - k*t → k = a0 / horizon
    This ensures a(horizon) ≈ 0.

    Returns k clamped so |a(t)| <= max_accel for all t in [0, horizon].
    """
    k_raw = a0 / max(horizon, 0.1)
    max_k = 2.0 * max_accel / max(horizon, 0.1)
    return float(np.clip(k_raw, -max_k, max_k))


def predict_trajectory(
    initial: VehicleState,
    a0: float,
    k: float,
    horizon: float,
    dt: float = 0.1,
    max_speed: float = 40.0,
    min_speed: float = 0.0,
) -> List[TrajectoryPoint]:
    """Predict vehicle trajectory using linear decay acceleration model.

    Args:
        initial: Starting vehicle state.
        a0: Initial longitudinal acceleration (m/s^2).
        k: Acceleration decay rate (positive = throttle fades, negative = brake fades).
        horizon: Prediction horizon in seconds.
        dt: Time step.
        max_speed: Speed clamp upper bound.
        min_speed: Speed clamp lower bound.

    Returns:
        List of TrajectoryPoint for t = 0, dt, 2*dt, ..., horizon.
    """
    n_steps = int(horizon / dt) + 1
    points: List[TrajectoryPoint] = []

    v0 = float(np.linalg.norm([initial.vx, initial.vy]))
    heading = initial.heading

    for i in range(n_steps):
        t = i * dt
        t2 = t * t
        t3 = t2 * t

        # Formula (4-2): a(t)
        a = a0 - k * t

        # Formula (4-3): v(t), with physical clamping
        v_raw = v0 + a0 * t - 0.5 * k * t2
        v = float(np.clip(v_raw, min_speed, max_speed))

        # Formula (4-4): x(t)
        x_raw = initial.x + v0 * t + 0.5 * a0 * t2 - (1.0 / 6.0) * k * t3
        x = x_raw

        # Lateral position (simplified: constant lateral speed if lane-changing)
        y = initial.y

        points.append(TrajectoryPoint(t=float(t), x=float(x), y=float(y), v=v, a=float(a)))

    return points


def predict_hv_response(
    hv_state: VehicleState,
    ev_action_longitudinal: float,
    target_speed: float,
    horizon: float = 5.0,
    dt: float = 0.1,
) -> List[TrajectoryPoint]:
    """Predict HV trajectory when responding to EV's lane-change action.

    The HV adjusts toward target_speed using a linear-decay acceleration
    profile.  The initial acceleration a0 is set proportional to the
    speed error, and k decays it toward zero.

    Args:
        hv_state: Current HV state.
        ev_action_longitudinal: EV's longitudinal acceleration (for context).
        target_speed: HV's optimal target speed (from game solution).
        horizon: Prediction horizon.
        dt: Time step.

    Returns:
        HV trajectory as list of TrajectoryPoint.
    """
    v_current = float(np.linalg.norm([hv_state.vx, hv_state.vy]))
    dv = target_speed - v_current

    # Initial acceleration proportional to speed error, bounded by physics
    max_a = 3.0
    a0 = float(np.clip(dv / 2.0, -max_a, max_a))

    k = estimate_decay_rate(a0, max_a, horizon)
    return predict_trajectory(hv_state, a0, k, horizon, dt)


def predict_ev_candidate(
    ev_state: VehicleState,
    longitudinal_accel: float,
    lateral_action: int,  # -1=LEFT, 0=STAY, 1=RIGHT (matching CandidateAction.lateral)
    horizon: float = 5.0,
    dt: float = 0.1,
    lane_width: float = 4.0,
    lc_duration: float = 3.0,
) -> List[TrajectoryPoint]:
    """Predict EV trajectory for a candidate action.

    Combines longitudinal motion (linear decay acceleration) with a
    simplified lateral model for lane changes.

    Args:
        ev_state: Current EV state.
        longitudinal_accel: Target longitudinal acceleration at t=0.
        lateral_action: -1=change left, 0=stay, 1=change right.
        horizon: Prediction horizon.
        dt: Time step.
        lane_width: Width of one lane (m).
        lc_duration: Expected lane-change duration (s).

    Returns:
        EV trajectory as list of TrajectoryPoint.
    """
    n_steps = int(horizon / dt) + 1
    k = estimate_decay_rate(longitudinal_accel, 3.0, horizon)

    v0 = float(np.linalg.norm([ev_state.vx, ev_state.vy]))
    heading = ev_state.heading

    # Lateral target
    if lateral_action == -1:  # LEFT
        y_target = ev_state.y + lane_width
    elif lateral_action == 1:  # RIGHT
        y_target = ev_state.y - lane_width
    else:
        y_target = ev_state.y

    points: List[TrajectoryPoint] = []
    t_switch = min(lc_duration, horizon)

    for i in range(n_steps):
        t = i * dt
        t2 = t * t
        t3 = t2 * t

        # Longitudinal: formulas (4-2)~(4-4)
        a = longitudinal_accel - k * t
        v = float(np.clip(v0 + longitudinal_accel * t - 0.5 * k * t2, 0.0, 40.0))
        x = ev_state.x + v0 * t + 0.5 * longitudinal_accel * t2 - (1.0 / 6.0) * k * t3

        # Lateral: sinusoidal lane-change profile
        if lateral_action in (-1, 1) and t <= t_switch:
            progress = t / t_switch
            # Smooth sigmoid-like transition for lateral motion
            lat_offset = (y_target - ev_state.y) * (3.0 * progress**2 - 2.0 * progress**3)
            y = ev_state.y + lat_offset
        else:
            y = y_target if t > t_switch else ev_state.y

        points.append(TrajectoryPoint(t=float(t), x=float(x), y=float(y), v=float(v), a=float(a)))

    return points


def compute_ttc_gap(
    ego_traj: List[TrajectoryPoint],
    other_traj: List[TrajectoryPoint],
    vehicle_length: float = 5.0,
) -> tuple:
    """Compute TTC and gap evolution along two trajectories.

    Paper formulas (5) and (6) for TTC and headway time.

    Returns (min_ttc, min_gap, ttc_at_end, gap_at_end).
    """
    min_ttc = np.inf
    min_gap = np.inf

    for ego_pt, other_pt in zip(ego_traj, other_traj):
        dx = other_pt.x - ego_pt.x
        gap = float(abs(dx) - vehicle_length)
        min_gap = min(min_gap, gap)

        dv = ego_pt.v - other_pt.v

        if gap > 0.0 and dv > 1e-6:
            ttc = gap / dv
        elif gap <= 0.0:
            ttc = 0.0
        else:
            ttc = np.inf

        min_ttc = min(min_ttc, ttc)

    # End-of-horizon values
    ego_end = ego_traj[-1]
    other_end = other_traj[-1]
    gap_end = float(abs(other_end.x - ego_end.x) - vehicle_length)
    dv_end = ego_end.v - other_end.v
    ttc_end = gap_end / dv_end if (gap_end > 0.0 and dv_end > 1e-6) else np.inf

    return float(min_ttc), float(min_gap), float(ttc_end), float(gap_end)
