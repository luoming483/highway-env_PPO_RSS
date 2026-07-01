"""Unit tests for Stackelberg game solver and FSM executor.

Usage:
    D:\\anaconda\\envs\\ppo_main\\python.exe -m stackelberg.test_units
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from stackelberg.config import DRIVING_STYLE_WEIGHTS, GameConfig
from stackelberg.fsm_executor import (
    ACTION_FASTER,
    ACTION_IDLE,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_SLOWER,
    FSMExecutor,
    FSMState,
    FSMSafetyGate,
    RateLimiter,
)
from stackelberg.game_solver import CandidateAction, GameResult, StackelbergSolver
from stackelberg.trajectory_predictor import (
    TrajectoryPoint,
    VehicleState,
    compute_ttc_gap,
    estimate_decay_rate,
    predict_ev_candidate,
    predict_hv_response,
    predict_trajectory,
)
from stackelberg.utility_functions import (
    compute_ev_cost,
    compute_hv_utility,
    compute_prediction_weighted_payoff,
    ev_comfort_cost,
    ev_efficiency_cost,
    ev_safety_cost_lateral,
    ev_safety_cost_longitudinal,
    hv_comfort_payoff,
    hv_efficiency_payoff,
    hv_safety_payoff,
    hv_space_payoff,
)

_passed = 0
_failed = 0


def check(condition, label):
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  PASS: {label}")
    else:
        _failed += 1
        print(f"  FAIL: {label}")


def check_approx(a, b, label, tol=1e-6):
    ok = abs(a - b) < tol
    if not ok:
        print(f"    expected={b:.6f}  got={a:.6f}")
    check(ok, label)


# ============================================================
# trajectory_predictor
# ============================================================
def test_estimate_decay_rate():
    print("\n[trajectory_predictor] estimate_decay_rate")
    k = estimate_decay_rate(3.0, 3.0, 5.0)
    check_approx(k, 0.6, "positive a0 -> k>0")

    k = estimate_decay_rate(-3.0, 3.0, 5.0)
    check_approx(k, -0.6, "negative a0 -> k<0")

    k = estimate_decay_rate(0.0, 3.0, 5.0)
    check_approx(k, 0.0, "zero a0 -> k=0")


def test_predict_trajectory_basic():
    print("\n[trajectory_predictor] predict_trajectory basic")
    s = VehicleState(x=0.0, y=0.0, vx=20.0, vy=0.0)
    traj = predict_trajectory(s, a0=2.0, k=0.4, horizon=5.0, dt=1.0)

    check(len(traj) == 6, "6 points for horizon=5 dt=1")
    check_approx(traj[0].t, 0.0, "t0=0")
    check_approx(traj[0].v, 20.0, "v0=20")
    check_approx(traj[0].a, 2.0, "a0=2")
    check(traj[0].x >= 0.0, "x0 >= 0")
    # a(t) = 2 - 0.4*t, so at t=5: a = 2 - 2 = 0
    check_approx(traj[5].a, 0.0, "a(5)=0", tol=0.01)
    check(traj[5].v >= 20.0, "speed increases with positive a0")


def test_predict_trajectory_deceleration():
    print("\n[trajectory_predictor] predict_trajectory deceleration")
    s = VehicleState(x=0.0, y=0.0, vx=20.0, vy=0.0)
    traj = predict_trajectory(s, a0=-3.0, k=-0.6, horizon=5.0, dt=1.0)

    check(traj[0].a == -3.0, "a0 = -3")
    # speed should decrease
    check(traj[5].v < 20.0, "speed decreases with negative a0")


def test_predict_trajectory_speed_clamp():
    print("\n[trajectory_predictor] predict_trajectory speed clamp")
    s = VehicleState(x=0.0, y=0.0, vx=38.0, vy=0.0)
    traj = predict_trajectory(s, a0=3.0, k=0.1, horizon=3.0, dt=0.5, max_speed=40.0)
    for pt in traj:
        check(pt.v <= 40.0, f"v<={40} at t={pt.t:.1f}")
        check(pt.v >= 0.0, f"v>=0 at t={pt.t:.1f}")


def test_predict_hv_response():
    print("\n[trajectory_predictor] predict_hv_response")
    hv = VehicleState(x=100.0, y=0.0, vx=20.0, vy=0.0)
    traj = predict_hv_response(hv, ev_action_longitudinal=1.0,
                               target_speed=25.0, horizon=5.0, dt=0.5)
    check(len(traj) == 11, "11 points for horizon=5 dt=0.5")
    # HV should accelerate toward target_speed
    check(traj[0].a > 0.0, "HV accelerates toward target speed")


def test_predict_ev_candidate_lane_change():
    print("\n[trajectory_predictor] predict_ev_candidate lane change")
    ev = VehicleState(x=0.0, y=0.0, vx=20.0, vy=0.0)
    # LEFT lane change
    traj = predict_ev_candidate(ev, longitudinal_accel=1.0,
                                lateral_action=-1, horizon=5.0, dt=0.25,
                                lane_width=4.0, lc_duration=3.0)
    check(len(traj) == 21, "21 points for horizon=5 dt=0.25")
    # y should increase (left = +lane_width)
    check(traj[0].y == 0.0, "y starts at 0")
    check(traj[-1].y > 3.5, "y ends near +4 (LEFT)")
    # RIGHT lane change
    traj_r = predict_ev_candidate(ev, longitudinal_accel=0.0,
                                  lateral_action=1, horizon=5.0, dt=0.25,
                                  lane_width=4.0, lc_duration=3.0)
    check(traj_r[-1].y < -3.5, "y ends near -4 (RIGHT)")


def test_compute_ttc_gap():
    print("\n[trajectory_predictor] compute_ttc_gap")
    ego = [TrajectoryPoint(t=0, x=0, y=0, v=25, a=0),
           TrajectoryPoint(t=1, x=25, y=0, v=25, a=0)]
    other = [TrajectoryPoint(t=0, x=30, y=0, v=20, a=0),
             TrajectoryPoint(t=1, x=50, y=0, v=20, a=0)]
    min_ttc, min_gap, ttc_end, gap_end = compute_ttc_gap(ego, other, vehicle_length=5.0)
    # t=0: gap=30-0-5=25, dv=5 -> ttc=5
    # t=1: gap=50-25-5=20, dv=5 -> ttc=4  ← min over trajectory
    check_approx(min_ttc, 4.0, "min ttc=4s (gap shrinks from 25->20, dv=5)")
    check_approx(min_gap, 20.0, "min gap=20m")


def test_compute_ttc_gap_collision():
    print("\n[trajectory_predictor] compute_ttc_gap collision")
    ego = [TrajectoryPoint(t=0, x=0, y=0, v=25, a=0)]
    other = [TrajectoryPoint(t=0, x=3, y=0, v=20, a=0)]
    min_ttc, min_gap, _, _ = compute_ttc_gap(ego, other, vehicle_length=5.0)
    check(min_gap < 0, "negative gap = collision")
    check(min_ttc == 0.0, "ttc=0 for collision")


# ============================================================
# utility_functions — HV
# ============================================================
def test_hv_safety_payoff():
    print("\n[utility_functions] hv_safety_payoff")
    # Large TTC, large headway → safe
    j1 = hv_safety_payoff(ttc_at_lc_end=10.0, headway_at_lc_end=5.0,
                          v_rel_at_lc_end=2.0, tau=1.1, a_brake=6.0)
    check(j1 > 0.0, "safe scenario gives positive payoff")

    # Small TTC — note: formula J=τ/(Th+a*·TTC/v_rel), smaller denominator = larger payoff
    j2 = hv_safety_payoff(ttc_at_lc_end=1.0, headway_at_lc_end=0.5,
                          v_rel_at_lc_end=5.0, tau=1.1, a_brake=6.0)
    check(j2 > j1, "tighter following → larger safety payoff (inverse denominator)")


def test_hv_space_payoff():
    print("\n[utility_functions] hv_space_payoff")
    # EV far ahead → high payoff for HV
    j = hv_space_payoff(dx=200.0, dx_lim=100.0)
    check_approx(j, 1.0, "EV far ahead → +1")

    # EV far behind → low payoff
    j = hv_space_payoff(dx=-200.0, dx_lim=100.0)
    check_approx(j, -1.0, "EV far behind → -1")

    # EV just ahead
    j = hv_space_payoff(dx=50.0, dx_lim=100.0)
    check_approx(j, 0.5, "EV at 50/100 → 0.5")


def test_hv_comfort_payoff():
    print("\n[utility_functions] hv_comfort_payoff")
    j = hv_comfort_payoff([0.5, -0.3, 0.1], max_accel=3.0)
    check_approx(j, -0.5 / 3.0, "max|a|=0.5 → -0.167")

    j = hv_comfort_payoff([], max_accel=3.0)
    check_approx(j, 0.0, "empty history → 0")


def test_hv_efficiency_payoff():
    print("\n[utility_functions] hv_efficiency_payoff")
    j = hv_efficiency_payoff(v_actual=25.0, v_target=25.0, dv_max=10.0)
    check_approx(j, 0.0, "at target speed → 0")

    j = hv_efficiency_payoff(v_actual=15.0, v_target=25.0, dv_max=10.0)
    check_approx(j, -1.0, "10 m/s off → -1")


def test_compute_hv_utility():
    print("\n[utility_functions] compute_hv_utility")
    ev_traj = [TrajectoryPoint(t=0, x=0, y=0, v=25, a=1),
               TrajectoryPoint(t=1, x=26, y=0, v=26, a=0)]
    hv_traj = [TrajectoryPoint(t=0, x=-20, y=0, v=22, a=0.5),
               TrajectoryPoint(t=1, x=2, y=0, v=22.5, a=0)]
    weights = DRIVING_STYLE_WEIGHTS["normal"]
    result = compute_hv_utility(ev_traj, hv_traj, hv_target_speed=25.0,
                                style_weights=weights)
    check(isinstance(result.total, float), "returns float total")
    check(isinstance(result.safe, float), "returns float safe")
    check(isinstance(result.com, float), "returns float com")


# ============================================================
# utility_functions — EV
# ============================================================
def test_ev_safety_cost_longitudinal_collision():
    print("\n[utility_functions] ev_safety_cost_longitudinal collision")
    cfg = GameConfig()
    ev = VehicleState(x=10.0, y=0.0, vx=25.0, vy=0.0)
    fv = VehicleState(x=12.0, y=0.0, vx=20.0, vy=0.0)  # only 2m ahead
    cost = ev_safety_cost_longitudinal(ev, fv, cfg)
    # ds = 2 - 5 = -3 → collision penalty
    check(cost > 400.0, "collision penalty > 400")


def test_ev_safety_cost_longitudinal_normal():
    print("\n[utility_functions] ev_safety_cost_longitudinal normal")
    cfg = GameConfig()
    ev = VehicleState(x=0.0, y=0.0, vx=25.0, vy=0.0)
    fv = VehicleState(x=50.0, y=0.0, vx=20.0, vy=0.0)
    cost = ev_safety_cost_longitudinal(ev, fv, cfg)
    check(cost < 100.0, "normal gap gives low cost")
    check(cost >= 0.0, "cost non-negative")


def test_ev_safety_cost_longitudinal_ttc_penalty():
    print("\n[utility_functions] ev_safety_cost_longitudinal TTC penalty")
    cfg = GameConfig()
    ev = VehicleState(x=0.0, y=0.0, vx=25.0, vy=0.0)
    fv = VehicleState(x=10.0, y=0.0, vx=20.0, vy=0.0)  # gap=5m, dv=5 → ttc=1s
    cost = ev_safety_cost_longitudinal(ev, fv, cfg, min_ttc=1.0, min_gap=5.0)
    # ttc_penalty = 50 * exp(-1) ≈ 18.4
    check(cost > 10.0, "TTC penalty added for TTC<3s")


def test_ev_safety_cost_lateral():
    print("\n[utility_functions] ev_safety_cost_lateral")
    cfg = GameConfig()
    # Same lane — no lateral risk
    ev = VehicleState(x=0.0, y=0.0, vx=25.0, vy=0.0)
    rv = VehicleState(x=-20.0, y=0.0, vx=22.0, vy=0.0)
    cost = ev_safety_cost_lateral(ev, rv, cfg)
    check_approx(cost, 0.0, "same lane → 0 lateral cost")

    # Different lane, close laterally
    ev2 = VehicleState(x=0.0, y=0.0, vx=25.0, vy=0.0)
    rv2 = VehicleState(x=-20.0, y=0.5, vx=22.0, vy=0.0)
    cost2 = ev_safety_cost_lateral(ev2, rv2, cfg)
    check(cost2 >= 0.0, "nearby lateral → valid cost")


def test_ev_comfort_cost():
    print("\n[utility_functions] ev_comfort_cost")
    c = ev_comfort_cost(a_current=2.0, a_previous=0.0)
    check_approx(c, 8.0, "a=2 da=2 → 4+4=8")

    c = ev_comfort_cost(a_current=0.0, a_previous=0.0)
    check_approx(c, 0.0, "a=0 da=0 → 0")


def test_ev_efficiency_cost():
    print("\n[utility_functions] ev_efficiency_cost")
    c = ev_efficiency_cost(v_current=25.0, v_desire=25.0)
    check_approx(c, 0.0, "at desired speed → 0")

    c = ev_efficiency_cost(v_current=20.0, v_desire=25.0)
    check_approx(c, 25.0, "5 m/s off → 25")


def test_compute_ev_cost():
    print("\n[utility_functions] compute_ev_cost")
    cfg = GameConfig()
    ev_traj = [TrajectoryPoint(t=0, x=0, y=0, v=25, a=1),
               TrajectoryPoint(t=2, x=52, y=0, v=27, a=0)]
    fv = VehicleState(x=80.0, y=0.0, vx=20.0, vy=0.0)
    rv = VehicleState(x=-50.0, y=4.0, vx=15.0, vy=0.0)
    result = compute_ev_cost(ev_traj, fv, rv, cfg)
    check(isinstance(result.total, float), "returns float total")
    check(result.safe >= 0.0, "safety cost >= 0")
    check(result.eff >= 0.0, "efficiency cost >= 0")


def test_prediction_weighted_payoff():
    print("\n[utility_functions] compute_prediction_weighted_payoff")
    cfg = GameConfig()
    traj1 = [TrajectoryPoint(t=0, x=0, y=0, v=25, a=0)]
    traj2 = [TrajectoryPoint(t=0, x=5, y=0, v=26, a=0)]

    def dummy_payoff(traj):
        return traj[0].v

    result = compute_prediction_weighted_payoff([traj1, traj2], dummy_payoff, attenuation=0.85)
    # 0.85^0 * 25 + 0.85^1 * 26 = 25 + 22.1 = 47.1
    expected = 25.0 + 0.85 * 26.0
    check_approx(result, expected, f"weighted payoff: {expected:.1f}")


# ============================================================
# game_solver — pure functions
# ============================================================
def test_candidate_action_dataclass():
    print("\n[game_solver] CandidateAction dataclass")
    c = CandidateAction(lateral=-1, accel=2.0)
    check(c.label == "LEFT_+2.0", f"label: {c.label}")
    check(c.lateral == -1, "lateral=-1")
    check(c.accel == 2.0, "accel=2.0")


def test_game_result_dataclass():
    print("\n[game_solver] GameResult dataclass")
    r = GameResult(
        action=0, lateral_choice=-1, optimal_accel=1.5,
        ev_cost_original_lane=5.0, ev_cost_target_lane=4.0,
        cost_improvement=1.0, hv_driving_style="normal",
        hv_predicted_speed=20.0, min_ttc=5.0, min_gap=15.0,
        game_success=True, candidates_evaluated=18,
    )
    check(r.game_success, "game_success=True")
    check(r.cost_improvement > 0, "positive cost improvement = LC beneficial")


def test_generate_candidates():
    print("\n[game_solver] _generate_candidates")
    solver = StackelbergSolver(GameConfig())

    # Lane 0 (leftmost) — can STAY or go RIGHT
    c0 = solver._generate_candidates(current_lane_id=0, num_lanes=3)
    laterals_0 = set(c.lateral for c in c0)
    check(laterals_0 == {0, 1}, "lane 0: only STAY + RIGHT")

    # Lane 1 (middle) — can go LEFT, STAY, RIGHT
    c1 = solver._generate_candidates(current_lane_id=1, num_lanes=3)
    laterals_1 = set(c.lateral for c in c1)
    check(laterals_1 == {-1, 0, 1}, "lane 1: LEFT + STAY + RIGHT")

    # Lane 2 (rightmost) — can go LEFT or STAY
    c2 = solver._generate_candidates(current_lane_id=2, num_lanes=3)
    laterals_2 = set(c.lateral for c in c2)
    check(laterals_2 == {-1, 0}, "lane 2: only LEFT + STAY")

    # Each lateral has len(accel_candidates) entries
    accel_count = len(GameConfig().accel_candidates)
    check(len(c1) == 3 * accel_count, f"middle lane: 3x{accel_count}={3*accel_count} candidates")


def test_classify_driving_style():
    print("\n[game_solver] _classify_driving_style")
    solver = StackelbergSolver(GameConfig())

    # None → normal
    s = solver._classify_driving_style(None)
    check(s == "normal", "None RV → normal")

    # Steady speed → conservative
    rv = VehicleState(x=-30, y=0, vx=20.0, vy=0.0)
    for _ in range(15):
        s = solver._classify_driving_style(rv, rv_id=1)
    check(s == "conservative", "steady speed → conservative")

    # Highly variable speed → aggressive
    solver2 = StackelbergSolver(GameConfig())
    speeds = [20.0, 28.0, 15.0, 30.0, 18.0, 27.0, 14.0, 31.0, 17.0, 29.0,
              19.0, 26.0, 16.0, 32.0, 15.0]
    for v in speeds:
        rv2 = VehicleState(x=-30, y=0, vx=v, vy=0.0)
        s = solver2._classify_driving_style(rv2, rv_id=2)
    check(s == "aggressive", f"highly variable speed → aggressive (got: {s})")


# ============================================================
# FSM — RateLimiter
# ============================================================
def test_rate_limiter():
    print("\n[fsm_executor] RateLimiter")
    rl = RateLimiter(max_jerk=2.5)
    a1 = rl.smooth(1.0, dt=0.25)
    check_approx(a1, 1.0, "first call returns target")

    # Large jump should be limited
    a2 = rl.smooth(4.0, dt=0.25)
    max_delta = 2.5 * 0.25  # 0.625
    check(abs(a2 - a1) <= max_delta + 0.01, f"jump limited: delta={abs(a2-a1):.3f} <= {max_delta:.3f}")

    # Negative jump
    a3 = rl.smooth(-2.0, dt=0.25)
    check(abs(a3 - a2) <= max_delta + 0.01, f"negative jump limited: delta={abs(a3-a2):.3f}")

    # Reset
    rl.reset()
    a4 = rl.smooth(10.0, dt=0.25)
    check_approx(a4, 10.0, "after reset, returns target directly")


# ============================================================
# FSM — SafetyGate
# ============================================================
def test_safety_gate_predict_gap():
    print("\n[fsm_executor] SafetyGate.predict_safety_gap")
    gate = FSMSafetyGate(GameConfig())
    g = gate.predict_safety_gap(current_gap=30.0, relative_speed=-5.0,
                                prediction_horizon=2.0)
    # gap(2) = 30 + (-5)*2 = 20  (closing)
    check_approx(g, 20.0, "closing: 30 - 10 = 20")

    g = gate.predict_safety_gap(current_gap=20.0, relative_speed=3.0,
                                prediction_horizon=2.0)
    check_approx(g, 26.0, "opening: 20 + 6 = 26")


def test_safety_gate_predict_ttc():
    print("\n[fsm_executor] SafetyGate.predict_ttc")
    gate = FSMSafetyGate(GameConfig())
    ttc = gate.predict_ttc(gap=25.0, relative_speed=5.0)
    check_approx(ttc, 5.0, "gap=25 dv=5 → ttc=5")

    ttc = gate.predict_ttc(gap=25.0, relative_speed=0.0)
    check(math.isinf(ttc), "dv=0 → ttc=inf")

    ttc = gate.predict_ttc(gap=0.0, relative_speed=5.0)
    check_approx(ttc, 0.0, "gap=0 → ttc=0")

    ttc = gate.predict_ttc(gap=10.0, relative_speed=-3.0)
    check(math.isinf(ttc), "dv<0 (not closing) → ttc=inf")


def test_safety_gate_cost_improvement():
    print("\n[fsm_executor] SafetyGate.check_cost_improvement")
    gate = FSMSafetyGate(GameConfig())
    # threshold = 0.10
    check(gate.check_cost_improvement(lc_cost=4.0, keep_cost=5.0),
          "improvement 1.0 > 0.10 → True")
    check(not gate.check_cost_improvement(lc_cost=5.0, keep_cost=5.05),
          "improvement 0.05 < 0.10 → False")
    check(not gate.check_cost_improvement(lc_cost=6.0, keep_cost=5.0),
          "negative improvement → False")


def test_safety_gate_lane_change_feasibility():
    print("\n[fsm_executor] SafetyGate.check_lane_change_feasibility")
    cfg = GameConfig()
    gate = FSMSafetyGate(cfg)

    # Ample gap: 60m front gap, slight closing, fast ego
    ok, reason = gate.check_lane_change_feasibility(
        front_gap=60.0, front_rel_speed=2.0, ego_speed=25.0)
    check(ok, f"ample gap feasible (got: {reason})")

    # Too small gap
    ok, reason = gate.check_lane_change_feasibility(
        front_gap=10.0, front_rel_speed=2.0, ego_speed=25.0)
    check(not ok, f"small gap rejected (got: {reason})")

    # Too small rear gap
    ok, reason = gate.check_lane_change_feasibility(
        front_gap=60.0, front_rel_speed=2.0, ego_speed=25.0,
        rear_gap=5.0, rear_rel_speed=0.0)
    check(not ok, f"small rear gap rejected (got: {reason})")


def test_safety_gate_check_safe():
    print("\n[fsm_executor] SafetyGate.check_safe")
    cfg = GameConfig()
    gate = FSMSafetyGate(cfg)

    # Safe: large gaps, no closing
    ok, reason = gate.check_safe(front_gap=50.0, front_rel_speed=-3.0,
                                 rear_gap=40.0, rear_rel_speed=1.0)
    check(ok, f"safe scenario (got: {reason})")

    # Unsafe: small front gap, closing fast
    ok, reason = gate.check_safe(front_gap=5.0, front_rel_speed=10.0,
                                 rear_gap=40.0, rear_rel_speed=1.0)
    check(not ok, f"unsafe front gap (got: {reason})")


# ============================================================
# FSM — State transitions (pure logic, no env needed)
# ============================================================
def test_fsm_initial_state():
    print("\n[fsm_executor] FSM initial state")
    fsm = FSMExecutor(GameConfig())
    check(fsm.state == FSMState.LANE_KEEPING, "initial state = LANE_KEEPING")
    check(fsm.time_in_state == 0.0, "time_in_state = 0")
    check(fsm.cooldown_remaining == 0.0, "cooldown=0")


def test_fsm_reset():
    print("\n[fsm_executor] FSM reset")
    fsm = FSMExecutor(GameConfig())
    fsm._transition(FSMState.LC_PREPARATION)  # force state
    fsm._sim_cooldown_remaining = 5.0
    fsm.reset()
    check(fsm.state == FSMState.LANE_KEEPING, "reset → LANE_KEEPING")
    check(fsm.cooldown_remaining == 0.0, "reset clears cooldown")


def test_fsm_tick():
    print("\n[fsm_executor] FSM tick")
    fsm = FSMExecutor(GameConfig())
    fsm._tick(0.5)
    check(fsm.time_in_state == 0.5, "tick 0.5s → time=0.5")
    fsm._tick(0.25)
    check(fsm.time_in_state == 0.75, "tick 0.25s → time=0.75")


def test_fsm_transition_resets_timer():
    print("\n[fsm_executor] FSM transition resets timer")
    fsm = FSMExecutor(GameConfig())
    fsm._tick(1.0)
    check(fsm.time_in_state == 1.0, "time=1s in LANE_KEEPING")
    fsm._transition(FSMState.LC_PREPARATION)
    check(fsm.time_in_state == 0.0, "transition resets time to 0")
    check(fsm.state == FSMState.LC_PREPARATION, "now in LC_PREPARATION")


# ============================================================
# config — Driving style weights
# ============================================================
def test_driving_style_weights():
    print("\n[config] Driving style weights")
    check("aggressive" in DRIVING_STYLE_WEIGHTS, "aggressive style exists")
    check("normal" in DRIVING_STYLE_WEIGHTS, "normal style exists")
    check("conservative" in DRIVING_STYLE_WEIGHTS, "conservative style exists")

    agg = DRIVING_STYLE_WEIGHTS["aggressive"]
    con = DRIVING_STYLE_WEIGHTS["conservative"]
    # Conservative weights safety more heavily
    check(con["w_safe"] > agg["w_safe"], "conservative weights safety more than aggressive")
    # Aggressive weights efficiency more heavily
    check(agg["w_eff"] > con["w_eff"], "aggressive weights efficiency more than conservative")


def test_game_config_defaults():
    print("\n[config] GameConfig defaults")
    cfg = GameConfig()
    check(cfg.max_accel > 0, "max_accel > 0")
    check(cfg.max_brake > cfg.max_accel, "max_brake > max_accel")
    check(cfg.desired_speed > 0, "desired_speed > 0")
    check(cfg.min_safe_distance > 0, "min_safe_distance > 0")
    check(cfg.prediction_horizon > 0, "prediction_horizon > 0")
    check(cfg.lc_cooldown > 0, "lc_cooldown > 0")
    check(len(cfg.accel_candidates) > 0, "has accel candidates")
    check(cfg.cost_improvement_threshold > 0, "cost_improvement_threshold > 0")


# ============================================================
# Integration: solver cost ordering
# ============================================================
def test_solver_cost_ordering():
    """Verify that stay+decelerate costs less than stay+accelerate when
    close to a front vehicle (basic sanity check on cost functions)."""
    print("\n[integration] solver cost ordering")
    cfg = GameConfig()
    ev = VehicleState(x=0.0, y=0.0, vx=20.0, vy=0.0)
    fv = VehicleState(x=15.0, y=0.0, vx=15.0, vy=0.0)  # only 10m gap
    rv = VehicleState(x=-80.0, y=4.0, vx=18.0, vy=0.0)

    # Accelerate toward front vehicle → should be more expensive
    traj_accel = predict_ev_candidate(ev, longitudinal_accel=2.0, lateral_action=0,
                                      horizon=cfg.prediction_horizon, dt=cfg.dt)
    cost_accel = compute_ev_cost(traj_accel, fv, rv, cfg).total

    # Decelerate → should be less expensive
    traj_decel = predict_ev_candidate(ev, longitudinal_accel=-2.0, lateral_action=0,
                                      horizon=cfg.prediction_horizon, dt=cfg.dt)
    cost_decel = compute_ev_cost(traj_decel, fv, rv, cfg).total

    check(cost_decel < cost_accel,
          f"decel safer than accel near front vehicle (decel={cost_decel:.1f} < accel={cost_accel:.1f})")


# ============================================================
def main():
    global _passed, _failed
    _passed = 0
    _failed = 0

    print("=" * 60)
    print("Stackelberg Module — Unit Tests")
    print("=" * 60)

    tests = [
        # trajectory_predictor
        test_estimate_decay_rate,
        test_predict_trajectory_basic,
        test_predict_trajectory_deceleration,
        test_predict_trajectory_speed_clamp,
        test_predict_hv_response,
        test_predict_ev_candidate_lane_change,
        test_compute_ttc_gap,
        test_compute_ttc_gap_collision,
        # utility_functions — HV
        test_hv_safety_payoff,
        test_hv_space_payoff,
        test_hv_comfort_payoff,
        test_hv_efficiency_payoff,
        test_compute_hv_utility,
        # utility_functions — EV
        test_ev_safety_cost_longitudinal_collision,
        test_ev_safety_cost_longitudinal_normal,
        test_ev_safety_cost_longitudinal_ttc_penalty,
        test_ev_safety_cost_lateral,
        test_ev_comfort_cost,
        test_ev_efficiency_cost,
        test_compute_ev_cost,
        test_prediction_weighted_payoff,
        # game_solver
        test_candidate_action_dataclass,
        test_game_result_dataclass,
        test_generate_candidates,
        test_classify_driving_style,
        # fsm_executor
        test_rate_limiter,
        test_safety_gate_predict_gap,
        test_safety_gate_predict_ttc,
        test_safety_gate_cost_improvement,
        test_safety_gate_lane_change_feasibility,
        test_safety_gate_check_safe,
        test_fsm_initial_state,
        test_fsm_reset,
        test_fsm_tick,
        test_fsm_transition_resets_timer,
        # config
        test_driving_style_weights,
        test_game_config_defaults,
        # integration
        test_solver_cost_ordering,
    ]

    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            _failed += 1
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 60)
    print(f"Results: {_passed} passed, {_failed} failed, {len(tests)} total")
    if _failed == 0:
        print("ALL UNIT TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
