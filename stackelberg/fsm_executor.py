"""Four-state Finite State Machine executor for lane-change governance.

Implements tech roadmap Section 2.1.2: "执行层：基于安全门控FSM的指令治理与平滑"

States:
    LANE_KEEPING   → default, follow lane
    LC_PREPARATION → game proposed lane change, safety checks in progress
    LC_EXECUTION   → lateral command locked, executing lane change
    STATE_RECOVERY  → post-lane-change or abort, cooling down

Key mechanisms:
    - Safety gating (formulas 4-5, 4-6): predicted TTC and gap check
    - Single-pulse trigger (formula 4-7): only trigger when cost improvement > threshold
    - Cooling delay: prevent rapid successive lane changes
    - Rate limiting: smooth acceleration commands via jerk limiting
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np

from scene_utils import check_lane_exists, get_lane_longitudinal

from .config import GameConfig

# highway-env action constants (matching rss_safety.py)
ACTION_LEFT = 0
ACTION_IDLE = 1
ACTION_RIGHT = 2
ACTION_FASTER = 3
ACTION_SLOWER = 4


class FSMState(Enum):
    LANE_KEEPING = 0
    LC_PREPARATION = 1
    LC_EXECUTION = 2
    STATE_RECOVERY = 3


@dataclass
class FSMInfo:
    state: str
    state_id: int
    action: int
    intervened: bool
    reason: str
    time_in_state: float
    cooldown_remaining: float
    game_action_overridden: bool


class RateLimiter:
    """Acceleration rate limiter — prevents jerk from exceeding max_jerk."""

    def __init__(self, max_jerk: float = 2.5):
        self.max_jerk = max_jerk
        self._prev_accel: Optional[float] = None

    def smooth(self, target_accel: float, dt: float) -> float:
        """Limit acceleration change rate.

        Args:
            target_accel: Desired acceleration.
            dt: Time step since last call.

        Returns:
            Smoothed acceleration respecting jerk limit.
        """
        if self._prev_accel is None:
            self._prev_accel = target_accel
            return target_accel

        max_delta = self.max_jerk * max(dt, 0.01)
        delta = target_accel - self._prev_accel
        clamped_delta = float(np.clip(delta, -max_delta, max_delta))
        result = self._prev_accel + clamped_delta
        self._prev_accel = result
        return result

    def reset(self) -> None:
        self._prev_accel = None


class FSMSafetyGate:
    """Safety gating module implementing formulas (4-5) and (4-6)."""

    def __init__(self, config: GameConfig):
        self.config = config

    def predict_safety_gap(
        self,
        current_gap: float,
        relative_speed: float,
        prediction_horizon: float = 2.0,
    ) -> float:
        """Predict safety gap at future time τ — formula (4-5).

        gap(τ) = gap(0) + v_rel * τ

        Uses linear extrapolation as specified in the tech roadmap.
        """
        return current_gap + relative_speed * prediction_horizon

    def predict_ttc(self, gap: float, relative_speed: float) -> float:
        """Compute predicted TTC — formula (4-6).

        TTC = gap / v_rel  (when v_rel > 0, i.e., ego faster than front)

        Returns inf if not a closing situation.
        """
        if gap <= 0.0:
            return 0.0
        if relative_speed > 1e-6:
            return gap / relative_speed
        return float('inf')

    def check_lane_change_feasibility(
        self,
        front_gap: float,
        front_rel_speed: float,
        ego_speed: float,
        rear_gap: float = float('inf'),
        rear_rel_speed: float = 0.0,
    ) -> Tuple[bool, str]:
        """Check if a lane change can be safely completed before the gap closes.

        Uses a conservative lc_time of 5.0s to match highway-env's actual
        lane-change duration (20+ steps at 4Hz).  The post-LC TTC must be
        >= 8s so that even after a 5s lane change on a closing trajectory,
        the gradual SLOWER action can still decelerate before collision.
        """
        cfg = self.config
        lc_time = 5.0  # highway-env actual lane-change time (conservative)

        gap_after_lc = front_gap - front_rel_speed * lc_time

        # Phase 1: Post-LC TTC — dominant safety criterion
        if front_rel_speed > 1e-6 and gap_after_lc > 0.0:
            post_lc_ttc = gap_after_lc / front_rel_speed
            min_post_lc_ttc = 8.0
            if post_lc_ttc < min_post_lc_ttc:
                return False, (
                    f"lc_ttc_low: post-LC TTC={post_lc_ttc:.1f}s < {min_post_lc_ttc}s "
                    f"(gap_after={gap_after_lc:.1f}m, rel={front_rel_speed:.1f}m/s)"
                )

        # Phase 2: Absolute post-LC gap
        safety_margin = cfg.min_safe_distance * 4.0  # 20m
        if gap_after_lc < safety_margin:
            return False, f"lc_gap_low: after LC gap={gap_after_lc:.1f}m < {safety_margin:.1f}m"

        # Minimum initial gap
        min_lc_gap = cfg.min_safe_distance * 4.0  # 20m
        if front_gap < min_lc_gap:
            return False, f"lc_too_close: gap={front_gap:.1f}m < {min_lc_gap:.1f}m"

        # Minimum rear gap: must have enough space behind in target lane
        min_lc_rear_gap = cfg.min_safe_distance * 3.0  # 15m
        if rear_gap < min_lc_rear_gap:
            return False, f"lc_rear_too_close: rear_gap={rear_gap:.1f}m < {min_lc_rear_gap:.1f}m"

        return True, "lc_feasible"

    def check_safe(
        self,
        front_gap: float,
        front_rel_speed: float,
        rear_gap: float,
        rear_rel_speed: float,
    ) -> Tuple[bool, str]:
        """Run safety gating checks.

        Returns (is_safe, reason).
        """
        cfg = self.config
        predict_horizon = cfg.fsm_safety_horizon
        safe_gap = cfg.min_safe_distance * cfg.gap_safety_margin  # add margin

        # Front safety check
        pred_front_gap = self.predict_safety_gap(front_gap, -front_rel_speed, predict_horizon)
        front_ttc = self.predict_ttc(front_gap, front_rel_speed)

        if pred_front_gap < safe_gap:
            return False, f"front_gap_insufficient: {pred_front_gap:.1f}m < {safe_gap:.1f}m"
        if front_ttc < cfg.ttc_safe_threshold:
            return False, f"front_ttc_critical: {front_ttc:.1f}s < {cfg.ttc_safe_threshold}s"

        # Rear safety check (for lane change)
        if rear_gap < float('inf'):
            pred_rear_gap = self.predict_safety_gap(rear_gap, rear_rel_speed, predict_horizon)
            rear_ttc = self.predict_ttc(rear_gap, rear_rel_speed)
            if pred_rear_gap < safe_gap:
                return False, f"rear_gap_insufficient: {pred_rear_gap:.1f}m < {safe_gap:.1f}m"
            if rear_ttc < cfg.ttc_safe_threshold:
                return False, f"rear_ttc_critical: {rear_ttc:.1f}s < {cfg.ttc_safe_threshold}s"

        return True, "safe"

    def check_cost_improvement(
        self,
        lc_cost: float,
        keep_cost: float,
    ) -> bool:
        """Check if cost improvement justifies lane change — formula (4-7).

        C_keep - C_lc > ε  →  lane change is significantly beneficial.

        This is the single-pulse trigger condition.
        """
        return (keep_cost - lc_cost) > self.config.cost_improvement_threshold


class FSMExecutor:
    """Four-state FSM for lane-change governance and command smoothing.

    Pipeline:
        Game proposal → Safety Gate → FSM transition → Action output
    """

    def __init__(self, config: GameConfig):
        self.config = config
        self._state = FSMState.LANE_KEEPING
        self._sim_time_in_state: float = 0.0
        self._sim_cooldown_remaining: float = 0.0
        self._game_action = ACTION_IDLE
        self._accel_target = 0.0
        self._rate_limiter = RateLimiter(max_jerk=config.max_jerk)
        self._safety_gate = FSMSafetyGate(config)
        self._lc_lateral: int = 0
        self._lc_target_gap_at_start: float = float('inf')
        self._lc_origin_lane: int = 0  # ego lane when LC was approved
        self._lc_action_sent: bool = False  # lane-change action already issued

    @property
    def state(self) -> FSMState:
        return self._state

    @property
    def time_in_state(self) -> float:
        return self._sim_time_in_state

    @property
    def cooldown_remaining(self) -> float:
        return max(0.0, self._sim_cooldown_remaining)

    def _transition(self, new_state: FSMState) -> None:
        if new_state != self._state:
            self._state = new_state
            self._sim_time_in_state = 0.0

    def _tick(self, dt: float) -> None:
        """Advance FSM simulation time by dt seconds."""
        self._sim_time_in_state += dt
        self._sim_cooldown_remaining = max(0.0, self._sim_cooldown_remaining - dt)

    def _get_gaps_from_env(self, env, lane_offset: int = 0) -> Tuple[float, float, float, float]:
        """Extract (front_gap, front_rel_speed, rear_gap, rear_rel_speed) from env.

        Args:
            env: highway-env environment.
            lane_offset: 0 = current lane, -1 = left lane, +1 = right lane.
        """
        ego = env.unwrapped.vehicle
        road = env.unwrapped.road
        ego_lane = ego.lane_index

        target_lane = (ego_lane[0], ego_lane[1], ego_lane[2] + lane_offset)
        if not self._lane_exists(road, target_lane):
            return (float('inf'), 0.0, float('inf'), 0.0)

        try:
            lane = road.network.get_lane(target_lane)
            ego_s, _ = lane.local_coordinates(ego.position)
            ego_s = float(ego_s)
        except (ValueError, IndexError):
            return (float('inf'), 0.0, float('inf'), 0.0)

        front, rear = road.neighbour_vehicles(ego, target_lane)

        front_gap = float('inf')
        front_rel_speed = 0.0
        if front is not None:
            try:
                front_s, _ = lane.local_coordinates(front.position)
                front_gap = float(front_s) - ego_s
                front_rel_speed = float(ego.speed - front.speed)
            except (ValueError, IndexError):
                pass

        rear_gap = float('inf')
        rear_rel_speed = 0.0
        if rear is not None:
            try:
                rear_s, _ = lane.local_coordinates(rear.position)
                rear_gap = ego_s - float(rear_s)
                rear_rel_speed = float(rear.speed - ego.speed)
            except (ValueError, IndexError):
                pass

        return front_gap, front_rel_speed, rear_gap, rear_rel_speed

    def _lane_exists(self, road, lane_index: Tuple) -> bool:
        return check_lane_exists(road, lane_index)

    def _get_min_front_ttc(self, env) -> float:
        """Find minimum TTC to front vehicles in the ego's lane or adjacent lanes.

        Filters by lateral proximity so vehicles in distant lanes are not
        treated as threats. Uses the road network to find true lane-relative
        front vehicles.
        """
        ego = env.unwrapped.vehicle
        road = env.unwrapped.road
        ego_lane = ego.lane_index
        ego_speed = float(ego.speed)
        vehicle_length = self.config.vehicle_length
        min_ttc = float('inf')

        # Check current lane and adjacent lanes
        graph = road.network.graph
        start, end = ego_lane[0], ego_lane[1]
        if start not in graph or end not in graph.get(start, {}):
            return min_ttc

        lanes = graph[start][end]
        current_lane_idx = ego_lane[2]
        lane_indices_to_check = [current_lane_idx]
        if current_lane_idx > 0:
            lane_indices_to_check.append(current_lane_idx - 1)
        if current_lane_idx < len(lanes) - 1:
            lane_indices_to_check.append(current_lane_idx + 1)

        for lane_idx in lane_indices_to_check:
            try:
                lane = lanes[lane_idx]
                ego_s, _ = lane.local_coordinates(ego.position)
                ego_s = float(ego_s)
            except (ValueError, IndexError):
                continue

            front, _ = road.neighbour_vehicles(ego, (start, end, lane_idx))
            if front is not None:
                try:
                    front_s, _ = lane.local_coordinates(front.position)
                    gap = float(front_s) - ego_s - vehicle_length
                    rel_speed = ego_speed - float(front.speed)
                    if gap <= 0:
                        return 0.0
                    if rel_speed > 1e-6:
                        ttc = gap / rel_speed
                        if ttc < min_ttc:
                            min_ttc = ttc
                except (ValueError, IndexError):
                    continue

        return min_ttc

    def _get_min_front_gap(self, env) -> float:
        """Find minimum gap to front vehicles in ego's lane or adjacent lanes."""
        ego = env.unwrapped.vehicle
        road = env.unwrapped.road
        ego_lane = ego.lane_index
        vehicle_length = self.config.vehicle_length
        min_gap = float('inf')

        graph = road.network.graph
        start, end = ego_lane[0], ego_lane[1]
        if start not in graph or end not in graph.get(start, {}):
            return min_gap

        lanes = graph[start][end]
        current_lane_idx = ego_lane[2]
        lane_indices_to_check = [current_lane_idx]
        if current_lane_idx > 0:
            lane_indices_to_check.append(current_lane_idx - 1)
        if current_lane_idx < len(lanes) - 1:
            lane_indices_to_check.append(current_lane_idx + 1)

        for lane_idx in lane_indices_to_check:
            try:
                lane = lanes[lane_idx]
                ego_s, _ = lane.local_coordinates(ego.position)
                ego_s = float(ego_s)
            except (ValueError, IndexError):
                continue

            front, _ = road.neighbour_vehicles(ego, (start, end, lane_idx))
            if front is not None:
                try:
                    front_s, _ = lane.local_coordinates(front.position)
                    gap = float(front_s) - ego_s - vehicle_length
                    if gap < min_gap:
                        min_gap = gap
                except (ValueError, IndexError):
                    continue

        return min_gap

    def _assess_rear_danger(self, env) -> Tuple[float, float, bool]:
        """Check rear vehicle safety in the ego's current lane.

        Returns (rear_ttc, rear_gap, is_dangerous).
        """
        _, _, rear_gap, rear_rel_speed = self._get_gaps_from_env(env)
        rear_ttc = self._safety_gate.predict_ttc(rear_gap, -rear_rel_speed)
        cfg = self.config
        is_dangerous = (
            rear_ttc < cfg.rear_ttc_warning
            or rear_gap < cfg.rear_gap_warning
        )
        return rear_ttc, rear_gap, is_dangerous

    def _should_brake(
        self,
        game_accel: float,
        ego_speed: float,
        rear_ttc: float,
        rear_gap: float,
        front_ttc: float,
        dt: float,
    ) -> Tuple[int, str]:
        """Decide braking action with rear-vehicle awareness and speed floor.

        Graduated braking:
          - If speed <= min_cruise_speed and front TTC > 4s: no braking (IDLE)
          - If rear vehicle is critically close (TTC < rear_ttc_critical): no braking
          - If rear vehicle is close (TTC < rear_ttc_warning): gentle braking
          - Otherwise: normal braking (trust game solver)
        """
        cfg = self.config

        # Speed floor: accelerate back to cruising speed when safe
        if ego_speed <= cfg.min_cruise_speed and front_ttc > 4.0:
            smoothed = self._rate_limiter.smooth(cfg.max_accel * 0.5, dt)
            return ACTION_FASTER, "speed_floor_accel"

        # Rear danger: don't brake if rear vehicle would hit us
        if rear_ttc < cfg.rear_ttc_critical:
            return ACTION_IDLE, f"rear_critical: rear_ttc={rear_ttc:.1f}s"

        # Rear warning: accelerate gently to create gap, or hold if unsafe
        if rear_ttc < cfg.rear_ttc_warning or rear_gap < cfg.rear_gap_warning:
            if ego_speed > cfg.min_cruise_speed:
                return ACTION_SLOWER, f"rear_aware_brake: rear_ttc={rear_ttc:.1f}s"
            else:
                # Accelerate to create rear gap
                smoothed = self._rate_limiter.smooth(cfg.max_accel * 0.5, dt)
                return ACTION_FASTER, f"rear_aware_accel: rear_ttc={rear_ttc:.1f}s"

        # Normal: game solver decides
        if game_accel < -0.5:
            if ego_speed > cfg.min_cruise_speed:
                return ACTION_SLOWER, "game_brake"
            else:
                # At speed floor but game wants brake — try to escape via acceleration
                smoothed = self._rate_limiter.smooth(cfg.max_accel * 0.5, dt)
                return ACTION_FASTER, "game_speed_floor_accel"
        elif game_accel > 0.5:
            smoothed = self._rate_limiter.smooth(min(game_accel, cfg.max_accel), dt)
            return (ACTION_FASTER if smoothed > 0.2 else ACTION_IDLE), "game_accel"
        else:
            return ACTION_IDLE, "game_idle"

    def process(
        self,
        game_result,
        env,
        dt: float = 0.25,
    ) -> Tuple[int, FSMInfo]:
        """Process game result through FSM to produce final action.

        Args:
            game_result: GameResult from StackelbergSolver.solve().
            env: highway-env environment.
            dt: Time step (policy_frequency = 4Hz → 0.25s default).

        Returns:
            (final_action, fsm_info)
        """
        self._tick(dt)

        intervened = False
        reason = ""
        game_action_overridden = False

        ego_lane_id = env.unwrapped.vehicle.lane_index[2] if len(env.unwrapped.vehicle.lane_index) > 2 else 0

        front_gap, front_rel_speed, rear_gap, rear_rel_speed = self._get_gaps_from_env(env)
        is_safe, safety_reason = self._safety_gate.check_safe(
            front_gap, front_rel_speed, rear_gap, rear_rel_speed,
        )

        ego_speed = float(env.unwrapped.vehicle.speed)
        front_ttc_cur = self._safety_gate.predict_ttc(front_gap, front_rel_speed)
        rear_ttc, rear_gap, rear_dangerous = self._assess_rear_danger(env)

        lc_feasible = True
        lc_feasible_reason = ""
        target_lane_safe = True
        target_lane_reason = ""
        if game_result.lateral_choice != 0:
            # Check feasibility and safety using TARGET lane gaps, not current lane.
            tgt_gap, tgt_rel, tgt_rear_gap, tgt_rear_rel = self._get_gaps_from_env(
                env, lane_offset=game_result.lateral_choice,
            )
            lc_feasible, lc_feasible_reason = self._safety_gate.check_lane_change_feasibility(
                tgt_gap, tgt_rel, ego_speed, tgt_rear_gap, tgt_rear_rel,
            )
            target_lane_safe, target_lane_reason = self._safety_gate.check_safe(
                tgt_gap, tgt_rel, tgt_rear_gap, tgt_rear_rel,
            )

        # ---- Hard emergency brake (RSS-style shield, runs every step) ----
        # During LC states, check the target lane ONLY if the lane change
        # hasn't completed yet.  _lc_origin_lane stores the ego's lane when
        # the LC was approved; compare against current lane to detect completion.
        emergency_gap = front_gap
        emergency_rel = front_rel_speed
        if self._state in (FSMState.LC_PREPARATION, FSMState.LC_EXECUTION):
            intended_target = self._lc_origin_lane + self._lc_lateral
            if ego_lane_id != intended_target:  # lane change still in progress
                tgt_gap, tgt_rel, _, _ = self._get_gaps_from_env(env, lane_offset=self._lc_lateral)
                if tgt_gap < emergency_gap:
                    emergency_gap = tgt_gap
                    emergency_rel = tgt_rel

        front_ttc_emer = self._safety_gate.predict_ttc(emergency_gap, emergency_rel)
        # Rear-aware emergency brake: if rear vehicle is critically close,
        # braking would cause a rear-end collision — prefer lane-change escape.
        if front_ttc_emer < 3.0 or emergency_gap < 8.0:
            # True emergency: must brake regardless of rear
            self._transition(FSMState.STATE_RECOVERY)
            self._sim_cooldown_remaining = self.config.lc_cooldown
            return ACTION_SLOWER, FSMInfo(
                state=self._state.name,
                state_id=self._state.value,
                action=ACTION_SLOWER,
                intervened=True,
                reason=f"emergency_brake: ttc={front_ttc_emer:.1f}s gap={emergency_gap:.1f}m",
                time_in_state=self.time_in_state,
                cooldown_remaining=self.cooldown_remaining,
                game_action_overridden=True,
            )
        if front_ttc_emer < 8.0 or emergency_gap < 20.0:
            # Elevated risk: brake only if rear is safe
            if rear_ttc < self.config.rear_ttc_critical:
                # Rear danger — suppress brake, try lane-change
                self._transition(FSMState.STATE_RECOVERY)
                self._sim_cooldown_remaining = self.config.lc_cooldown
                return ACTION_IDLE, FSMInfo(
                    state=self._state.name,
                    state_id=self._state.value,
                    action=ACTION_IDLE,
                    intervened=True,
                    reason=f"brake_suppressed_rear: front_ttc={front_ttc_emer:.1f}s rear_ttc={rear_ttc:.1f}s",
                    time_in_state=self.time_in_state,
                    cooldown_remaining=self.cooldown_remaining,
                    game_action_overridden=True,
                )
            else:
                self._transition(FSMState.STATE_RECOVERY)
                self._sim_cooldown_remaining = self.config.lc_cooldown
                return ACTION_SLOWER, FSMInfo(
                    state=self._state.name,
                    state_id=self._state.value,
                    action=ACTION_SLOWER,
                    intervened=True,
                    reason=f"caution_brake: ttc={front_ttc_emer:.1f}s gap={emergency_gap:.1f}m",
                    time_in_state=self.time_in_state,
                    cooldown_remaining=self.cooldown_remaining,
                    game_action_overridden=True,
                )

        proposed_action = game_result.action
        cost_improves = self._safety_gate.check_cost_improvement(
            game_result.ev_cost_target_lane,
            game_result.ev_cost_original_lane,
        )

        if self._state == FSMState.LANE_KEEPING:
            # Speed gate: lane changes are physically impossible below 3 m/s
            speed_too_low_for_lc = ego_speed < 3.0

            if (game_result.lateral_choice != 0
                    and is_safe
                    and target_lane_safe
                    and lc_feasible
                    and cost_improves
                    and self.cooldown_remaining <= 0.0
                    and game_result.game_success
                    and not speed_too_low_for_lc):
                self._transition(FSMState.LC_PREPARATION)
                self._game_action = proposed_action
                self._lc_lateral = game_result.lateral_choice
                self._lc_origin_lane = ego_lane_id
                self._accel_target = game_result.optimal_accel
                self._lc_target_gap_at_start = tgt_gap
                self._lc_action_sent = False
                reason = "lc_proposed"
            else:
                # Lane change blocked — determine fallback action
                if proposed_action in (ACTION_LEFT, ACTION_RIGHT):
                    game_action_overridden = True

                # Low-speed recovery: force acceleration to regain maneuverability
                if ego_speed < 5.0 and front_ttc_cur > 4.0:
                    smoothed = self._rate_limiter.smooth(self.config.max_accel, dt)
                    return ACTION_FASTER, FSMInfo(
                        state=self._state.name,
                        state_id=self._state.value,
                        action=ACTION_FASTER,
                        intervened=True,
                        reason=f"low_speed_recovery: speed={ego_speed:.1f}m/s",
                        time_in_state=self.time_in_state,
                        cooldown_remaining=self.cooldown_remaining,
                        game_action_overridden=True,
                    )

                if not is_safe:
                    # Current lane unsafe — rear-aware braking
                    final_action, brake_reason = self._should_brake(
                        game_result.optimal_accel, ego_speed,
                        rear_ttc, rear_gap, front_ttc_cur, dt,
                    )
                    reason = f"safety_brake: {safety_reason} | {brake_reason}"
                    intervened = True
                elif (game_result.lateral_choice != 0
                      and game_result.ev_cost_original_lane > 50.0):
                    # Current lane is safe, cost is high → trust game solver longitudinal
                    final_action, _ = self._should_brake(
                        game_result.optimal_accel, ego_speed,
                        rear_ttc, rear_gap, front_ttc_cur, dt,
                    )
                    reason = f"danger_stay: cost_orig={game_result.ev_cost_original_lane:.0f}"
                    game_action_overridden = True
                elif not target_lane_safe and game_result.lateral_choice != 0:
                    final_action, _ = self._should_brake(
                        game_result.optimal_accel, ego_speed,
                        rear_ttc, rear_gap, front_ttc_cur, dt,
                    )
                    reason = f"target_lane_unsafe: {target_lane_reason}"
                    game_action_overridden = True
                elif not lc_feasible and game_result.lateral_choice != 0:
                    final_action, _ = self._should_brake(
                        game_result.optimal_accel, ego_speed,
                        rear_ttc, rear_gap, front_ttc_cur, dt,
                    )
                    reason = f"lc_not_feasible: {lc_feasible_reason}"
                    game_action_overridden = True
                elif not cost_improves and game_result.lateral_choice != 0:
                    final_action, _ = self._should_brake(
                        game_result.optimal_accel, ego_speed,
                        rear_ttc, rear_gap, front_ttc_cur, dt,
                    )
                    reason = "cost_insufficient_improvement"
                else:
                    final_action, _ = self._should_brake(
                        game_result.optimal_accel, ego_speed,
                        rear_ttc, rear_gap, front_ttc_cur, dt,
                    )
                    reason = "lane_keeping"
                return final_action, FSMInfo(
                    state=self._state.name,
                    state_id=self._state.value,
                    action=final_action,
                    intervened=intervened,
                    reason=reason,
                    time_in_state=self.time_in_state,
                    cooldown_remaining=self.cooldown_remaining,
                    game_action_overridden=game_action_overridden,
                )

        elif self._state == FSMState.LC_PREPARATION:
            # Abort if speed too low for lane change
            if ego_speed < 2.0:
                self._transition(FSMState.LANE_KEEPING)
                return ACTION_FASTER, FSMInfo(
                    state=self._state.name,
                    state_id=self._state.value,
                    action=ACTION_FASTER,
                    intervened=True,
                    reason=f"aborted_low_speed: speed={ego_speed:.1f}m/s",
                    time_in_state=self.time_in_state,
                    cooldown_remaining=self.cooldown_remaining,
                    game_action_overridden=True,
                )

            if not is_safe:
                # Safety check failed — abort, choose recovery action based on speed
                self._transition(FSMState.LANE_KEEPING)
                recovery_action = ACTION_FASTER if ego_speed < 10.0 else ACTION_SLOWER
                reason = f"aborted_safety: {safety_reason}"
                intervened = True
                return recovery_action, FSMInfo(
                    state=self._state.name,
                    state_id=self._state.value,
                    action=recovery_action,
                    intervened=intervened,
                    reason=reason,
                    time_in_state=self.time_in_state,
                    cooldown_remaining=self.cooldown_remaining,
                    game_action_overridden=True,
                )

            if self.time_in_state > self.config.lc_prep_timeout:
                # Timeout — abort, release to game solver
                self._transition(FSMState.LANE_KEEPING)
                reason = "prep_timeout"
                return ACTION_IDLE, FSMInfo(
                    state=self._state.name,
                    state_id=self._state.value,
                    action=ACTION_IDLE,
                    intervened=True,
                    reason=reason,
                    time_in_state=self.time_in_state,
                    cooldown_remaining=self.cooldown_remaining,
                    game_action_overridden=True,
                )

            # Confirm and execute
            if is_safe and cost_improves:
                self._transition(FSMState.LC_EXECUTION)
                reason = "lc_executing"
            else:
                # Still waiting for safe conditions
                return ACTION_IDLE, FSMInfo(
                    state=self._state.name,
                    state_id=self._state.value,
                    action=ACTION_IDLE,
                    intervened=False,
                    reason="waiting_safe_conditions",
                    time_in_state=self.time_in_state,
                    cooldown_remaining=self.cooldown_remaining,
                    game_action_overridden=False,
                )

        if self._state == FSMState.LC_EXECUTION:
            # Abort lane change if speed drops to near-zero (physically impossible)
            if ego_speed < 1.0:
                self._transition(FSMState.STATE_RECOVERY)
                self._sim_cooldown_remaining = self.config.lc_cooldown
                return ACTION_FASTER, FSMInfo(
                    state=self._state.name,
                    state_id=self._state.value,
                    action=ACTION_FASTER,
                    intervened=True,
                    reason=f"aborted_stuck: speed={ego_speed:.1f}m/s",
                    time_in_state=self.time_in_state,
                    cooldown_remaining=self.cooldown_remaining,
                    game_action_overridden=True,
                )

            intended_target = self._lc_origin_lane + self._lc_lateral

            # Lane change completed — ego reached the intended target lane
            if ego_lane_id == intended_target:
                self._transition(FSMState.STATE_RECOVERY)
                self._sim_cooldown_remaining = self.config.lc_cooldown
                reason = "lc_completed"
                return ACTION_IDLE, FSMInfo(
                    state=self._state.name,
                    state_id=self._state.value,
                    action=ACTION_IDLE,
                    intervened=True,
                    reason=reason,
                    time_in_state=self.time_in_state,
                    cooldown_remaining=self.cooldown_remaining,
                    game_action_overridden=True,
                )

            # Check target-lane TTC and gap stability — only while
            # the lane change is still in progress.
            tgt_ttc_caution = float('inf')
            tgt_gap = float('inf')
            gap_unstable = False
            if self._lc_lateral != 0:
                tgt_gap, tgt_rel, _, _ = self._get_gaps_from_env(env, lane_offset=self._lc_lateral)
                tgt_ttc_caution = self._safety_gate.predict_ttc(tgt_gap, tgt_rel)
                gap_drop_ratio = (self._lc_target_gap_at_start - tgt_gap) / max(self._lc_target_gap_at_start, 1.0)
                gap_unstable = gap_drop_ratio > 0.4

            if not is_safe or tgt_ttc_caution < 8.0 or gap_unstable:
                self._transition(FSMState.STATE_RECOVERY)
                self._sim_cooldown_remaining = self.config.lc_cooldown
                if not is_safe:
                    reason = f"emergency_abort: {safety_reason}"
                elif gap_unstable:
                    reason = (f"gap_unstable: tgt_gap {self._lc_target_gap_at_start:.0f}m"
                              f"→{tgt_gap:.0f}m (drop {gap_drop_ratio:.0%})")
                else:
                    reason = f"caution_abort: target TTC={tgt_ttc_caution:.1f}s < 8.0s"
                intervened = True
                return ACTION_SLOWER, FSMInfo(
                    state=self._state.name,
                    state_id=self._state.value,
                    action=ACTION_SLOWER,
                    intervened=intervened,
                    reason=reason,
                    time_in_state=self.time_in_state,
                    cooldown_remaining=self.cooldown_remaining,
                    game_action_overridden=True,
                )

            if self.time_in_state > self.config.lc_exec_timeout:
                self._transition(FSMState.STATE_RECOVERY)
                self._sim_cooldown_remaining = self.config.lc_cooldown
                reason = "lc_timeout"
                intervened = True
                return ACTION_IDLE, FSMInfo(
                    state=self._state.name,
                    state_id=self._state.value,
                    action=ACTION_IDLE,
                    intervened=intervened,
                    reason=reason,
                    time_in_state=self.time_in_state,
                    cooldown_remaining=self.cooldown_remaining,
                    game_action_overridden=True,
                )

            # Issue lane-change action ONCE — highway-env's DiscreteMetaAction
            # cumulatively increments target_lane_index, so multiple LEFT/RIGHT
            # actions cause over-shoot (lane 0→1→2 instead of 0→1).
            if not self._lc_action_sent:
                self._lc_action_sent = True
                final_action = self._game_action
                reason = "lc_started"
            else:
                final_action = ACTION_IDLE
                reason = "lc_in_progress"

            return final_action, FSMInfo(
                state=self._state.name,
                state_id=self._state.value,
                action=final_action,
                intervened=False,
                reason=reason,
                time_in_state=self.time_in_state,
                cooldown_remaining=self.cooldown_remaining,
                game_action_overridden=False,
            )

        elif self._state == FSMState.STATE_RECOVERY:
            if self.cooldown_remaining <= 0.0:
                self._transition(FSMState.LANE_KEEPING)
                reason = "cooldown_complete"
            else:
                reason = "cooling_down"

            # IDLE during recovery — cooldown prevents lane changes,
            # the game solver resumes control on next LANE_KEEPING step.
            intervened = True
            return ACTION_IDLE, FSMInfo(
                state=self._state.name,
                state_id=self._state.value,
                action=ACTION_IDLE,
                intervened=intervened,
                reason=f"recovery_hold: {reason}",
                time_in_state=self.time_in_state,
                cooldown_remaining=self.cooldown_remaining,
                game_action_overridden=True,
            )

        # Fallback
        return ACTION_IDLE, FSMInfo(
            state=self._state.name,
            state_id=self._state.value,
            action=ACTION_IDLE,
            intervened=False,
            reason="fallback",
            time_in_state=self.time_in_state,
            cooldown_remaining=self.cooldown_remaining,
            game_action_overridden=False,
        )


    def reset(self) -> None:
        self._state = FSMState.LANE_KEEPING
        self._sim_time_in_state = 0.0
        self._sim_cooldown_remaining = 0.0
        self._rate_limiter.reset()
        self._lc_target_gap_at_start = float('inf')
        self._lc_origin_lane = 0
        self._lc_action_sent = False
