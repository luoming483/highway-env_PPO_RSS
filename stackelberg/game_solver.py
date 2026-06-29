"""Stackelberg game equilibrium solver for lane-change decision-making.

Implements the leader-follower game from:
    Shi B, Zhai L, Liu C. "Stackelberg Game Based on Trajectory Prediction
    for Lane Change in Mixed Traffic." IEEE Access.

Game structure:
    EV (autonomous vehicle) = Leader  → proposes lane-change intent + acceleration
    HV (human-driven vehicle) = Follower → responds with optimal speed

Simplified from the paper's MPC+GA bi-level optimization to discrete
candidate enumeration for real-time highway-env integration.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import DRIVING_STYLE_WEIGHTS, GameConfig
from .trajectory_predictor import (
    TrajectoryPoint,
    VehicleState,
    predict_ev_candidate,
    predict_hv_response,
)
from .utility_functions import compute_ev_cost, compute_hv_utility

# highway-env DiscreteMetaAction mapping
ACTION_LEFT = 0
ACTION_IDLE = 1
ACTION_RIGHT = 2
ACTION_FASTER = 3
ACTION_SLOWER = 4

LATERAL_TO_ACTION = {-1: ACTION_LEFT, 0: ACTION_IDLE, 1: ACTION_RIGHT}


@dataclass
class CandidateAction:
    lateral: int        # -1=left, 0=stay, 1=right
    accel: float        # longitudinal acceleration (m/s^2)
    label: str = ""     # human-readable label

    def __post_init__(self):
        lat_name = {-1: "LEFT", 0: "STAY", 1: "RIGHT"}
        self.label = f"{lat_name[self.lateral]}_{self.accel:+.1f}"


@dataclass
class GameResult:
    """Output of one Stackelberg game solve."""
    action: int                     # highway-env action (0-4)
    lateral_choice: int             # -1, 0, 1
    optimal_accel: float            # chosen longitudinal acceleration
    ev_cost_original_lane: float    # cost of staying in current lane
    ev_cost_target_lane: float      # cost of lane-changing
    cost_improvement: float         # positive = lane change is beneficial
    hv_driving_style: str           # "aggressive" / "normal" / "conservative"
    hv_predicted_speed: float       # HV optimal response speed
    min_ttc: float
    min_gap: float
    game_success: bool              # whether a valid equilibrium was found
    candidates_evaluated: int


@dataclass
class KeyVehicles:
    """Surrounding vehicles relevant to lane-change decision."""
    fv_curr: Optional[VehicleState] = None   # front vehicle in current lane
    fv_target: Optional[VehicleState] = None # front vehicle in target lane
    rv_target: Optional[VehicleState] = None # rear vehicle in target lane (the HV)


class StackelbergSolver:
    """Core Stackelberg game solver for lane-changing decisions."""

    def __init__(self, config: GameConfig):
        self.config = config
        self._prev_accel: float = 0.0
        self._rv_speed_history: Dict[int, List[float]] = {}

    def _extract_vehicle_state(self, vehicle) -> VehicleState:
        """Extract VehicleState from highway-env vehicle object."""
        pos = vehicle.position
        if hasattr(pos, 'tolist'):
            pos = pos.tolist()
        return VehicleState(
            x=float(pos[0]),
            y=float(pos[1]),
            vx=float(vehicle.speed),
            vy=0.0,
            ax=getattr(vehicle, 'acceleration', 0.0) or 0.0,
            heading=float(getattr(vehicle, 'heading', 0.0) or 0.0),
        )

    def _identify_key_vehicles(self, env) -> KeyVehicles:
        """Identify FV_curr, FV_target, RV_target from highway-env state."""
        ego = env.unwrapped.vehicle
        ego_lane = ego.lane_index
        road = env.unwrapped.road

        result = KeyVehicles()
        all_vehicles = [v for v in road.vehicles if v is not ego]

        ego_pos = np.array(ego.position)
        ego_lane_coords = road.network.get_lane(ego_lane).local_coordinates(ego.position)
        ego_s = float(ego_lane_coords[0])

        for v in all_vehicles:
            v_pos = np.array(v.position)
            v_lane = v.lane_index

            # Get longitudinal position in ego's lane
            try:
                v_coords = road.network.get_lane(ego_lane).local_coordinates(v.position)
                v_s = float(v_coords[0])
                lat_dist = float(v_coords[1])
            except (ValueError, IndexError):
                continue

            # Same lane
            if v_lane == ego_lane:
                if v_s > ego_s:  # ahead
                    if result.fv_curr is None or v_s < (
                        road.network.get_lane(ego_lane).local_coordinates(
                            result.fv_curr._raw_position
                        )[0] if hasattr(result.fv_curr, '_raw_position') else float('inf')
                    ):
                        state = self._extract_vehicle_state(v)
                        state._raw_position = v_pos
                        state._s = v_s
                        result.fv_curr = state

        # Target lane vehicles
        for lateral in [-1, 1]:  # left, right
            target_lane = (ego_lane[0], ego_lane[1], ego_lane[2] + lateral)
            if not self._lane_exists(road, target_lane):
                continue

            target_lane_obj = road.network.get_lane(target_lane)
            try:
                tgt_coords = target_lane_obj.local_coordinates(ego_pos)
                ego_s_tgt = float(tgt_coords[0])
            except (ValueError, IndexError):
                continue

            best_fv = None
            best_rv = None
            best_fv_s = float('inf')
            best_rv_s = float('-inf')

            for v in all_vehicles:
                if v.lane_index != target_lane:
                    continue
                try:
                    v_coords = target_lane_obj.local_coordinates(v.position)
                    v_s = float(v_coords[0])
                except (ValueError, IndexError):
                    continue

                if v_s > ego_s_tgt:  # ahead
                    if v_s < best_fv_s:
                        best_fv_s = v_s
                        best_fv = v
                else:  # behind
                    if v_s > best_rv_s:
                        best_rv_s = v_s
                        best_rv = v

            if lateral == -1 and not result.rv_target:  # left lane priority
                pass  # will be set below based on which lane we're evaluating

            # Store all results
            pass

        return result

    def _lane_exists(self, road, lane_index: Tuple) -> bool:
        start, end, lane_id = lane_index
        graph = road.network.graph
        if start not in graph or end not in graph[start]:
            return False
        return 0 <= lane_id < len(graph[start][end])

    def _get_vehicles_for_lane(self, env, target_lateral: int) -> Tuple[Optional[VehicleState], Optional[VehicleState], Optional[VehicleState]]:
        """Get (FV_curr, FV_target, RV_target) for a specific target lane.

        target_lateral: -1 for left lane, 0 for current lane, 1 for right lane.
        Returns three VehicleState objects.
        """
        ego = env.unwrapped.vehicle
        ego_lane = ego.lane_index
        road = env.unwrapped.road
        all_vehicles = [v for v in road.vehicles if v is not ego]

        try:
            ego_coords = road.network.get_lane(ego_lane).local_coordinates(ego.position)
            ego_s = float(ego_coords[0])
        except (ValueError, IndexError):
            return None, None, None

        # FV in current lane
        fv_curr = None
        fv_curr_s = float('inf')
        for v in all_vehicles:
            if v.lane_index != ego_lane:
                continue
            try:
                v_s = float(road.network.get_lane(ego_lane).local_coordinates(v.position)[0])
                if v_s > ego_s and v_s < fv_curr_s:
                    fv_curr_s = v_s
                    fv_curr = v
            except (ValueError, IndexError):
                continue

        if target_lateral == 0:
            fv_state = self._extract_vehicle_state(fv_curr) if fv_curr else None
            return fv_state, None, None

        # Target lane
        target_lane = (ego_lane[0], ego_lane[1], ego_lane[2] + target_lateral)
        if not self._lane_exists(road, target_lane):
            return (
                self._extract_vehicle_state(fv_curr) if fv_curr else None,
                None, None
            )

        target_lane_obj = road.network.get_lane(target_lane)
        try:
            ego_s_tgt = float(target_lane_obj.local_coordinates(ego.position)[0])
        except (ValueError, IndexError):
            ego_s_tgt = ego_s

        fv_target = None
        rv_target = None
        fv_s_min = float('inf')
        rv_s_max = float('-inf')

        for v in all_vehicles:
            if v.lane_index != target_lane:
                continue
            try:
                v_s = float(target_lane_obj.local_coordinates(v.position)[0])
            except (ValueError, IndexError):
                continue

            if v_s > ego_s_tgt and v_s < fv_s_min:
                fv_s_min = v_s
                fv_target = v
            elif v_s <= ego_s_tgt and v_s > rv_s_max:
                rv_s_max = v_s
                rv_target = v

        return (
            self._extract_vehicle_state(fv_curr) if fv_curr else None,
            self._extract_vehicle_state(fv_target) if fv_target else None,
            self._extract_vehicle_state(rv_target) if rv_target else None,
        )

    def _classify_driving_style(
        self,
        rv_state: Optional[VehicleState],
        rv_id: int = 0,
    ) -> str:
        """Classify HV driving style from observed behavior.

        Uses speed volatility and TTC variance (from tech roadmap 2.3.1).
        Falls back to "normal" when insufficient history.
        """
        if rv_state is None:
            return "normal"

        # Track speed history for this RV
        if rv_id not in self._rv_speed_history:
            self._rv_speed_history[rv_id] = []
        history = self._rv_speed_history[rv_id]
        v = float(np.linalg.norm([rv_state.vx, rv_state.vy]))
        history.append(v)
        if len(history) > 50:
            history.pop(0)

        if len(history) < 10:
            return "normal"

        speeds = np.array(history)
        speed_volatility = float(np.std(speeds) / max(np.mean(speeds), 1e-3))

        # Speed volatility thresholds
        if speed_volatility > 0.15:
            return "aggressive"
        elif speed_volatility < 0.03:
            return "conservative"
        return "normal"

    def _solve_hv_best_response(
        self,
        ev_candidate: CandidateAction,
        fv_target: Optional[VehicleState],
        rv_target: VehicleState,
        style: str,
    ) -> float:
        """Find HV optimal target speed given EV's proposed action.

        The HV selects target speed to maximize its utility J_HV.
        We enumerate candidate target speeds for the HV.

        Returns optimal target speed for HV.
        """
        config = self.config
        weights = DRIVING_STYLE_WEIGHTS[style]
        horizon = config.prediction_horizon

        # Generate EV trajectory for this candidate
        ev_lateral = 0
        if ev_candidate.lateral != 0:
            pass  # EV's lateral position doesn't affect HV calculation much

        # EV simplified trajectory (longitudinal only for HV interaction)
        ev_traj = predict_ev_candidate(
            ev_state=self._last_ev_state,
            longitudinal_accel=ev_candidate.accel,
            lateral_action=0,  # STAY for HV computation (focus on longitudinal)
            horizon=horizon,
            dt=config.dt,
            lane_width=4.0,
            lc_duration=config.lc_duration,
        )

        # Enumerate HV speed candidates
        v_current = float(np.linalg.norm([rv_target.vx, rv_target.vy]))
        speed_options = np.linspace(
            max(5.0, v_current - 8.0),
            min(35.0, v_current + 8.0),
            15,
        )

        best_speed = v_current
        best_utility = -float('inf')

        for v_target in speed_options:
            hv_traj = predict_hv_response(rv_target, ev_candidate.accel, float(v_target), horizon, config.dt)
            utility = compute_hv_utility(
                ev_traj, hv_traj,
                hv_target_speed=float(v_target),
                style_weights=weights,
                tau=config.tau_system + config.tau_driver,
                a_brake=config.max_brake,
                dx_lim=100.0,
                max_accel=config.max_accel,
                dv_max=config.max_speed_deviation,
            )
            if utility.total > best_utility:
                best_utility = utility.total
                best_speed = float(v_target)

        return best_speed

    def _evaluate_ev_cost_for_candidate(
        self,
        candidate: CandidateAction,
        fv_curr: Optional[VehicleState],
        fv_target: Optional[VehicleState],
        rv_target: Optional[VehicleState],
    ) -> float:
        """Evaluate EV total cost for a candidate action.

        Returns scalar cost (lower = better).
        """
        config = self.config
        ev_state = self._last_ev_state

        ev_traj = predict_ev_candidate(
            ev_state=ev_state,
            longitudinal_accel=candidate.accel,
            lateral_action=candidate.lateral,
            horizon=config.prediction_horizon,
            dt=config.dt,
            lane_width=self._lane_width if hasattr(self, '_lane_width') else 4.0,
            lc_duration=config.lc_duration,
        )

        # Default states if None
        dummy_fv = VehicleState(x=ev_state.x + 200.0, y=ev_state.y, vx=30.0, vy=0.0)
        dummy_rv = VehicleState(x=ev_state.x - 200.0, y=ev_state.y, vx=0.0, vy=0.0)

        # For lane-change candidates, use target lane's FV (EV ends up in target lane).
        # For STAY, use current lane's FV.
        if candidate.lateral != 0:
            fv_ref = fv_target if fv_target is not None else dummy_fv
        else:
            fv_ref = fv_curr if fv_curr is not None else dummy_fv

        # rv_ref = 目标车道后车，用于横向安全代价计算
        rv_ref = rv_target if rv_target is not None else dummy_rv

        result = compute_ev_cost(
            ev_traj=ev_traj,
            fv_state=fv_ref,
            rv_state=rv_ref,
            config=config,
            prev_accel=self._prev_accel,
            horizon=config.prediction_horizon,
        )

        return result.total

    def _generate_candidates(self, current_lane_id: int, num_lanes: int) -> List[CandidateAction]:
        """Generate all candidate (lateral, longitudinal) action combinations."""
        candidates = []

        # Lateral options based on lane position
        lateral_options = [0]  # always include STAY
        if current_lane_id > 0:
            lateral_options.append(-1)  # can go left
        if current_lane_id < num_lanes - 1:
            lateral_options.append(1)   # can go right

        for lat in lateral_options:
            for accel in self.config.accel_candidates:
                candidates.append(CandidateAction(lateral=lat, accel=float(accel)))

        return candidates

    def solve(self, env) -> GameResult:
        """Run Stackelberg game to find optimal lane-change decision.

        Args:
            env: highway-env environment (unwrapped for state access).

        Returns:
            GameResult with optimal action and diagnostic info.
        """
        config = self.config
        ego = env.unwrapped.vehicle
        ego_lane = ego.lane_index
        num_lanes = len(ego_lane) if isinstance(ego_lane, (list, tuple)) else 4

        # Store EV state for trajectory prediction
        self._last_ev_state = self._extract_vehicle_state(ego)
        road = env.unwrapped.road

        # Get road dimensions
        graph = road.network.graph
        all_lanes = []
        for start in graph:
            for end in graph.get(start, {}):
                all_lanes.extend(graph[start][end])
        self._lane_width = float(all_lanes[0].width) if all_lanes else 4.0

        current_lane_id = ego_lane[2] if len(ego_lane) > 2 else 0
        candidates = self._generate_candidates(current_lane_id, num_lanes)

        # Compute cost of staying (baseline)
        fv_curr, _, _ = self._get_vehicles_for_lane(env, 0)
        stay_candidate = CandidateAction(lateral=0, accel=0.0)
        cost_stay = self._evaluate_ev_cost_for_candidate(stay_candidate, fv_curr, None, None)

        best_result: Optional[GameResult] = None
        best_cost = float('inf')
        evaluated = 0

        for candidate in candidates:
            # Get relevant vehicles for this lateral action
            fv_curr, fv_target, rv_target = self._get_vehicles_for_lane(env, candidate.lateral)

            if candidate.lateral != 0 and fv_target is None and rv_target is None:
                # Can't access target lane
                continue

            # Classify RV driving style
            rv_id = id(rv_target._raw_position) if hasattr(rv_target, '_raw_position') and rv_target is not None else 0
            style = self._classify_driving_style(rv_target, rv_id)

            # If there's an RV, solve for HV best response
            if rv_target is not None and candidate.lateral != 0:
                hv_best_speed = self._solve_hv_best_response(candidate, fv_target, rv_target, style)
                # Update RV state with predicted response
                rv_target_resp = VehicleState(
                    x=rv_target.x,
                    y=rv_target.y,
                    vx=hv_best_speed,
                    vy=rv_target.vy,
                    ax=rv_target.ax,
                    heading=rv_target.heading,
                )
            else:
                hv_best_speed = 0.0
                rv_target_resp = rv_target

            # Evaluate EV cost
            cost = self._evaluate_ev_cost_for_candidate(candidate, fv_curr, fv_target, rv_target_resp)
            evaluated += 1

            if cost < best_cost:
                best_cost = cost
                # Compute TTC and gap
                ev_traj = predict_ev_candidate(
                    self._last_ev_state, candidate.accel, candidate.lateral,
                    config.prediction_horizon, config.dt,
                    self._lane_width, config.lc_duration,
                )
                min_ttc = float('inf')
                min_gap = float('inf')
                if rv_target is not None:
                    rv_traj = predict_hv_response(rv_target, hv_best_speed, hv_best_speed,
                                                  config.prediction_horizon, config.dt)
                    from .trajectory_predictor import compute_ttc_gap
                    min_ttc, min_gap, _, _ = compute_ttc_gap(ev_traj, rv_traj)

                # Map to highway-env action
                if candidate.lateral == -1:
                    action = ACTION_LEFT
                elif candidate.lateral == 1:
                    action = ACTION_RIGHT
                elif candidate.accel > 0.5:
                    action = ACTION_FASTER
                elif candidate.accel < -0.5:
                    action = ACTION_SLOWER
                else:
                    action = ACTION_IDLE

                cost_improvement = cost_stay - cost

                best_result = GameResult(
                    action=action,
                    lateral_choice=candidate.lateral,
                    optimal_accel=candidate.accel,
                    ev_cost_original_lane=float(cost_stay),
                    ev_cost_target_lane=float(cost),
                    cost_improvement=float(cost_improvement),
                    hv_driving_style=style,
                    hv_predicted_speed=float(hv_best_speed),
                    min_ttc=float(min_ttc),
                    min_gap=float(min_gap),
                    game_success=True,
                    candidates_evaluated=evaluated,
                )

        if best_result is None:
            return GameResult(
                action=ACTION_IDLE,
                lateral_choice=0,
                optimal_accel=0.0,
                ev_cost_original_lane=float(cost_stay),
                ev_cost_target_lane=float(cost_stay),
                cost_improvement=0.0,
                hv_driving_style="normal",
                hv_predicted_speed=0.0,
                min_ttc=float('inf'),
                min_gap=float('inf'),
                game_success=False,
                candidates_evaluated=evaluated,
            )

        self._prev_accel = best_result.optimal_accel
        return best_result
