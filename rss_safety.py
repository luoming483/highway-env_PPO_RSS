from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np


@dataclass
class RSSConfig:
    response_time: float = 0.8
    rear_response_time: float = 0.6
    min_distance: float = 5.0
    max_brake: float = 6.0
    ttc_threshold: float = 2.0
    lane_width: float = 4.0
    intervention_penalty: float = -2.5
    nearby_vehicle_horizon: float = 45.0
    lane_change_side_gap: float = 8.0
    enable_shield: bool = True


class RSSSafetyWrapper(gym.Wrapper):
    """
    Action-end RSS shield for highway-env discrete meta-actions.
    0: LANE_LEFT, 1: IDLE, 2: LANE_RIGHT, 3: FASTER, 4: SLOWER
    """

    ACTION_LEFT = 0
    ACTION_IDLE = 1
    ACTION_RIGHT = 2
    ACTION_FASTER = 3
    ACTION_SLOWER = 4

    def __init__(self, env: gym.Env, rss_config: RSSConfig):
        super().__init__(env)
        self.rss = rss_config
        self._episode_rss_intervened = False
        self._episode_had_crash = False

    def reset(self, **kwargs):
        self._episode_rss_intervened = False
        self._episode_had_crash = False
        return self.env.reset(**kwargs)

    def _lane_exists(self, lane_index: Tuple[str, str, int]) -> bool:
        start, end, lane_id = lane_index
        graph = self.unwrapped.road.network.graph
        if start not in graph or end not in graph[start]:
            return False
        lanes = graph[start][end]
        return 0 <= lane_id < len(lanes)

    def _lane_longitudinal(self, lane_index: Tuple[str, str, int], vehicle) -> float:
        lane = self.unwrapped.road.network.get_lane(lane_index)
        longi, _ = lane.local_coordinates(vehicle.position)
        return float(longi)

    def _safe_distance_front(self, ego_speed: float, front_speed: float) -> float:
        braking_term = max(0.0, (ego_speed**2 - front_speed**2) / (2.0 * max(self.rss.max_brake, 1e-3)))
        return float(self.rss.min_distance + ego_speed * self.rss.response_time + braking_term)

    def _safe_distance_rear(self, rear_speed: float, ego_speed: float) -> float:
        braking_term = max(0.0, (rear_speed**2 - ego_speed**2) / (2.0 * max(self.rss.max_brake, 1e-3)))
        return float(self.rss.min_distance + rear_speed * self.rss.rear_response_time + braking_term)

    def _front_rear_on_lane(self, lane_index: Tuple[str, str, int]):
        ego = self.unwrapped.vehicle
        front, rear = self.unwrapped.road.neighbour_vehicles(ego, lane_index)
        return front, rear

    def _front_gap_and_ttc(
        self,
        ego,
        front,
        lane_index: Tuple[str, str, int],
    ) -> Tuple[float, float]:
        if front is None:
            return np.inf, np.inf
        ego_s = self._lane_longitudinal(lane_index, ego)
        front_s = self._lane_longitudinal(lane_index, front)
        gap = float(front_s - ego_s)
        rel_speed = float(ego.speed - front.speed)
        ttc = np.inf
        if gap > 0.0 and rel_speed > 1e-6:
            ttc = gap / rel_speed
        return gap, float(ttc)

    def _rear_gap_and_ttc(
        self,
        ego,
        rear,
        lane_index: Tuple[str, str, int],
    ) -> Tuple[float, float]:
        if rear is None:
            return np.inf, np.inf
        ego_s = self._lane_longitudinal(lane_index, ego)
        rear_s = self._lane_longitudinal(lane_index, rear)
        gap = float(ego_s - rear_s)
        rel_speed = float(rear.speed - ego.speed)
        ttc = np.inf
        if gap > 0.0 and rel_speed > 1e-6:
            ttc = gap / rel_speed
        return gap, float(ttc)

    def _candidate_lane(self, action: int) -> Tuple[str, str, int]:
        lane = self.unwrapped.vehicle.lane_index
        if action == self.ACTION_LEFT:
            return lane[0], lane[1], lane[2] - 1
        if action == self.ACTION_RIGHT:
            return lane[0], lane[1], lane[2] + 1
        return lane

    def _compute_min_distance(self) -> float:
        ego = self.unwrapped.vehicle
        other_vehicles = [v for v in self.unwrapped.road.vehicles if v is not ego]
        if not other_vehicles:
            return np.inf
        dists = [float(np.linalg.norm(v.position - ego.position)) for v in other_vehicles]
        return float(np.min(dists))

    def _compute_min_ttc(self) -> float:
        ego = self.unwrapped.vehicle
        other_vehicles = [v for v in self.unwrapped.road.vehicles if v is not ego]
        min_ttc = np.inf
        for other in other_vehicles:
            dy = float(abs(other.position[1] - ego.position[1]))
            if dy > self.rss.lane_width * 0.65:
                continue
            dx = float(other.position[0] - ego.position[0])
            if dx > 0.0:
                closing = float(ego.speed - other.speed)
                if closing > 1e-6:
                    min_ttc = min(min_ttc, dx / closing)
            elif dx < 0.0:
                closing = float(other.speed - ego.speed)
                if closing > 1e-6:
                    min_ttc = min(min_ttc, (-dx) / closing)
        return float(min_ttc)

    def _assess_action_risk(self, action: int) -> Tuple[bool, str, dict]:
        ego = self.unwrapped.vehicle
        lane_index = self._candidate_lane(action)
        details = {
            "front_gap": np.inf,
            "front_ttc": np.inf,
            "rear_gap": np.inf,
            "rear_ttc": np.inf,
            "safe_front_distance": np.inf,
            "safe_rear_distance": np.inf,
            "nearby_risk": False,
        }

        if not self._lane_exists(lane_index):
            return True, "invalid_lane", details

        front, rear = self._front_rear_on_lane(lane_index)
        front_gap, front_ttc = self._front_gap_and_ttc(ego, front, lane_index)
        rear_gap, rear_ttc = self._rear_gap_and_ttc(ego, rear, lane_index)
        safe_front = self._safe_distance_front(float(ego.speed), float(front.speed) if front is not None else 0.0)
        safe_rear = self._safe_distance_rear(float(rear.speed) if rear is not None else 0.0, float(ego.speed))

        details.update(
            {
                "front_gap": front_gap,
                "front_ttc": front_ttc,
                "rear_gap": rear_gap,
                "rear_ttc": rear_ttc,
                "safe_front_distance": safe_front,
                "safe_rear_distance": safe_rear,
            }
        )
        lane_change = action in (self.ACTION_LEFT, self.ACTION_RIGHT)
        nearby_risk, nearby_reason = self._scan_nearby_vehicle_risk(
            ego,
            lane_index,
            details,
            check_rear=lane_change,
            check_side=lane_change,
        )
        if lane_change:
            if nearby_risk:
                return True, nearby_reason, details
            if front_gap < safe_front or front_ttc < self.rss.ttc_threshold:
                return True, "lane_change_front_risk", details
            if rear_gap < safe_rear or rear_ttc < self.rss.ttc_threshold:
                return True, "lane_change_rear_risk", details
            return False, "", details

        if action == self.ACTION_FASTER:
            if nearby_risk:
                return True, nearby_reason, details
            if front_gap < safe_front or front_ttc < self.rss.ttc_threshold:
                return True, "accel_front_risk", details
        if action == self.ACTION_IDLE:
            if nearby_risk:
                return True, nearby_reason, details
            if front_ttc < (0.8 * self.rss.ttc_threshold):
                return True, "idle_imminent_front_risk", details
        if action == self.ACTION_SLOWER:
            if front_ttc < (0.5 * self.rss.ttc_threshold):
                return True, "slower_insufficient", details
        return False, "", details

    def _scan_nearby_vehicle_risk(
        self,
        ego,
        lane_index: Tuple[str, str, int],
        details: dict,
        check_rear: bool,
        check_side: bool,
    ) -> Tuple[bool, str]:
        lane = self.unwrapped.road.network.get_lane(lane_index)
        ego_s, _ = lane.local_coordinates(ego.position)
        ego_s = float(ego_s)
        horizon = float(self.rss.nearby_vehicle_horizon)
        side_gap = float(self.rss.lane_change_side_gap)
        lateral_margin = float(self.rss.lane_width * 0.75)

        for other in self.unwrapped.road.vehicles:
            if other is ego:
                continue
            other_s, other_lat = lane.local_coordinates(other.position)
            if abs(float(other_lat)) > lateral_margin:
                continue
            other_s = float(other_s)
            gap = float(other_s - ego_s)
            if abs(gap) > horizon:
                continue

            if check_side and abs(gap) < side_gap:
                details["nearby_risk"] = True
                return True, "nearby_vehicle_side_risk"

            if gap > 0.0:
                closing = float(ego.speed - other.speed)
                ttc = gap / closing if closing > 1e-6 else np.inf
                safe_front = self._safe_distance_front(float(ego.speed), float(other.speed))
                if gap < safe_front or ttc < self.rss.ttc_threshold:
                    details["nearby_risk"] = True
                    details["front_gap"] = min(float(details["front_gap"]), gap)
                    details["front_ttc"] = min(float(details["front_ttc"]), float(ttc))
                    return True, "nearby_vehicle_front_risk"
            else:
                if not check_rear:
                    continue
                rear_gap = -gap
                closing = float(other.speed - ego.speed)
                ttc = rear_gap / closing if closing > 1e-6 else np.inf
                safe_rear = self._safe_distance_rear(float(other.speed), float(ego.speed))
                if rear_gap < safe_rear or ttc < self.rss.ttc_threshold:
                    details["nearby_risk"] = True
                    details["rear_gap"] = min(float(details["rear_gap"]), rear_gap)
                    details["rear_ttc"] = min(float(details["rear_ttc"]), float(ttc))
                    return True, "nearby_vehicle_rear_risk"

        return False, ""

    def _choose_safe_action(self, original_action: int, risk_reason: str) -> int:
        if risk_reason == "slower_insufficient":
            return self.ACTION_SLOWER
        if original_action == self.ACTION_FASTER:
            return self.ACTION_SLOWER
        if original_action == self.ACTION_IDLE:
            return self.ACTION_SLOWER
        if original_action in (self.ACTION_LEFT, self.ACTION_RIGHT):
            return self.ACTION_SLOWER
        return self.ACTION_SLOWER

    def step(self, action):
        original_action = int(action)
        unsafe, reason, details = self._assess_action_risk(original_action)
        intervened = bool(self.rss.enable_shield and unsafe)
        final_action = self._choose_safe_action(original_action, reason) if intervened else original_action
        changed_action = bool(intervened and final_action != original_action)

        obs, reward, terminated, truncated, info = self.env.step(final_action)
        if info is None:
            info = {}

        if changed_action:
            self._episode_rss_intervened = True

        if bool(info.get("crashed", False)):
            self._episode_had_crash = True

        if terminated or truncated:
            if self._episode_rss_intervened and self._episode_had_crash:
                reward = float(reward) + float(self.rss.intervention_penalty)
            self._episode_rss_intervened = False
            self._episode_had_crash = False

        min_ttc = self._compute_min_ttc()
        min_distance = self._compute_min_distance()

        info["rss_enabled"] = bool(self.rss.enable_shield)
        info["rss_intervened"] = changed_action
        info["rss_original_action"] = original_action
        info["rss_final_action"] = final_action
        info["rss_reason"] = reason if intervened else ""
        info["rss_penalty"] = float(self.rss.intervention_penalty) if changed_action else 0.0
        info["rss_min_ttc"] = float(min_ttc)
        info["rss_min_distance"] = float(min_distance)
        info["rss_front_gap"] = float(details["front_gap"])
        info["rss_front_ttc"] = float(details["front_ttc"])
        info["rss_safe_front_distance"] = float(details["safe_front_distance"])
        info["rss_safe_rear_distance"] = float(details["safe_rear_distance"])
        return obs, reward, terminated, truncated, info
