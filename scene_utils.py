"""Shared scene understanding utilities for MoE Highway framework.

Single source of truth for: lane existence, longitudinal position, blocked
detection, density classification, TTC/gap computation.

All modules (rss/, stackelberg/, moe_hybrid.py, train.py, tools/) import from here.
"""

from typing import Tuple

import numpy as np


def check_lane_exists(road, lane_index: Tuple) -> bool:
    """Check if a lane index is valid in the road network."""
    start, end, lane_id = lane_index
    graph = road.network.graph
    if start not in graph or end not in graph.get(start, {}):
        return False
    return 0 <= lane_id < len(graph[start][end])


def get_lane_longitudinal(road, lane_index: Tuple, vehicle) -> float:
    """Get longitudinal position of a vehicle along a specific lane."""
    lane = road.network.get_lane(lane_index)
    longi, _ = lane.local_coordinates(vehicle.position)
    return float(longi)


def compute_front_ttc_gap(env, lane_offset: int = 0) -> Tuple[float, float, float]:
    """Compute (front_gap, front_ttc, front_speed) for ego in the target lane.

    Args:
        env: highway-env environment (uses .unwrapped).
        lane_offset: 0=current lane, -1=left, +1=right.

    Returns:
        (gap, ttc, front_speed) — inf for gap/ttc if no front vehicle.
    """
    ego = env.unwrapped.vehicle
    road = env.unwrapped.road
    ego_lane = ego.lane_index
    target_lane = (ego_lane[0], ego_lane[1], ego_lane[2] + lane_offset)

    if not check_lane_exists(road, target_lane):
        return float("inf"), float("inf"), 0.0

    try:
        ego_s = get_lane_longitudinal(road, target_lane, ego)
    except (ValueError, IndexError):
        return float("inf"), float("inf"), 0.0

    front, _ = road.neighbour_vehicles(ego, target_lane)
    if front is None:
        return float("inf"), float("inf"), 0.0

    try:
        front_s = get_lane_longitudinal(road, target_lane, front)
    except (ValueError, IndexError):
        return float("inf"), float("inf"), 0.0

    gap = float(front_s - ego_s)
    rel_speed = float(ego.speed - front.speed)
    ttc = gap / rel_speed if gap > 0.0 and rel_speed > 1e-6 else float("inf")
    return gap, float(ttc), float(front.speed)


def check_ego_blocked(env, gap_threshold: float = 80.0, speed_ratio: float = 0.85) -> Tuple[bool, float, float]:
    """Check if ego is blocked behind a significantly slower front vehicle.

    Args:
        env: highway-env environment (uses .unwrapped).
        gap_threshold: maximum following gap to be considered "blocked".
        speed_ratio: front_speed < speed_ratio * ego_speed → blocked.

    Returns:
        (is_blocked, front_gap, front_speed)
    """
    ego = env.unwrapped.vehicle
    road = env.unwrapped.road
    front, _ = road.neighbour_vehicles(ego, ego.lane_index)
    if front is None:
        return False, float("inf"), 0.0
    try:
        ego_s = get_lane_longitudinal(road, ego.lane_index, ego)
        front_s = get_lane_longitudinal(road, ego.lane_index, front)
        gap = front_s - ego_s
        blocked = (0.0 < gap < gap_threshold) and (float(front.speed) < speed_ratio * float(ego.speed))
        return blocked, float(gap), float(front.speed)
    except (ValueError, IndexError):
        return False, float("inf"), 0.0


def classify_density(env) -> str:
    """Classify traffic density from env config's vehicles_density.

    Returns: "sparse" | "medium" | "dense"
    """
    env_density = float(env.unwrapped.config.get("vehicles_density", 1.0))
    if env_density <= 0.9:
        return "sparse"
    elif env_density <= 1.3:
        return "medium"
    else:
        return "dense"


def compute_min_ttc(env, lane_width: float = 4.0) -> float:
    """Compute minimum TTC to any nearby vehicle (filtered by lateral proximity)."""
    ego = env.unwrapped.vehicle
    other_vehicles = [v for v in env.unwrapped.road.vehicles if v is not ego]
    min_ttc = float("inf")
    for other in other_vehicles:
        dy = float(abs(other.position[1] - ego.position[1]))
        if dy > lane_width * 0.65:
            continue
        dx = float(other.position[0] - ego.position[0])
        if dx > 0.0:
            closing = float(ego.speed - other.speed)
            if closing > 1e-6:
                min_ttc = min(min_ttc, dx / closing)
    return float(min_ttc)
