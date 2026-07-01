"""Diagnose: how often is the blocked condition actually firing?"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
from config import ENV_CONFIG, RSS_CONFIG, TRAIN_RSS_OVERRIDES
from rss import RSSConfig, RSSSafetyWrapper
from train import BlockedPenaltyWrapper, ForceExploreWrapper

rss_params = dict(RSS_CONFIG)
rss_params.update(TRAIN_RSS_OVERRIDES)
train_rss = RSSConfig(**rss_params)

env = gym.make("highway-fast-v0", config=ENV_CONFIG)
env = BlockedPenaltyWrapper(env)
env = RSSSafetyWrapper(env, rss_config=train_rss)
env = ForceExploreWrapper(env)

total_steps = 0
blocked_steps = 0
lc_attempts = 0
lc_successes = 0
bonus_steps = 0

for ep in range(5):
    obs, _ = env.reset()
    for _ in range(400):
        total_steps += 1
        action = 3  # Always FASTER

        # Manual blocked check (same logic as in wrappers)
        ego = env.unwrapped.vehicle
        road = env.unwrapped.road
        front, _ = road.neighbour_vehicles(ego, ego.lane_index)
        is_blocked = False
        gap = float('inf')
        front_speed = 0
        if front is not None:
            lane = road.network.get_lane(ego.lane_index)
            ego_s = float(lane.local_coordinates(ego.position)[0])
            front_s = float(lane.local_coordinates(front.position)[0])
            gap = front_s - ego_s
            front_speed = float(front.speed)
            if (0 < gap < 80.0 and front_speed < 0.90 * ego.speed):
                is_blocked = True

        obs, reward, term, trunc, info = env.step(action)

        if is_blocked:
            blocked_steps += 1
        if info.get("force_explored"):
            lc_attempts += 1
        if info.get("blocked_bonus", 0) != 0:
            bonus_steps += 1
        if reward > 0.5:  # Large positive reward (possible LC completion)
            lc_successes += 1

        if term or trunc:
            break

print(f"Total steps: {total_steps}")
print(f"Blocked steps: {blocked_steps} ({blocked_steps/total_steps*100:.1f}%)")
print(f"Force-explore overrides: {lc_attempts}")
print(f"Bonus steps (blocked+LC): {bonus_steps}")
print(f"Steps with large reward: {lc_successes}")

# Detailed stats on gap and speed ratio
print(f"\nDetailed stats from a typical episode:")
obs, _ = env.reset()
gaps = []
speed_ratios = []
for _ in range(400):
    ego = env.unwrapped.vehicle
    road = env.unwrapped.road
    front, _ = road.neighbour_vehicles(ego, ego.lane_index)
    if front is not None:
        lane = road.network.get_lane(ego.lane_index)
        ego_s = float(lane.local_coordinates(ego.position)[0])
        front_s = float(lane.local_coordinates(front.position)[0])
        gap = front_s - ego_s
        if gap > 0:
            gaps.append(gap)
            speed_ratios.append(float(front.speed) / max(float(ego.speed), 1e-6))
    obs, reward, term, trunc, info = env.step(3)
    if term or trunc:
        break

if gaps:
    gaps = np.array(gaps)
    ratios = np.array(speed_ratios)
    print(f"  Front gap: mean={np.mean(gaps):.0f}m, median={np.median(gaps):.0f}m, min={np.min(gaps):.0f}m")
    print(f"  Speed ratio: mean={np.mean(ratios):.3f}, median={np.median(ratios):.3f}, min={np.min(ratios):.3f}")
    print(f"  Gap < 80m: {np.sum(gaps < 80)}/{len(gaps)} ({np.sum(gaps < 80)/len(gaps)*100:.1f}%)")
    print(f"  Speed ratio < 0.90: {np.sum(ratios < 0.90)}/{len(ratios)} ({np.sum(ratios < 0.90)/len(ratios)*100:.1f}%)")
    both = np.sum((gaps < 80) & (ratios < 0.90))
    print(f"  BOTH (gap<80 & ratio<0.90): {both}/{len(gaps)} ({both/len(gaps)*100:.1f}%)")
