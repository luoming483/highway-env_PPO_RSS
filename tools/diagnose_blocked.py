"""Diagnose: how often is the blocked condition actually firing?"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
from config import ENV_CONFIG, RSS_CONFIG, TRAIN_RSS_OVERRIDES
from rss import RSSConfig, RSSSafetyWrapper
from scene_utils import check_ego_blocked, compute_front_ttc_gap
from ppo.train import BlockedPenaltyWrapper, ForceExploreWrapper

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
        is_blocked, gap, front_speed = check_ego_blocked(env, gap_threshold=80.0, speed_ratio=0.90)

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
    gap, _, front_speed = compute_front_ttc_gap(env)
    if gap > 0 and gap < float('inf'):
        gaps.append(gap)
        speed_ratios.append(front_speed / max(float(env.unwrapped.vehicle.speed), 1e-6))
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
