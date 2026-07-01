"""Debug: Verify ForceExplore wrapper is actually firing during training."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
from config import ENV_CONFIG, RSS_CONFIG, TRAIN_RSS_OVERRIDES, BLOCKED_PENALTY
from rss import RSSConfig, RSSSafetyWrapper
from train import BlockedPenaltyWrapper, ForceExploreWrapper

# Build the exact same wrapper chain used in training
rss_params = dict(RSS_CONFIG)
rss_params.update(TRAIN_RSS_OVERRIDES)
train_rss = RSSConfig(**rss_params)

env = gym.make("highway-fast-v0", config=ENV_CONFIG)
env = BlockedPenaltyWrapper(env)
env = RSSSafetyWrapper(env, rss_config=train_rss)
env = ForceExploreWrapper(env, explore_prob=0.25, decay_steps=100_000)

print("Wrapper chain:")
e = env
while True:
    print(f"  {type(e).__name__}")
    if hasattr(e, 'env') and isinstance(e.env, gym.Env):
        e = e.env
    else:
        break

# Run a few episodes and count actions
total_actions = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
total_overrides = 0

for ep in range(3):
    obs, _ = env.reset()
    for _ in range(200):
        # Agent always outputs FASTER (action 3) — simulating the trained behavior
        agent_action = 3  # FASTER
        obs, reward, term, trunc, info = env.step(agent_action)
        total_actions[agent_action] += 1
        info_override = info.get("force_explored", False)
        if info_override:
            total_overrides += 1
        if term or trunc:
            break

print(f"\nTotal steps: {sum(total_actions.values())}")
print(f"Action distribution: {total_actions}")
print(f"Overrides: {total_overrides} ({total_overrides/sum(total_actions.values())*100:.1f}%)")
