"""Quick collision test across multiple seeds."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import highway_env  # noqa
import numpy as np
from stackelberg.config import GameConfig
from stackelberg.expert import StackelbergExpert

seeds = [42, 123, 456, 789, 1024]
results = []

for seed in seeds:
    env_config = {
        "observation": {"type": "Kinematics", "vehicles_count": 20},
        "action": {
            "type": "DiscreteMetaAction",
            "target_speeds": [0, 5, 10, 15, 20, 25, 30],
        },
        "lanes_count": 3,
        "vehicles_count": 20,
        "vehicles_density": 1.5,
        "duration": 20,
        "simulation_frequency": 15,
        "policy_frequency": 4,
        "collision_reward": -5.0,
        "normalize_reward": True,
        "offroad_terminal": True,
    }
    env = gym.make("highway-fast-v0", config=env_config)
    expert = StackelbergExpert(GameConfig())

    obs, _ = env.reset(seed=seed)
    expert.reset()
    done = False
    step = 0
    crashed = False
    actions = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    fsm_states = {}
    reasons = []

    while not done and step < 200:
        action, info = expert.decide(env, dt=0.25)
        obs, reward, terminated, truncated, env_info = env.step(action)
        step += 1
        actions[action] = actions.get(action, 0) + 1
        state = info.get("fsm_state", "?")
        fsm_states[state] = fsm_states.get(state, 0) + 1
        reason = info.get("fsm_reason", "")
        if reason:
            reasons.append(reason)
        if env_info.get("crashed", False):
            crashed = True
        if terminated or truncated:
            break

    env.close()

    status = "CRASH" if crashed else "OK"
    results.append((seed, crashed, step, actions, fsm_states, reasons[-5:] if reasons else []))
    print(f"Seed {seed:4d}: {status:5s}  steps={step:3d}  "
          f"LEFT={actions[0]:3d} IDLE={actions[1]:3d} RIGHT={actions[2]:3d} "
          f"FASTER={actions[3]:3d} SLOWER={actions[4]:3d}  FSM={fsm_states}")
    if reasons:
        print(f"         Last 3 reasons: {reasons[-3:]}")

crashed_count = sum(1 for r in results if r[1])
print(f"\n=== Collision rate: {crashed_count}/{len(results)} ===")
