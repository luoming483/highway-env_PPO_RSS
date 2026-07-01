"""Debug single seed to trace decision timeline with gap info."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import highway_env  # noqa
import numpy as np
from stackelberg.config import GameConfig
from stackelberg.expert import StackelbergExpert

seed = 42
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

# Access FSM internals for gap debugging
fsm = expert.fsm

print(f"{'Step':>4s} {'Action':>8s} {'FSM':>16s} {'Reason':>32s} "
      f"{'Lat':>4s} {'EgoLane':>8s} {'GapCur':>8s} {'GapTgt':>8s} "
      f"{'TtcCur':>7s} {'TtcTgt':>7s} {'Crash':>5s}")
print("-" * 150)

while not done and step < 30:
    ego = env.unwrapped.vehicle
    ego_lane = ego.lane_index
    ego_lane_str = str(ego_lane[2]) if len(ego_lane) > 2 else "?"

    cur_gap, cur_rel, _, _ = fsm._get_gaps_from_env(env, lane_offset=0)
    cur_ttc = fsm._safety_gate.predict_ttc(cur_gap, cur_rel)

    action, info = expert.decide(env, dt=0.25)
    obs, reward, terminated, truncated, env_info = env.step(action)
    step += 1

    crashed = env_info.get("crashed", False)
    lat = info.get("game_lateral", 0)
    lat_str = {-1: "L", 0: "S", 1: "R"}.get(lat, "?")

    tgt_gap = float('inf')
    tgt_ttc = float('inf')
    if lat != 0:
        tgt_gap, tgt_rel_get, _, _ = fsm._get_gaps_from_env(env, lane_offset=lat)
        tgt_ttc = fsm._safety_gate.predict_ttc(tgt_gap, tgt_rel_get)

    ttc_s = f"{cur_ttc:.1f}" if np.isfinite(cur_ttc) else "inf"
    tgt_ttc_s = f"{tgt_ttc:.1f}" if np.isfinite(tgt_ttc) else "inf"

    crash_marker = " ***" if crashed else ""

    print(f"{step:4d} {info.get('action_label', '?'):>8s} {info.get('fsm_state', '?'):>16s} "
          f"{info.get('fsm_reason', ''):>32s} {lat_str:>4s} {ego_lane_str:>8s} "
          f"{cur_gap:8.1f} {tgt_gap:8.1f} {ttc_s:>7s} {tgt_ttc_s:>7s} {crash_marker}")

    if terminated or truncated:
        break

env.close()
print(f"\nFinal: crashed={env_info.get('crashed', False) if env_info else '?'}, steps={step}")
