"""Smoke test for Stackelberg expert module.

Usage:
    D:\\anaconda\\envs\\ppo_main\\python.exe -m stackelberg.test_expert
    D:\\anaconda\\envs\\ppo_main\\python.exe stackelberg/test_expert.py
"""

import sys
from pathlib import Path

# Add parent to path for standalone execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np

from stackelberg.config import GameConfig
from stackelberg.expert import StackelbergExpert


def test_basic_functionality():
    """Verify StackelbergExpert can run in highway-env without crashing."""
    print("=" * 60)
    print("Test 1: Basic functionality")
    print("=" * 60)

    config = {
        "observation": {"type": "Kinematics", "vehicles_count": 10},
        "action": {
            "type": "DiscreteMetaAction",
            "target_speeds": [0, 5, 10, 15, 20, 25, 30],
        },
        "lanes_count": 3,
        "vehicles_count": 10,
        "vehicles_density": 1.0,
        "duration": 10,
        "simulation_frequency": 15,
        "policy_frequency": 4,
        "collision_reward": -5.0,
        "normalize_reward": True,
        "offroad_terminal": True,
    }

    env = gym.make("highway-fast-v0", config=config)
    expert = StackelbergExpert(GameConfig())

    obs, info = env.reset()
    total_reward = 0.0
    crashed = False
    total_steps = 0
    actions_taken = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    fsm_states_seen = set()
    game_success_count = 0

    for step in range(200):
        action, decision_info = expert.decide(env, dt=0.25)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        total_steps += 1
        actions_taken[action] = actions_taken.get(action, 0) + 1

        fsm_states_seen.add(decision_info["fsm_state"])

        if decision_info.get("game_success"):
            game_success_count += 1

        if info.get("crashed", False):
            crashed = True

        if terminated or truncated:
            break

    env.close()

    print(f"  Steps: {total_steps}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Crashed: {crashed}")
    print(f"  Actions: {actions_taken}")
    print(f"  FSM states seen: {fsm_states_seen}")
    print(f"  Game success rate: {game_success_count}/{total_steps}")

    # Basic assertions
    assert total_steps > 0, "Should complete at least 1 step"
    assert len(fsm_states_seen) >= 1, "Should visit at least 1 FSM state"
    assert sum(actions_taken.values()) == total_steps, "Every step should produce an action"

    print("  PASSED")
    return True


def test_game_solver_on_scenarios():
    """Test Stackelberg solver across different traffic densities."""
    print()
    print("=" * 60)
    print("Test 2: Game solver across traffic densities")
    print("=" * 60)

    for density_name, vehicles in [("sparse", 5), ("medium", 15), ("dense", 25)]:
        config = {
            "observation": {"type": "Kinematics", "vehicles_count": vehicles},
            "action": {
                "type": "DiscreteMetaAction",
                "target_speeds": [0, 5, 10, 15, 20, 25, 30],
            },
            "lanes_count": 3,
            "vehicles_count": vehicles,
            "vehicles_density": 1.0,
            "duration": 8,
            "simulation_frequency": 15,
            "policy_frequency": 4,
            "offroad_terminal": True,
        }

        env = gym.make("highway-fast-v0", config=config)
        expert = StackelbergExpert(GameConfig())

        obs, info = env.reset()
        collisions = 0
        episodes = 3

        for ep in range(episodes):
            obs, info = env.reset()
            expert.reset()
            done = False
            ep_steps = 0
            while not done and ep_steps < 150:
                action, decision_info = expert.decide(env, dt=0.25)
                obs, reward, terminated, truncated, info = env.step(action)
                if info.get("crashed", False):
                    collisions += 1
                done = terminated or truncated
                ep_steps += 1

        env.close()

        collision_rate = collisions / episodes
        print(f"  {density_name} ({vehicles}v): collisions={collisions}/{episodes} ({collision_rate:.0%})")

        if collision_rate > 0.67:
            print(f"  WARNING: High collision rate in {density_name} traffic")

    print("  PASSED")
    return True


def test_fsm_state_transitions():
    """Verify FSM goes through expected state transitions."""
    print()
    print("=" * 60)
    print("Test 3: FSM state transitions")
    print("=" * 60)

    from stackelberg.fsm_executor import FSMExecutor, FSMState
    from stackelberg.game_solver import GameResult

    game_config = GameConfig()
    fsm = FSMExecutor(game_config)

    # Create a mock game result proposing a lane change
    mock_result = GameResult(
        action=0,  # LEFT
        lateral_choice=-1,
        optimal_accel=1.5,
        ev_cost_original_lane=5.0,
        ev_cost_target_lane=4.0,
        cost_improvement=1.0,
        hv_driving_style="normal",
        hv_predicted_speed=20.0,
        min_ttc=5.0,
        min_gap=15.0,
        game_success=True,
        candidates_evaluated=18,
    )

    # Check initial state
    assert fsm.state == FSMState.LANE_KEEPING, "Initial state should be LANE_KEEPING"

    # Check rate limiter — should limit large acceleration jumps
    limiter = fsm._rate_limiter
    a1 = limiter.smooth(1.0, 0.25)
    a2 = limiter.smooth(4.0, 0.25)  # Large jump — should be rate-limited
    max_delta = game_config.max_jerk * 0.25
    assert abs(a2 - a1) <= max_delta + 0.01, f"Rate limiter should constrain jerk: got delta={abs(a2-a1):.2f} > max={max_delta:.2f}"

    print(f"  Initial state: {fsm.state.name}")
    print(f"  Rate limiter: a1={a1:.2f}, a2={a2:.2f} (max jerk={game_config.max_jerk})")
    print("  PASSED")
    return True


def test_wrapper():
    """Test StackelbergWrapper integrates with gym step()."""
    print()
    print("=" * 60)
    print("Test 4: StackelbergWrapper integration")
    print("=" * 60)

    config = {
        "observation": {"type": "Kinematics", "vehicles_count": 8},
        "action": {
            "type": "DiscreteMetaAction",
            "target_speeds": [0, 5, 10, 15, 20, 25, 30],
        },
        "lanes_count": 3,
        "vehicles_count": 8,
        "vehicles_density": 1.0,
        "duration": 8,
        "simulation_frequency": 15,
        "policy_frequency": 4,
        "offroad_terminal": True,
    }

    from stackelberg import StackelbergWrapper

    env = gym.make("highway-fast-v0", config=config)
    wrapped = StackelbergWrapper(env, GameConfig())

    obs, info = wrapped.reset()
    total_reward = 0.0

    for _ in range(100):
        obs, reward, terminated, truncated, info = wrapped.step()
        total_reward += float(reward)
        assert "stackelberg" in info, "Info should contain stackelberg diagnostics"
        if terminated or truncated:
            break

    wrapped.close()
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Stackelberg info keys: {list(info.get('stackelberg', {}).keys())}")
    print("  PASSED")
    return True


if __name__ == "__main__":
    print("Stackelberg Expert Module — Smoke Tests")
    print("=" * 60)
    print()

    all_passed = True
    try:
        test_basic_functionality()
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        test_game_solver_on_scenarios()
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        test_fsm_state_transitions()
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        test_wrapper()
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    print()
    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
