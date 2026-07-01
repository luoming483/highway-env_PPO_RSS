"""Visualize trained PPO+RSS policy in highway-env with pygame.

Usage:
    # Default: latest our_method model with 20 vehicles
    D:\\anaconda\\envs\\ppo_main\\python.exe ppo/visualize.py

    # Specify model and scenario
    D:\\anaconda\\envs\\ppo_main\\python.exe ppo/visualize.py --seed 42 --vehicles 20 --duration 30

    # Console-only diagnostic mode
    D:\\anaconda\\envs\\ppo_main\\python.exe ppo/visualize.py --render console
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np

# ---- Auto-detect latest PPO model ----
def _find_best_model(seed: int = 42) -> str:
    candidates = [
        Path("runs/20260615_163841/models"),
        Path("results_v1_30k/models"),
        Path("results/models"),
    ]
    for base in candidates:
        model_path = base / f"our_method_seed{seed}" / "final_model.zip"
        if model_path.exists():
            return str(model_path.resolve())
    raise FileNotFoundError(f"No PPO model found for seed {seed}")


# Terminal colors
C_RESET = "\033[0m"
C_RED = "\033[91m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_CYAN = "\033[96m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"


def run_pygame(env, model, fps_limit=15):
    """Interactive pygame window with diagnostic output."""
    from stable_baselines3 import PPO

    model_obj = PPO.load(model, device="cpu")

    obs, _ = env.reset()
    done = False
    truncated = False
    step = 0
    last_time = time.time()
    frame_delay = 1.0 / fps_limit

    print("=" * 80)
    print("PPO+RSS Policy — Visual Debug")
    print("=" * 80)
    print(f"Model: {model}")
    print()
    print("Controls (terminal):")
    print("  Enter     = pause / single step")
    print("  q + Enter = quit")
    print()
    print(f"{'Step':>5s} {'Action':>8s} {'Reward':>8s} {'Speed':>7s} "
          f"{'Lane':>5s} {'TTC':>8s} {'Gap':>8s} {'RSS':>8s}")
    print("-" * 75)

    auto_mode = True
    try:
        while not (done or truncated):
            if not auto_mode:
                cmd = input().strip().lower()
                if cmd == "q":
                    break
                else:
                    auto_mode = True
                    continue
            else:
                import msvcrt
                if msvcrt.kbhit():
                    ch = msvcrt.getch()
                    if ch in (b'q', b'Q'):
                        break
                    auto_mode = False
                    print("[Paused. Enter = step, any other key = resume]")
                    continue
                time.sleep(frame_delay)

            action, _ = model_obj.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray):
                action = int(action.item())
            else:
                action = int(action)

            obs, reward, terminated, truncated, env_info = env.step(action)
            done = terminated or truncated
            step += 1

            speed = float(env.unwrapped.vehicle.speed)
            lane = env.unwrapped.vehicle.lane_index[2]
            ttc = float(env_info.get("rss_min_ttc", float("inf")))
            gap = float(env_info.get("rss_min_distance", float("inf")))
            rss_intervened = env_info.get("rss_intervened", False)
            action_names = {0: "LEFT", 1: "IDLE", 2: "RIGHT", 3: "FASTER", 4: "SLOWER"}

            ttc_s = f"{ttc:.1f}" if np.isfinite(ttc) else "inf"
            gap_s = f"{gap:.1f}" if np.isfinite(gap) else "inf"
            rss_s = f"{C_RED}RSS{C_RESET}" if rss_intervened else "-"
            crash_marker = f" {C_RED}*** CRASH ***{C_RESET}" if env_info.get("crashed", False) else ""

            print(f"{step:5d} {action_names.get(action, '?'):>8s} {reward:8.2f} {speed:6.1f}m/s "
                  f"{lane:5d} {ttc_s:>8s} {gap_s:>8s} {rss_s:>8s}{crash_marker}")

    except KeyboardInterrupt:
        pass

    env.close()
    crashed = env_info.get("crashed", False) if env_info else False
    print(f"\nDone. Steps: {step}, Crashed: {crashed}")


def run_console(env, model):
    """Console-only diagnostic mode with step-by-step control."""
    from stable_baselines3 import PPO

    model_obj = PPO.load(model, device="cpu")

    obs, _ = env.reset()
    done = False
    truncated = False
    step = 0
    auto_steps = 0

    print("PPO+RSS — Console Diagnostic Mode")
    print("=" * 60)
    print("Controls: Enter=step, 'a'=auto-run 50 steps, 'q'=quit")
    input("Press Enter to start...")

    while not (done or truncated):
        if auto_steps > 0:
            auto_steps -= 1
            time.sleep(0.1)
        else:
            cmd = input().strip().lower()
            if cmd == "a":
                auto_steps = 50
                continue
            elif cmd == "q":
                break

        action, _ = model_obj.predict(obs, deterministic=True)
        if isinstance(action, np.ndarray):
            action = int(action.item())
        else:
            action = int(action)

        obs, reward, terminated, truncated, env_info = env.step(action)
        done = terminated or truncated
        step += 1

        speed = float(env.unwrapped.vehicle.speed)
        ttc = float(env_info.get("rss_min_ttc", float("inf")))
        gap = float(env_info.get("rss_min_distance", float("inf")))
        rss = env_info.get("rss_intervened", False)

        os.system("cls" if os.name == "nt" else "clear")
        print("=" * 60)
        print(f"  PPO+RSS — Step {step}")
        print("=" * 60)
        action_names = {0: "LEFT", 1: "IDLE", 2: "RIGHT", 3: "FASTER", 4: "SLOWER"}
        print(f"  Action:     {action_names.get(action, '?')}")
        print(f"  Speed:      {speed:.1f} m/s")
        print(f"  Reward:     {reward:.2f}")
        ttc_c = C_GREEN if np.isfinite(ttc) and ttc > 3.0 else (C_YELLOW if np.isfinite(ttc) and ttc > 1.5 else C_RED)
        print(f"  Min TTC:    {ttc_c}{ttc:.2f}s{C_RESET}" if np.isfinite(ttc) else f"  Min TTC:    inf")
        print(f"  Min Gap:    {gap:.1f}m")
        print(f"  RSS Interv: {C_RED if rss else C_GREEN}{rss}{C_RESET}")
        print(f"  Crashed:    {C_RED if env_info.get('crashed') else C_GREEN}{env_info.get('crashed', False)}{C_RESET}")

    env.close()
    print(f"\nEpisode finished. Steps: {step}")


def main():
    parser = argparse.ArgumentParser(description="Visualize PPO+RSS policy in highway-env")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained PPO model .zip (auto-detect if omitted)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Model seed to use (default: 42)")
    parser.add_argument("--render", type=str, default="human",
                        choices=["human", "console"],
                        help="Render mode: human (pygame), console (terminal)")
    parser.add_argument("--vehicles", type=int, default=20,
                        help="Number of vehicles")
    parser.add_argument("--duration", type=int, default=30,
                        help="Episode duration (s)")
    parser.add_argument("--density", type=float, default=1.0,
                        help="Vehicle density")
    parser.add_argument("--fps", type=int, default=15,
                        help="Display FPS")
    parser.add_argument("--env-seed", type=int, default=None,
                        help="Environment random seed")
    args = parser.parse_args()

    # Resolve model path
    model_path = args.model or _find_best_model(args.seed)
    print(f"Model: {model_path}")

    # Build env matching training config
    env_config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": args.vehicles,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "absolute": False,
        },
        "action": {
            "type": "DiscreteMetaAction",
            "target_speeds": [0, 5, 10, 15, 20, 25, 30],
        },
        "lanes_count": 4,
        "vehicles_count": args.vehicles,
        "vehicles_density": args.density,
        "duration": args.duration,
        "simulation_frequency": 8,
        "policy_frequency": 4,
        "collision_reward": -5.0,
        "normalize_reward": True,
        "offroad_terminal": True,
    }

    # Create env with RSS wrapper (matching training setup)
    from config import RSS_CONFIG
    from rss import RSSConfig as _RSSConfig, RSSSafetyWrapper
    from gymnasium.wrappers import FlattenObservation

    base_env = gym.make(
        "highway-fast-v0",
        config=env_config,
        render_mode="human" if args.render == "human" else None,
    )
    rss_env = RSSSafetyWrapper(base_env, rss_config=_RSSConfig(**RSS_CONFIG))
    env = FlattenObservation(rss_env)

    if args.render == "console":
        run_console(env, model_path)
    else:
        run_pygame(env, model_path, fps_limit=args.fps)


if __name__ == "__main__":
    main()
