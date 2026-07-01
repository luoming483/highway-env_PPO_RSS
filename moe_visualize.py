"""MoE Hybrid Expert — Pygame Visualization for group meeting demo.

Usage:
    D:\\anaconda\\envs\\ppo_main\\python.exe moe_visualize.py
    D:\\anaconda\\envs\\ppo_main\\python.exe moe_visualize.py --density 1.5 --duration 60
    D:\\anaconda\\envs\\ppo_main\\python.exe moe_visualize.py --seed 789 --fps 10

Controls:
    Space / Enter  =  pause / resume
    S              =  single-step (when paused)
    Q / Esc        =  quit
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from moe_hybrid import HybridExpert, make_env

# Terminal colors
RESET = "\033[0m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"

ACTION_NAMES = {0: "LEFT", 1: "IDLE", 2: "RIGHT", 3: "FASTER", 4: "SLOWER"}
EXPERT_COLORS = {"rss_emergency": RED, "stackelberg": CYAN, "ppo_rss": GREEN}

# Separator line
SEP = "-" * 120


def _find_ppo_model(seed: int = 42) -> str:
    candidates = [
        Path("results/models/test_lc_phased_v3_seed42/final_model.zip"),
        Path("results/models/our_method_seed42/final_model.zip"),
        Path("runs/20260615_163841/models/our_method_seed42/final_model.zip"),
    ]
    for p in candidates:
        if p.exists():
            return str(p.resolve())
    raise FileNotFoundError(f"No PPO model found for seed {seed}")


def clear_screen():
    import os
    os.system("cls" if os.name == "nt" else "clear")


def run_visualization(hybrid, base_env, flat_env, fps: int = 15, seed: int = None):
    """Run MoE Hybrid Expert with pygame window + color-coded terminal HUD."""
    obs, _ = flat_env.reset(seed=seed)
    hybrid.reset()

    step = 0
    paused = False
    frame_delay = 1.0 / fps
    crashed = False
    total_reward = 0.0

    expert_seq = []
    speed_seq = []
    action_seq = []

    last_time = time.time()

    # Header
    clear_screen()
    print(BOLD + "=" * 120 + RESET)
    print(BOLD + "  MoE Hybrid Expert — Live Visualization (Group Meeting Demo)" + RESET)
    print(BOLD + "=" * 120 + RESET)
    print(f"  Gate: TTC<3s→Emergency | LC beneficial→Stackelberg | Default→PPO+RSS")
    print(f"  Controls: [Space/Enter]=pause  [S]=step  [Q/Esc]=quit")
    print(SEP)
    print(f"{'Step':>5s} {'Expert':>16s} {'Action':>8s} {'Speed':>7s} "
          f"{'Lane':>5s} {'FrontTTC':>9s} {'CostImp':>8s} "
          f"{'Blocked':>8s} {'Crashed':>8s}  Reason")
    print(SEP)

    try:
        while True:
            # Keyboard input (Windows)
            import msvcrt
            if msvcrt.kbhit():
                ch = msvcrt.getch().lower()
                if ch in (b' ', b'\r'):
                    paused = not paused
                    status = "PAUSED" if paused else "RESUMED"
                    print(f"\n  {YELLOW}[{status}]{RESET}")
                    if paused:
                        print(f"  Press [S] to single-step, [Space] to resume, [Q] to quit")
                elif ch == b's' and paused:
                    pass  # single step: skip the sleep, do one iteration
                elif ch in (b'q', b'\x1b'):
                    print(f"\n  {YELLOW}[QUIT by user]{RESET}")
                    break

            if paused and not (msvcrt.kbhit() and msvcrt.getch().lower() == b's'):
                time.sleep(0.05)
                continue

            # Decide action
            action, info = hybrid.decide(base_env, obs, dt=0.25)
            obs, reward, terminated, truncated, env_info = flat_env.step(action)
            total_reward += float(reward)
            step += 1

            # Track
            expert = info["moe_expert"]
            speed = info["scene_ego_speed"]
            front_ttc = info["scene_front_ttc"]
            cost_imp = info["game_cost_improvement"]
            lane = base_env.unwrapped.vehicle.lane_index[2]
            blocked = "YES" if info.get("blocked_aware", False) else "-"

            expert_seq.append(expert)
            speed_seq.append(speed)
            action_seq.append(action)

            if env_info.get("crashed", False):
                crashed = True

            # Print step
            color = EXPERT_COLORS.get(expert, RESET)
            ttc_s = f"{front_ttc:.1f}s" if np.isfinite(front_ttc) else "inf"
            crash_s = f"{RED}YES{RESET}" if env_info.get("crashed", False) else "-"

            print(f"{step:5d} {color}{expert:>16s}{RESET} {ACTION_NAMES.get(action, '?'):>8s} "
                  f"{speed:6.1f}m/s {lane:5d} {ttc_s:>9s} {cost_imp:8.3f} "
                  f"{blocked:>8s} {crash_s:>8s}  {info['moe_reason'][:55]}")

            if terminated or truncated:
                break

            time.sleep(frame_delay)

    except KeyboardInterrupt:
        pass

    # Summary
    if not expert_seq:
        return

    # Count distribution
    ppo_n = expert_seq.count("ppo_rss")
    stack_n = expert_seq.count("stackelberg")
    rss_n = expert_seq.count("rss_emergency")
    total = len(expert_seq)

    avg_speed = np.mean(speed_seq) if speed_seq else 0.0
    lc_count = sum(1 for a in action_seq if a in (0, 2))

    flat_env.close()

    print(SEP)
    print(BOLD + "  EPISODE SUMMARY" + RESET)
    print(SEP)
    print(f"  Steps:           {step}")
    print(f"  Crashed:         {RED + 'YES' if crashed else GREEN + 'NO'}{RESET}")
    print(f"  Avg Speed:       {avg_speed:.1f} m/s")
    print(f"  LC Actions:      {lc_count}")
    print(f"  Total Reward:    {total_reward:.1f}")
    print()
    print(f"  Expert Distribution:")
    bar_w = 40
    ppo_bar = int(bar_w * ppo_n / total)
    stack_bar = int(bar_w * stack_n / total)
    rss_bar = bar_w - ppo_bar - stack_bar
    print(f"    {GREEN}PPO+RSS{RESET}      {ppo_n:4d} ({ppo_n/total:5.1%})  "
          f"{GREEN}{'=' * ppo_bar}{RESET}")
    print(f"    {CYAN}Stackelberg{RESET}  {stack_n:4d} ({stack_n/total:5.1%})  "
          f"{CYAN}{'=' * stack_bar}{RESET}")
    print(f"    {RED}RSS Emerg{RESET}    {rss_n:4d} ({rss_n/total:5.1%})  "
          f"{RED}{'=' * rss_bar}{RESET}")

    # Expert transition analysis
    transitions = sum(1 for i in range(1, total) if expert_seq[i] != expert_seq[i - 1])
    print(f"\n  Expert Transitions: {transitions}")
    print(SEP)


def main():
    parser = argparse.ArgumentParser(
        description="MoE Hybrid Expert — Pygame Visualization"
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Path to PPO model .zip")
    parser.add_argument("--seed", type=int, default=42,
                        help="PPO model seed (default: 42)")
    parser.add_argument("--env-seed", type=int, default=None,
                        help="Environment random seed")
    parser.add_argument("--vehicles", type=int, default=20,
                        help="Number of vehicles")
    parser.add_argument("--duration", type=int, default=40,
                        help="Episode duration in seconds")
    parser.add_argument("--density", type=float, default=1.2,
                        help="Vehicle density (0.8=sparse, 1.2=medium, 1.5=dense)")
    parser.add_argument("--fps", type=int, default=12,
                        help="Display FPS (lower = slower, easier to observe)")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable pygame window (terminal-only)")
    args = parser.parse_args()

    model_path = args.model or _find_ppo_model(args.seed)
    env_seed = args.env_seed if args.env_seed is not None else args.seed

    print(f"PPO model: {model_path}")
    print(f"Config: density={args.density}, vehicles={args.vehicles}, "
          f"duration={args.duration}s, fps={args.fps}")
    if args.no_render:
        print(f"{YELLOW}PyGame window disabled (--no-render){RESET}")
    print()

    base_env, flat_env = make_env(
        vehicles=args.vehicles,
        duration=args.duration,
        density=args.density,
        seed=env_seed,
        render=not args.no_render,
    )

    hybrid = HybridExpert(ppo_model_path=model_path)
    run_visualization(hybrid, base_env, flat_env, fps=args.fps, seed=env_seed)


if __name__ == "__main__":
    main()
