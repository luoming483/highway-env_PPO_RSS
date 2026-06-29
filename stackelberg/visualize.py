"""Visualize Stackelberg expert behavior in highway-env.

Two modes:
  --render human   : highway-env pygame window + console diagnostic dump (always works)
  --render plot    : matplotlib figure with decision info overlay (requires matplotlib)
  --save video.mp4 : headless run, save to MP4 (requires: pip install imageio[ffmpeg])

Usage:
    # Interactive pygame window (zero extra deps)
    D:\\anaconda\\envs\\ppo_main\\python.exe stackelberg/visualize.py

    # Console-only with rich diagnostics
    D:\\anaconda\\envs\\ppo_main\\python.exe stackelberg/visualize.py --render console

    # Save to video (requires imageio[ffmpeg])
    D:\\anaconda\\envs\\ppo_main\\python.exe stackelberg/visualize.py --save video.mp4

    # Custom scenario
    D:\\anaconda\\envs\\ppo_main\\python.exe stackelberg/visualize.py --vehicles 8 --duration 15
"""

import argparse
import io
import os
import sys
import time
from pathlib import Path

# Fix Unicode output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np

from stackelberg import GameConfig, StackelbergExpert

# Terminal colors (ANSI)
C_RESET = "\033[0m"
C_RED = "\033[91m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_BLUE = "\033[94m"
C_CYAN = "\033[96m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"

FSM_COLOR = {
    "LANE_KEEPING": C_GREEN,
    "LC_PREPARATION": C_YELLOW,
    "LC_EXECUTION": C_CYAN,
    "STATE_RECOVERY": C_BLUE,
}

STYLE_COLOR = {
    "aggressive": C_RED,
    "normal": C_DIM,
    "conservative": C_GREEN,
}


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_dashboard(step, info, env_info, width=80):
    """Print a rich terminal dashboard of the expert's internal state."""
    clear_screen()

    # Top bar
    fsm_state = info.get("fsm_state", "?")
    fsm_c = FSM_COLOR.get(fsm_state, C_RESET)
    action_label = info.get("action_label", "?")

    print(C_BOLD + "=" * width + C_RESET)
    print(f"  {C_BOLD}Stackelberg Expert — Step {step}{C_RESET}"
          f"  |  Action: {C_YELLOW}{action_label}{C_RESET}"
          f"  |  FSM: {fsm_c}{fsm_state}{C_RESET}"
          f"  |  Crashed: {C_RED if env_info.get('crashed') else C_GREEN}{env_info.get('crashed', False)}{C_RESET}")
    print("=" * width)

    # FSM Section
    print(f"\n{C_BOLD}--- Finite State Machine ---{C_RESET}")
    print(f"  State:           {fsm_c}{fsm_state}{C_RESET}")
    print(f"  Time in state:   {info.get('fsm_time_in_state', 0):.2f}s")
    print(f"  Cooldown left:   {info.get('fsm_cooldown_remaining', 0):.2f}s")
    print(f"  Reason:          {C_DIM}{info.get('fsm_reason', '')}{C_RESET}")
    overridden = info.get("fsm_game_overridden", False)
    print(f"  Game overridden: {C_RED if overridden else C_GREEN}{overridden}{C_RESET}")

    # FSM state diagram
    states = ["LANE_KEEPING", "LC_PREPARATION", "LC_EXECUTION", "STATE_RECOVERY"]
    diagram = ""
    for s in states:
        marker = "#" if s == fsm_state else "-"
        c = FSM_COLOR.get(s, C_RESET)
        diagram += f"{c}[{marker * 5}] {s[:6]:6s} {C_RESET}> "
    diagram = diagram.rstrip("→ ")
    print(f"\n  {diagram}")

    # Game Solver Section
    print(f"\n{C_BOLD}--- Stackelberg Game Solver ---{C_RESET}")
    lateral = info.get("game_lateral", 0)
    lat_str = {-1: "LEFT", 0: "STAY", 1: "RIGHT"}.get(lateral, str(lateral))
    print(f"  Lateral choice:     {lat_str}")
    print(f"  Optimal accel:      {info.get('game_optimal_accel', 0):+.2f} m/s²")
    cost_orig = info.get("game_cost_original", 0)
    cost_tgt = info.get("game_cost_target", 0)
    impr = info.get("game_cost_improvement", 0)
    impr_c = C_GREEN if impr > 0 else C_RED
    print(f"  Cost (orig lane):   {cost_orig:.3f}")
    print(f"  Cost (target lane): {cost_tgt:.3f}")
    print(f"  Improvement:        {impr_c}{impr:+.3f}{C_RESET}")
    print(f"  Candidates evaluated: {info.get('game_candidates', 0)}")

    # Neighbor Vehicle Section
    print(f"\n{C_BOLD}--- Target-Lane Vehicle (HV / Follower) ---{C_RESET}")
    style = info.get("game_hv_style", "?")
    style_c = STYLE_COLOR.get(style, C_RESET)
    print(f"  Driving style:   {style_c}{style}{C_RESET}")
    print(f"  Predicted speed: {info.get('game_hv_speed', 0):.1f} m/s")

    # Safety Section
    print(f"\n{C_BOLD}--- Safety ---{C_RESET}")
    ttc = info.get("min_ttc", float("inf"))
    if np.isfinite(ttc):
        ttc_c = C_GREEN if ttc > 3.0 else (C_YELLOW if ttc > 1.5 else C_RED)
        print(f"  Min TTC:  {ttc_c}{ttc:.2f}s{C_RESET}")
    else:
        print(f"  Min TTC:  {C_GREEN}inf{C_RESET}")
    print(f"  Min gap:  {info.get('min_gap', 0):.2f}m")

    print(f"\n{C_DIM}{'=' * width}{C_RESET}")
    print(f"{C_DIM}Controls: Enter=step, Ctrl+C=quit, 'a'=auto-run 50 steps{C_RESET}")


def run_console(env_config, game_config, seed=None):
    """Console-only mode with rich diagnostic dashboard after each step."""
    env = gym.make("highway-fast-v0", config=env_config)
    expert = StackelbergExpert(game_config)

    obs, _ = env.reset(seed=seed)
    expert.reset()
    done = False
    truncated = False
    step = 0
    auto_steps = 0
    last_time = time.time()

    print_dashboard(step, {}, {}, 80)
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
            # else: Enter = single step

        dt = time.time() - last_time
        last_time = time.time()
        action, info = expert.decide(env, dt=min(dt, 0.5))
        obs, reward, terminated, truncated, env_info = env.step(action)
        done = terminated or truncated
        step += 1

        full_info = {k: v for k, v in env_info.items()} if env_info and isinstance(env_info, dict) else {}
        print_dashboard(step, info, full_info, 80)

    env.close()
    print(f"\n{C_BOLD}Episode finished. Steps: {step}{C_RESET}")


def run_pygame(env_config, game_config, fps_limit=15, seed=None):
    """Interactive pygame window with highway-env's built-in renderer.

    Decision info is printed to the console alongside the visual window.
    """
    env = gym.make("highway-fast-v0", config=env_config, render_mode="human")
    expert = StackelbergExpert(game_config)

    obs, _ = env.reset(seed=seed)
    expert.reset()
    done = False
    truncated = False
    step = 0
    auto_mode = True  # 默认自动运行
    last_time = time.time()
    frame_delay = 1.0 / fps_limit

    print("=" * 60)
    print("Stackelberg Expert — Visual Debug")
    print("=" * 60)
    print(f"Vehicles: {env_config['vehicles_count']}, Lanes: {env_config['lanes_count']}")
    print()
    print("Controls (在终端操作):")
    print("  Enter     = 暂停 / 单步")
    print("  q + Enter = 退出")
    print()
    print(f"{'Step':>5s} {'Action':>8s} {'FSM State':>16s} {'Reason':>30s} "
          f"{'Lat':>4s} {'Accel':>7s} {'Cost Orig':>10s} {'Cost Tgt':>10s} "
          f"{'Impr':>8s} {'HV Style':>10s} {'TTC':>8s}")
    print("-" * 125)

    try:
        while not (done or truncated):
            if not auto_mode:
                cmd = input().strip().lower()
                if cmd == "q":
                    break
                else:
                    auto_mode = True  # 任意键恢复自动
                    continue
            else:
                # 检查是否有键盘输入（暂停）
                import msvcrt
                if msvcrt.kbhit():
                    ch = msvcrt.getch()
                    if ch in (b'q', b'Q'):
                        break
                    auto_mode = False
                    print("[已暂停，按 Enter 单步，按其他键继续自动运行]")
                    continue
                time.sleep(frame_delay)

            dt = time.time() - last_time
            last_time = time.time()
            action, info = expert.decide(env, dt=min(dt, 0.5))
            obs, reward, terminated, truncated, env_info = env.step(action)
            done = terminated or truncated
            step += 1

            # Print diagnostic line
            fsm_state = info.get("fsm_state", "?")
            lat = {-1: "L", 0: "S", 1: "R"}.get(info.get("game_lateral", 0), "?")
            impr = info.get("game_cost_improvement", 0)
            impr_s = f"{impr:+.2f}" if impr != 0 else "0.00"
            ttc = info.get("min_ttc", float("inf"))
            ttc_s = f"{ttc:.1f}" if np.isfinite(ttc) else "inf"

            crash_marker = " *** CRASH ***" if env_info.get("crashed", False) else ""

            print(f"{step:5d} {info.get('action_label', '?'):>8s} {fsm_state:>16s} "
                  f"{info.get('fsm_reason', ''):>30s} {lat:>4s} "
                  f"{info.get('game_optimal_accel', 0):+6.2f} "
                  f"{info.get('game_cost_original', 0):10.3f} {info.get('game_cost_target', 0):10.3f} "
                  f"{impr_s:>8s} {info.get('game_hv_style', '?'):>10s} {ttc_s:>8s}"
                  f"{C_RED if env_info.get('crashed') else ''}{crash_marker}{C_RESET}")

    except KeyboardInterrupt:
        pass

    env.close()
    print(f"\nDone. Steps: {step}, Crashed: {env_info.get('crashed', False) if env_info else '?'}")


def run_matplotlib(env_config, game_config, fps=10):
    """Matplotlib-based visualization with embedded decision overlay.

    Requires: matplotlib (already used by plotting.py in this project)
    """
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.animation import FuncAnimation

    env = gym.make("highway-fast-v0", config=env_config, render_mode="rgb_array")
    expert = StackelbergExpert(game_config)

    obs, _ = env.reset()
    expert.reset()

    fig, (ax_env, ax_info) = plt.subplots(1, 2, figsize=(16, 6),
                                          gridspec_kw={"width_ratios": [3, 1]})
    fig.canvas.manager.set_window_title("Stackelberg Expert — Highway Env")

    ax_env.set_title("highway-env")
    ax_env.axis("off")

    ax_info.set_xlim(0, 10)
    ax_info.set_ylim(0, 30)
    ax_info.axis("off")
    ax_info.set_title("Decision Monitor")

    frame_placeholder = ax_env.imshow(np.zeros((300, 900, 3), dtype=np.uint8))

    texts = {}
    y_positions = list(range(29, 0, -1))
    for i, label in enumerate([
        "FSM State", "Action", "Lateral", "Accel",
        "Cost Orig", "Cost Tgt", "Improve", "HV Style",
        "HV Speed", "TTC", "Gap"
    ]):
        texts[label] = ax_info.text(0.5, y_positions[i], "", transform=ax_info.transData,
                                    fontsize=9, fontfamily="monospace", va="center", ha="left")

    done_flag = [False]
    step_count = [0]
    last_t = [time.time()]
    env_info_holder = [{}]
    info_holder = [{}]

    def update(_frame):
        if done_flag[0]:
            return [frame_placeholder] + list(texts.values())

        dt = time.time() - last_t[0]
        last_t[0] = time.time()
        action, info = expert.decide(env, dt=min(dt, 0.5))
        obs, reward, terminated, truncated, env_info = env.step(action)
        step_count[0] += 1

        if terminated or truncated:
            done_flag[0] = True

        env_info_holder[0] = env_info or {}
        info_holder[0] = info

        # Render frame
        frame = env.render()
        frame_placeholder.set_data(frame)

        # Update text panel
        fsm = info.get("fsm_state", "?")
        lat = {-1: "LEFT", 0: "STAY", 1: "RIGHT"}.get(info.get("game_lateral", 0), "?")
        ttc = info.get("min_ttc", float("inf"))
        ttc_s = f"{ttc:.1f}" if np.isfinite(ttc) else "inf"

        values = {
            "FSM State": fsm,
            "Action": info.get("action_label", "?"),
            "Lateral": lat,
            "Accel": f"{info.get('game_optimal_accel', 0):+.2f}",
            "Cost Orig": f"{info.get('game_cost_original', 0):.2f}",
            "Cost Tgt": f"{info.get('game_cost_target', 0):.2f}",
            "Improve": f"{info.get('game_cost_improvement', 0):+.2f}",
            "HV Style": info.get("game_hv_style", "?"),
            "HV Speed": f"{info.get('game_hv_speed', 0):.1f}",
            "TTC": ttc_s,
            "Gap": f"{info.get('min_gap', 0):.1f}",
        }

        for label, text_obj in texts.items():
            text_obj.set_text(f"{label}: {values.get(label, '?')}")

        return [frame_placeholder] + list(texts.values())

    ani = FuncAnimation(fig, update, interval=1000 / fps, blit=True, cache_frame_data=False)
    plt.tight_layout()
    plt.show()

    env.close()
    if env_info_holder[0]:
        print(f"Done. Crashed: {env_info_holder[0].get('crashed', False)}")


def run_headless_save(env_config, game_config, output_path, fps=15):
    """Headless run and save frames to video using imageio."""
    try:
        import imageio
    except ImportError:
        print("imageio not installed. Install with: pip install imageio[ffmpeg]")
        print("Falling back to saving individual frames as PNG...")
        _save_frames(env_config, game_config, output_path, fps)
        return

    env = gym.make("highway-fast-v0", config=env_config, render_mode="rgb_array")
    expert = StackelbergExpert(game_config)

    obs, _ = env.reset()
    expert.reset()
    done = False
    truncated = False
    step = 0
    last_time = time.time()
    frames = []

    print(f"Recording {env_config['duration']}s episode...")

    while not (done or truncated):
        dt = time.time() - last_time
        last_time = time.time()
        action, info = expert.decide(env, dt=min(dt, 0.5))
        obs, reward, terminated, truncated, env_info = env.step(action)
        done = terminated or truncated
        step += 1

        frame = env.render()
        frames.append(frame)

    env.close()

    output_path = str(output_path)
    if not output_path.endswith(".mp4"):
        output_path += ".mp4"

    writer = imageio.get_writer(output_path, fps=fps, codec="libx264")
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    print(f"Saved {len(frames)} frames to {output_path}")
    print(f"Crashed: {env_info.get('crashed', False) if env_info else False}")


def _save_frames(env_config, game_config, output_dir, fps=15):
    """Fallback: save individual frames as PNG."""
    out_dir = Path(output_dir)
    if out_dir.suffix:
        out_dir = out_dir.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make("highway-fast-v0", config=env_config, render_mode="rgb_array")
    expert = StackelbergExpert(game_config)

    obs, _ = env.reset()
    expert.reset()
    done = False
    step = 0

    import matplotlib.image as mpimg

    while not done and step < 500:
        action, info = expert.decide(env, dt=0.25)
        obs, reward, terminated, truncated, env_info = env.step(action)
        done = terminated or truncated
        step += 1

        frame = env.render()
        mpimg.imsave(str(out_dir / f"frame_{step:04d}.png"), frame)

    env.close()
    print(f"Saved {step} frames to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Visualize Stackelberg Expert in highway-env")
    parser.add_argument("--render", type=str, default="human",
                        choices=["human", "console", "plot"],
                        help="Render mode: human (pygame window), console (terminal dashboard), plot (matplotlib)")
    parser.add_argument("--save", type=str, default=None, help="Save video to file (requires imageio)")
    parser.add_argument("--vehicles", type=int, default=20, help="Max number of vehicles")
    parser.add_argument("--duration", type=int, default=20, help="Episode duration (seconds)")
    parser.add_argument("--lanes", type=int, default=3, help="Number of lanes")
    parser.add_argument("--density", type=float, default=1.5, help="Traffic density (vehicles/km). 1.5 = moderate, 2.0+ = busy (more lane changes).")
    parser.add_argument("--fps", type=int, default=15, help="Display/recording FPS")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: random)")
    args = parser.parse_args()

    env_config = {
        "observation": {"type": "Kinematics", "vehicles_count": args.vehicles},
        "action": {
            "type": "DiscreteMetaAction",
            "target_speeds": [0, 5, 10, 15, 20, 25, 30],
        },
        "lanes_count": args.lanes,
        "vehicles_count": args.vehicles,
        "vehicles_density": args.density,
        "duration": args.duration,
        "simulation_frequency": 15,
        "policy_frequency": 4,
        "collision_reward": -5.0,
        "normalize_reward": True,
        "offroad_terminal": True,
    }

    game_config = GameConfig()

    if args.save:
        run_headless_save(env_config, game_config, args.save, fps=args.fps)
    elif args.render == "console":
        run_console(env_config, game_config, seed=args.seed)
    elif args.render == "plot":
        run_matplotlib(env_config, game_config, fps=args.fps)
    else:
        run_pygame(env_config, game_config, fps_limit=args.fps, seed=args.seed)


if __name__ == "__main__":
    main()
