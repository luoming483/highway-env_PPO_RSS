"""Mixture-of-Experts Hybrid Decision Module for Autonomous Lane-Change.

Scene-adaptive expert selection:
    Perception -> Scene Analyzer -> Gate -> Expert -> RSS Validate -> Action

Three experts:
    Stackelberg  — lane-change decisions (game theory + FSM)
    PPO+RSS      — speed optimization with RSS safety shield
    RSS Emergency — hard safety override (TTC < 3s)

Gate logic (rule-based, from empirical analysis):
    if front_ttc < 3.0s              → RSS Emergency (brake)
    elif lane_change_is_beneficial   → Stackelberg (game-theoretic LC)
    else                             → PPO+RSS (speed optimization)

Usage:
    D:\\anaconda\\envs\\ppo_main\\python.exe moe_hybrid.py
    D:\\anaconda\\envs\\ppo_main\\python.exe moe_hybrid.py --seed 42 --vehicles 20 --duration 30
"""

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO

from config import RSS_CONFIG
from rss import RSSConfig, check_rss_safety
from scene_utils import check_lane_exists, check_ego_blocked, classify_density
from stackelberg.config import GameConfig
from stackelberg.expert import StackelbergExpert
from stackelberg.game_solver import GameResult

# highway-env action constants
ACTION_LEFT = 0
ACTION_IDLE = 1
ACTION_RIGHT = 2
ACTION_FASTER = 3
ACTION_SLOWER = 4

ACTION_NAMES = {0: "LEFT", 1: "IDLE", 2: "RIGHT", 3: "FASTER", 4: "SLOWER"}


@dataclass
class SceneFeatures:
    """Scene characterization extracted from env + game solver output."""
    front_ttc: float                     # TTC to front vehicle in current lane
    front_gap: float                     # gap to front vehicle (m)
    ego_speed: float                     # ego speed (m/s)
    min_ttc: float                       # minimum TTC to any nearby vehicle
    min_gap: float                       # minimum gap to any nearby vehicle
    cost_improvement: float              # from game solver (positive = LC beneficial)
    lateral_choice: int                  # -1=LEFT, 0=STAY, 1=RIGHT
    lc_is_feasible: bool                 # target lane physically accessible
    num_vehicles_nearby: int             # vehicles within nearby_horizon
    density_level: str = "sparse"        # "sparse" / "medium" / "dense"

    @classmethod
    def from_env(cls, env, game_result) -> "SceneFeatures":
        """Extract scene features from highway-env and game solver output."""
        ego = env.unwrapped.vehicle
        road = env.unwrapped.road
        ego_speed = float(ego.speed)
        ego_lane = ego.lane_index

        # Front vehicle in current lane
        front, _ = road.neighbour_vehicles(ego, ego_lane)
        front_ttc = float("inf")
        front_gap = float("inf")
        if front is not None:
            try:
                lane = road.network.get_lane(ego_lane)
                ego_s = float(lane.local_coordinates(ego.position)[0])
                front_s = float(lane.local_coordinates(front.position)[0])
                front_gap = front_s - ego_s
                rel_speed = ego_speed - float(front.speed)
                if front_gap > 0 and rel_speed > 1e-6:
                    front_ttc = front_gap / rel_speed
            except (ValueError, IndexError):
                pass

        # Count nearby vehicles
        all_vehicles = [v for v in road.vehicles if v is not ego]
        nearby_horizon = 50.0
        num_nearby = 0
        min_ttc = float("inf")
        min_gap = float("inf")
        for v in all_vehicles:
            dist = float(np.linalg.norm(np.array(v.position) - np.array(ego.position)))
            if dist < nearby_horizon:
                num_nearby += 1
            if dist < min_gap:
                min_gap = dist

        # TTC scan
        for v in all_vehicles:
            dy = abs(float(v.position[1] - ego.position[1]))
            if dy > 4.0 * 0.65:
                continue
            dx = float(v.position[0] - ego.position[0])
            if dx > 0:
                closing = ego_speed - float(v.speed)
                if closing > 1e-6:
                    min_ttc = min(min_ttc, dx / closing)

        # Density classification from env config
        density = classify_density(env)

        # Lane change feasibility: can we reach the target lane?
        lc_feasible = False
        if game_result.lateral_choice != 0:
            target_lane = (ego_lane[0], ego_lane[1], ego_lane[2] + game_result.lateral_choice)
            lc_feasible = check_lane_exists(road, target_lane)

        return cls(
            front_ttc=float(front_ttc),
            front_gap=float(front_gap),
            ego_speed=ego_speed,
            min_ttc=float(min_ttc),
            min_gap=float(min_gap),
            cost_improvement=float(game_result.cost_improvement),
            lateral_choice=int(game_result.lateral_choice),
            lc_is_feasible=lc_feasible,
            num_vehicles_nearby=num_nearby,
            density_level=density,
        )


@dataclass
class GateDecision:
    """Output of the MoE gating network."""
    expert: str               # "rss_emergency" / "stackelberg" / "ppo_rss"
    reason: str
    confidence: float         # 0.0 - 1.0, how confident is this selection
    scene: SceneFeatures


class MoEGate:
    """Density-adaptive scene gate for expert selection.

    Design philosophy: each expert has a pain point the other solves.
      - PPO maximizes speed but CANNOT lane-change → gets stuck behind slow vehicles.
      - Stackelberg executes game-theoretic LCs but drives conservatively (FSM braking).

    The gate fuses them by:
      1. Using PPO for speed in free-flow (default, ~85% of time).
      2. Activating Stackelberg when PPO is blocked (blocked-rescue mode).
      3. Activating Stackelberg when game solver finds a beneficial LC.
      4. Adapting cost-improvement thresholds to traffic density.

    Three-tier priority:
      1. RSS Emergency  — TTC < 3s, immediate braking.
      2. Stackelberg    — solver recommends LC, OR ego persistently blocked.
      3. PPO+RSS        — default speed optimization with RSS safety.
    """

    # Density-dependent cost thresholds: denser → easier to activate Stackelberg
    COST_THRESHOLDS = {"sparse": 0.20, "medium": 0.10, "dense": 0.05}

    def __init__(
        self,
        ttc_emergency: float = 3.0,
        cost_improvement_threshold: float = 0.10,
        min_speed_for_lc: float = 5.0,
    ):
        self.ttc_emergency = ttc_emergency
        self.cost_improvement_threshold = cost_improvement_threshold
        self.min_speed_for_lc = min_speed_for_lc
        self._prev_expert: str = "ppo_rss"
        self._stackelberg_streak: int = 0  # hysteresis counter
        self._stackelberg_lc_count: int = 0  # LCs executed in current streak
        self._rescue_cooldown: int = 0  # cooldown after failed blocked-rescue

    def feedback(self, action: int, expert: str):
        """Called after action execution; tracks whether Stackelberg actually
        performed a lane change during the current streak."""
        if expert == "stackelberg" and action in (ACTION_LEFT, ACTION_RIGHT):
            self._stackelberg_lc_count += 1

    def select(self, scene: SceneFeatures, is_blocked: bool = False) -> GateDecision:
        """Select the best expert for the current scene.

        Args:
            scene: SceneFeatures from env + game solver.
            is_blocked: True if ego has been persistently blocked for
                        several consecutive steps.
        """
        # ---- Tier 1: RSS Emergency ----
        if scene.front_ttc < self.ttc_emergency:
            self._stackelberg_streak = 0
            return GateDecision(
                expert="rss_emergency",
                reason=f"front_ttc={scene.front_ttc:.1f}s < {self.ttc_emergency}s",
                confidence=min(1.0, (self.ttc_emergency - scene.front_ttc) / self.ttc_emergency + 0.5),
                scene=scene,
            )
        if scene.min_ttc < self.ttc_emergency:
            self._stackelberg_streak = 0
            return GateDecision(
                expert="rss_emergency",
                reason=f"min_ttc={scene.min_ttc:.1f}s < {self.ttc_emergency}s (lateral threat)",
                confidence=min(1.0, (self.ttc_emergency - scene.min_ttc) / self.ttc_emergency + 0.5),
                scene=scene,
            )

        # ---- Tier 2: Stackelberg Activation ----
        density = scene.density_level
        cost_threshold = self.COST_THRESHOLDS.get(density, self.cost_improvement_threshold)

        # Safety precondition (FSM adds its own stricter checks)
        safe_for_lc = (
            scene.front_ttc > self.ttc_emergency
            and scene.ego_speed > self.min_speed_for_lc
        )

        if not safe_for_lc:
            self._stackelberg_streak = 0
            self._prev_expert = "ppo_rss"
            return GateDecision(
                expert="ppo_rss",
                reason=f"speed optimization: density={density}, speed={scene.ego_speed:.1f}m/s",
                confidence=0.8,
                scene=scene,
            )

        # Condition A: Solver recommends lane change
        solver_recommends = (
            scene.lateral_choice != 0
            and scene.lc_is_feasible
            and scene.cost_improvement > cost_threshold
        )

        # Condition B: Blocked rescue — PPO is stuck behind slower vehicle.
        # Force Stackelberg to attempt a game-theoretic lane change.
        # Skip if still in cooldown from a previous failed rescue.
        blocked_rescue = (
            is_blocked
            and not solver_recommends
            and density != "sparse"
            and self._rescue_cooldown <= 0
        )

        # Hysteresis: keep Stackelberg active after initial selection to
        # avoid flickering. Default 8-step window gives FSM time to find a
        # safe gap and execute the lane change maneuver.
        hysteresis = (
            self._prev_expert == "stackelberg"
            and self._stackelberg_streak < 8
        )

        if solver_recommends or blocked_rescue or hysteresis:
            if solver_recommends:
                # Reset rescue cooldown on proactive solver LC
                self._rescue_cooldown = 0
                reason = (
                    f"solver LC: cost_imp={scene.cost_improvement:.2f}, "
                    f"lat={scene.lateral_choice}, ttc={scene.front_ttc:.1f}s, "
                    f"density={density}"
                )
            elif blocked_rescue:
                reason = (
                    f"blocked rescue: ego blocked, "
                    f"density={density}, forcing Stackelberg LC attempt"
                )
            else:
                reason = (
                    f"hysteresis: continuing Stackelberg streak={self._stackelberg_streak}, "
                    f"lcs={self._stackelberg_lc_count}, density={density}"
                )

            self._stackelberg_streak += 1
            self._prev_expert = "stackelberg"
            return GateDecision(
                expert="stackelberg",
                reason=reason,
                confidence=0.7 if blocked_rescue else 0.85,
                scene=scene,
            )

        # ---- Tier 3: PPO+RSS Speed Optimization ----
        # If Stackelberg streak ended without any LC (failed rescue),
        # brief cooldown to prevent instant re-trigger oscillation.
        if self._prev_expert == "stackelberg" and self._stackelberg_lc_count == 0:
            self._rescue_cooldown = 3
        elif self._rescue_cooldown > 0:
            self._rescue_cooldown -= 1

        self._stackelberg_streak = 0
        self._stackelberg_lc_count = 0
        self._prev_expert = "ppo_rss"
        return GateDecision(
            expert="ppo_rss",
            reason=(
                f"speed optimization: "
                f"density={density}, "
                f"front_ttc={scene.front_ttc:.1f}s, "
                f"speed={scene.ego_speed:.1f}m/s"
            ),
            confidence=0.8,
            scene=scene,
        )


class HybridExpert:
    """Mixture-of-Experts decision module combining three experts.

    Pipeline:
        1. Stackelberg game solver → scene understanding
        2. MoE Gate → select expert based on scene features
        3. Selected expert → produce action
        4. (env's RSS wrapper provides final safety net)

    Usage:
        expert = HybridExpert(ppo_model_path="runs/.../final_model.zip")
        obs, _ = env.reset()
        action, info = expert.decide(env, obs)
    """

    def __init__(
        self,
        ppo_model_path: str,
        game_config: Optional[GameConfig] = None,
        gate: Optional[MoEGate] = None,
        rss_config: Optional[RSSConfig] = None,
        device: str = "cpu",
    ):
        self.game_config = game_config or GameConfig()
        self.gate = gate or MoEGate()
        self.rss_config = rss_config or RSSConfig(**RSS_CONFIG)

        # Initialize Stackelberg expert
        self.stackelberg = StackelbergExpert(self.game_config)

        # Load PPO model
        self.ppo_model = PPO.load(ppo_model_path, device=device)

        # Statistics
        self._expert_counts = {"rss_emergency": 0, "stackelberg": 0, "ppo_rss": 0}
        self._total_decisions = 0

        # Blocked persistence tracking
        self._blocked_steps: int = 0
        self._blocked_threshold: int = 3  # consecutive steps before gate relaxes

    def _check_ego_blocked(self, env) -> bool:
        blocked, _, _ = check_ego_blocked(env)
        return blocked

    def _build_rescue_game_result(self, env, original: "GameResult") -> "GameResult":
        """Build a GameResult that tells FSM to try a lane change in the
        direction with more space. Used when gate triggers blocked-rescue
        but the solver returned lateral_choice=0."""
        ego = env.unwrapped.vehicle
        ego_lane = ego.lane_index
        road = env.unwrapped.road
        num_lanes = len(road.network.graph[ego_lane[0]][ego_lane[1]])

        # Check left and right lanes for available space
        best_lateral = 0
        best_gap = -1.0

        for direction in [-1, 1]:  # LEFT, RIGHT
            target_lane_id = ego_lane[2] + direction
            if 0 <= target_lane_id < num_lanes:
                target_lane = (ego_lane[0], ego_lane[1], target_lane_id)
                front, _ = road.neighbour_vehicles(ego, target_lane)
                if front is not None:
                    try:
                        lane = road.network.get_lane(target_lane)
                        ego_s = float(lane.local_coordinates(ego.position)[0])
                        front_s = float(lane.local_coordinates(front.position)[0])
                        gap = front_s - ego_s
                        if gap > best_gap:
                            best_gap = gap
                            best_lateral = direction
                    except (ValueError, IndexError):
                        pass
                else:
                    # No front vehicle — large effective gap
                    if 200.0 > best_gap:
                        best_gap = 200.0
                        best_lateral = direction

        return GameResult(
            action=0 if best_lateral == -1 else 2 if best_lateral == 1 else 1,
            lateral_choice=best_lateral,
            optimal_accel=original.optimal_accel,
            ev_cost_original_lane=original.ev_cost_original_lane,
            ev_cost_target_lane=original.ev_cost_target_lane,
            cost_improvement=original.cost_improvement,
            hv_driving_style=original.hv_driving_style,
            hv_predicted_speed=original.hv_predicted_speed,
            min_ttc=original.min_ttc,
            min_gap=original.min_gap,
            game_success=best_lateral != 0,
            candidates_evaluated=original.candidates_evaluated,
        )

    def decide(
        self,
        env: gym.Env,
        obs: np.ndarray,
        dt: float = 0.25,
    ) -> Tuple[int, Dict[str, Any]]:
        """Make a decision using the best expert for the current scene.

        Args:
            env: highway-env environment (.unwrapped must be accessible).
            obs: Flattened observation for PPO inference.
            dt: Policy time step (0.25s at 4Hz).

        Returns:
            (action, info_dict) where info_dict contains gate decision and
            expert diagnostics.
        """
        self._total_decisions += 1

        # Step 1: Run Stackelberg game solver (lightweight, ~18 candidates)
        game_result = self.stackelberg.solver.solve(env)

        # Step 2: Extract scene features
        scene = SceneFeatures.from_env(env, game_result)

        # Step 3: Track blocked persistence for gate awareness
        if self._check_ego_blocked(env):
            self._blocked_steps += 1
        else:
            self._blocked_steps = 0
        is_persistently_blocked = self._blocked_steps >= self._blocked_threshold

        # Step 4: Gate selects expert (blocked-aware)
        decision = self.gate.select(scene, is_blocked=is_persistently_blocked)
        self._expert_counts[decision.expert] += 1

        # Step 5: Execute selected expert
        if decision.expert == "rss_emergency":
            action = ACTION_SLOWER
            expert_info = {}
        elif decision.expert == "stackelberg":
            # In blocked-rescue mode, the solver returned lat=0 (no LC needed)
            # but the gate overrides because PPO is stuck. Build a rescue
            # GameResult that tells the FSM which direction to try.
            if "blocked rescue" in decision.reason:
                fsm_input = self._build_rescue_game_result(env, game_result)
            else:
                fsm_input = game_result

            action, fsm_info = self.stackelberg.fsm.process(fsm_input, env, dt)
            expert_info = {
                "fsm_state": str(fsm_info.state) if hasattr(fsm_info, 'state') else "",
                "fsm_reason": fsm_info.reason if hasattr(fsm_info, 'reason') else "",
            }
            self.gate.feedback(action, "stackelberg")
            self.stackelberg._step_count += 1
        else:  # ppo_rss
            raw_action, _ = self.ppo_model.predict(obs, deterministic=True)
            if isinstance(raw_action, np.ndarray):
                raw_action = int(raw_action.item())
            else:
                raw_action = int(raw_action)
            # RSS safety check for PPO (no FSM protection)
            unsafe, reason = check_rss_safety(env, raw_action, self.rss_config)
            if unsafe:
                action = ACTION_SLOWER
            else:
                action = raw_action
            expert_info = {"rss_check": "passed" if not unsafe else f"overridden: {reason}"}

        # Build diagnostic info
        info = {
            "moe_expert": decision.expert,
            "moe_reason": decision.reason,
            "moe_confidence": decision.confidence,
            "scene_density": scene.density_level,
            "scene_front_ttc": scene.front_ttc,
            "scene_front_gap": scene.front_gap,
            "scene_min_ttc": scene.min_ttc,
            "scene_ego_speed": scene.ego_speed,
            "game_cost_improvement": scene.cost_improvement,
            "game_lateral_choice": scene.lateral_choice,
            "game_min_ttc": game_result.min_ttc,
            "game_hv_style": game_result.hv_driving_style,
            "blocked_steps": self._blocked_steps,
            "blocked_aware": is_persistently_blocked,
        }

        # Merge stackelberg-specific info if available
        if decision.expert == "stackelberg" and expert_info:
            info["fsm_state"] = expert_info.get("fsm_state", "")
            info["fsm_reason"] = expert_info.get("fsm_reason", "")

        return action, info

    def reset(self) -> None:
        self.stackelberg.reset()
        self.gate._prev_expert = "ppo_rss"
        self.gate._stackelberg_streak = 0
        self.gate._stackelberg_lc_count = 0
        self.gate._rescue_cooldown = 0
        self._blocked_steps = 0

    @property
    def expert_distribution(self) -> Dict[str, float]:
        total = max(self._total_decisions, 1)
        return {k: v / total for k, v in self._expert_counts.items()}


# ============================================================
# Environment factory
# ============================================================
def make_env(
    vehicles: int = 20,
    duration: int = 30,
    density: float = 1.0,
    seed: Optional[int] = None,
    render: bool = False,
) -> Tuple[gym.Env, gym.Env]:
    """Create env pair for HybridExpert.

    NOTE: vehicles_count must match the PPO model's training config (20 vehicles
    = 140-dim obs).

    Returns:
        (base_env, flat_env): Raw highway env and FlattenObservation env.
        Use base_env for Stackelberg (FSM provides safety).
        Use flat_env for PPO inference (RSS checked inside HybridExpert.decide).
    """
    PPO_VEHICLES = 20  # PPO was trained with 20 vehicles
    if vehicles != PPO_VEHICLES:
        print(f"[WARN] PPO model was trained with {PPO_VEHICLES} vehicles, "
              f"but {vehicles} requested. Using {PPO_VEHICLES} for PPO obs compatibility.")
        vehicles = PPO_VEHICLES

    env_config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": vehicles,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "absolute": False,
        },
        "action": {
            "type": "DiscreteMetaAction",
            "target_speeds": [0, 5, 10, 15, 20, 25, 30],
        },
        "lanes_count": 4,
        "vehicles_count": vehicles,
        "vehicles_density": density,
        "duration": duration,
        "simulation_frequency": 8,
        "policy_frequency": 4,
        "collision_reward": -5.0,
        "normalize_reward": True,
        "offroad_terminal": True,
    }

    render_mode = "human" if render else None
    base_env = gym.make("highway-fast-v0", config=env_config, render_mode=render_mode)

    flat_env = FlattenObservation(base_env)
    return base_env, flat_env


# ============================================================
# Auto-detect PPO model
# ============================================================
def _find_ppo_model(seed: int = 42, experiment: str = "our_method") -> str:
    candidates = [
        Path(f"runs/20260615_163841/models/{experiment}_seed{seed}/final_model.zip"),
        Path(f"results/models/{experiment}_seed{seed}/final_model.zip"),
    ]
    for p in candidates:
        if p.exists():
            return str(p.resolve())
    raise FileNotFoundError(f"No PPO model for seed={seed}, experiment={experiment}")


# ============================================================
# Diagnostic output
# ============================================================
C_RESET = "\033[0m"
C_RED = "\033[91m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_CYAN = "\033[96m"
C_BOLD = "\033[1m"

EXPERT_COLORS = {
    "rss_emergency": C_RED,
    "stackelberg": C_CYAN,
    "ppo_rss": C_GREEN,
}


def run_interactive(env, flat_env, hybrid, fps: int = 15, seed: int = None):
    """Run hybrid expert with pygame visualization and terminal diagnostics."""
    obs, _ = flat_env.reset(seed=seed)
    hybrid.reset()
    done = False
    step = 0
    frame_delay = 1.0 / fps

    print("=" * 80)
    print("MoE Hybrid Expert — Interactive Diagnostics")
    print("=" * 80)
    print(f"Gate config: emergency_ttc=3.0s, cost_improvement_threshold=0.10")
    print()
    print(f"{'Step':>5s} {'Expert':>16s} {'Action':>8s} {'Speed':>7s} "
          f"{'FrontTTC':>9s} {'CostImp':>8s} {'Reason'}")
    print("-" * 105)

    try:
        while not done:
            action, info = hybrid.decide(env, obs, dt=0.25)
            obs, reward, terminated, truncated, env_info = flat_env.step(action)
            done = terminated or truncated
            step += 1

            expert = info["moe_expert"]
            color = EXPERT_COLORS.get(expert, C_RESET)
            speed = info["scene_ego_speed"]
            front_ttc = info["scene_front_ttc"]
            cost_imp = info["game_cost_improvement"]

            ttc_s = f"{front_ttc:.1f}s" if np.isfinite(front_ttc) else "inf"
            crash_marker = f" {C_RED}*** CRASH ***{C_RESET}" if env_info.get("crashed", False) else ""

            print(f"{step:5d} {color}{expert:>16s}{C_RESET} {ACTION_NAMES.get(action, '?'):>8s} "
                  f"{speed:6.1f}m/s {ttc_s:>9s} {cost_imp:8.3f} "
                  f"{info['moe_reason'][:50]}{crash_marker}")

            time.sleep(frame_delay)
    except KeyboardInterrupt:
        pass

    flat_env.close()
    crashed = env_info.get("crashed", False) if 'env_info' in dir() else False
    dist = hybrid.expert_distribution
    print(f"\nDone. Steps: {step}, Crashed: {crashed}")
    print(f"Expert distribution: "
          f"Stackelberg={dist['stackelberg']:.0%}, "
          f"PPO+RSS={dist['ppo_rss']:.0%}, "
          f"RSS_Emergency={dist['rss_emergency']:.0%}")


def run_batch_eval(
    env_factory,
    hybrid_factory,
    seeds=(42, 123, 456, 789),
    steps_per_ep=200,
):
    """Batch evaluation across multiple seeds with full metrics."""
    results = []
    for seed in seeds:
        env, flat_env = env_factory(seed=seed)
        hybrid = hybrid_factory()
        obs, _ = flat_env.reset(seed=seed)
        hybrid.reset()
        done = False
        total_reward = 0.0
        total_steps = 0
        crashed = False
        expert_counts = {"rss_emergency": 0, "stackelberg": 0, "ppo_rss": 0}
        actions = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        speeds = []

        for _ in range(steps_per_ep):
            action, info = hybrid.decide(env, obs)
            obs, reward, terminated, truncated, env_info = flat_env.step(action)
            total_reward += float(reward)
            total_steps += 1
            actions[action] = actions.get(action, 0) + 1
            speeds.append(info["scene_ego_speed"])
            expert_counts[info["moe_expert"]] += 1

            if env_info.get("crashed", False):
                crashed = True
            if terminated or truncated:
                break

        flat_env.close()
        results.append({
            "seed": seed,
            "steps": total_steps,
            "reward": total_reward,
            "avg_speed": float(np.mean(speeds)) if speeds else 0.0,
            "crashed": crashed,
            "expert_dist": {k: v / max(total_steps, 1) for k, v in expert_counts.items()},
            "actions": actions,
        })

    return results


def print_batch_summary(results):
    print("\n" + "=" * 80)
    print("MoE Hybrid Expert — Batch Evaluation")
    print("=" * 80)
    print(f"{'Seed':>5s} {'Steps':>6s} {'Speed':>7s} {'Crashed':>8s} "
          f"{'Stack':>7s} {'PPO':>7s} {'RSS':>7s} {'Reward':>8s}")
    print("-" * 65)

    for r in results:
        d = r["expert_dist"]
        print(f"{r['seed']:5d} {r['steps']:6d} {r['avg_speed']:6.1f}m/s "
              f"{str(r['crashed']):>8s} "
              f"{d['stackelberg']:6.0%} {d['ppo_rss']:6.0%} {d['rss_emergency']:6.0%} "
              f"{r['reward']:8.1f}")

    speeds = [r["avg_speed"] for r in results]
    crashes = sum(1 for r in results if r["crashed"])
    stack_usage = np.mean([r["expert_dist"]["stackelberg"] for r in results])
    ppo_usage = np.mean([r["expert_dist"]["ppo_rss"] for r in results])
    rss_usage = np.mean([r["expert_dist"]["rss_emergency"] for r in results])

    print("-" * 65)
    print(f"Avg speed: {np.mean(speeds):.1f} m/s  |  "
          f"Crashes: {crashes}/{len(results)}  |  "
          f"Stackelberg: {stack_usage:.0%}  PPO+RSS: {ppo_usage:.0%}  RSS: {rss_usage:.0%}")


# ============================================================
def main():
    parser = argparse.ArgumentParser(description="MoE Hybrid Expert")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to PPO model .zip (auto-detect if omitted)")
    parser.add_argument("--seed", type=int, default=42,
                        help="PPO model seed (default: 42)")
    parser.add_argument("--experiment", type=str, default="our_method",
                        help="Experiment name for model lookup")
    parser.add_argument("--vehicles", type=int, default=20,
                        help="Number of vehicles")
    parser.add_argument("--duration", type=int, default=30,
                        help="Episode duration (s)")
    parser.add_argument("--density", type=float, default=1.0,
                        help="Vehicle density")
    parser.add_argument("--env-seed", type=int, default=None,
                        help="Environment random seed (default: use --seed)")
    parser.add_argument("--batch", action="store_true",
                        help="Run batch evaluation across 4 seeds")
    parser.add_argument("--fps", type=int, default=15,
                        help="Display FPS for interactive mode")
    args = parser.parse_args()

    # Resolve model path
    model_path = args.model or _find_ppo_model(args.seed, args.experiment)
    print(f"PPO model: {model_path}")

    hybrid = HybridExpert(ppo_model_path=model_path)

    if args.batch:
        env_seeds = [42, 123, 456, 789]

        def make_env_fn(seed):
            return make_env(
                vehicles=args.vehicles,
                duration=args.duration,
                density=args.density,
                seed=seed,
                render=False,
            )

        def make_hybrid_fn():
            return HybridExpert(ppo_model_path=model_path)

        results = run_batch_eval(make_env_fn, make_hybrid_fn, seeds=env_seeds)
        print_batch_summary(results)
    else:
        env_seed = args.env_seed if args.env_seed is not None else args.seed
        base_env, flat_env = make_env(
            vehicles=args.vehicles,
            duration=args.duration,
            density=args.density,
            seed=env_seed,
            render=True,
        )
        run_interactive(base_env, flat_env, hybrid, fps=args.fps, seed=env_seed)


if __name__ == "__main__":
    main()
