"""Top-level Stackelberg Game + FSM expert module.

Integrates the Stackelberg game solver (tactical layer) with the FSM
executor (execution layer) to produce safe, stable lane-change decisions.

Usage (standalone):
    from stackelberg import StackelbergExpert, GameConfig
    expert = StackelbergExpert(GameConfig())
    action, info = expert.decide(env)

Usage (as wrapper, similar to RSSSafetyWrapper):
    env = gym.make("highway-fast-v0")
    expert = StackelbergExpert(GameConfig())
    obs, _ = env.reset()
    action, info = expert.decide(env)
    obs, reward, terminated, truncated, info = env.step(action)
"""

import time
from typing import Any, Dict, Optional, Tuple

from .config import GameConfig
from .fsm_executor import ACTION_IDLE, FSMExecutor, FSMState
from .game_solver import GameResult, StackelbergSolver


class StackelbergExpert:
    """Stackelberg game-theoretic lane-change decision expert.

    Decision pipeline:
        Perception → Stackelberg Game Solver → FSM Governance → Action

    The game solver (tactical layer) evaluates candidate actions through
    leader-follower equilibrium, while the FSM (execution layer) enforces
    temporal stability, safety gating, and command smoothing.
    """

    def __init__(self, config: Optional[GameConfig] = None):
        self.config = config or GameConfig()
        self.solver = StackelbergSolver(self.config)
        self.fsm = FSMExecutor(self.config)
        self._last_decision_time: float = 0.0
        self._step_count: int = 0
        self._decision_history: list = []

    def decide(self, env, dt: Optional[float] = None) -> Tuple[int, Dict[str, Any]]:
        """Produce a lane-change decision for the current environment state.

        Args:
            env: highway-env environment (unwrapped reference accessible).
            dt: Time step override (default: computed from wall clock).

        Returns:
            (action, info_dict) where action is an int in {0,1,2,3,4}
            matching highway-env DiscreteMetaAction.
        """
        if dt is None:
            now = time.time()
            dt = now - self._last_decision_time if self._last_decision_time > 0 else 0.25
            self._last_decision_time = now

        self._step_count += 1

        # ---- Phase 1: Game-theoretic tactical reasoning ----
        game_result = self.solver.solve(env)

        # ---- Phase 2: FSM execution governance ----
        action, fsm_info = self.fsm.process(game_result, env, dt)

        # ---- Build diagnostic info ----
        info = self._build_info(game_result, fsm_info, dt)

        self._decision_history.append({
            "step": self._step_count,
            "action": action,
            "fsm_state": fsm_info.state,
            "game_lateral": game_result.lateral_choice,
        })

        return action, info

    def _build_info(self, game: GameResult, fsm_info, dt: float) -> Dict[str, Any]:
        """Aggregate game + FSM diagnostics."""
        return {
            # Action
            "action": fsm_info.action,
            "action_label": {0: "LEFT", 1: "IDLE", 2: "RIGHT", 3: "FASTER", 4: "SLOWER"}.get(
                fsm_info.action, "UNKNOWN"
            ),

            # FSM state
            "fsm_state": fsm_info.state,
            "fsm_time_in_state": fsm_info.time_in_state,
            "fsm_cooldown_remaining": fsm_info.cooldown_remaining,
            "fsm_intervened": fsm_info.intervened,
            "fsm_reason": fsm_info.reason,
            "fsm_game_overridden": fsm_info.game_action_overridden,

            # Game results
            "game_lateral": game.lateral_choice,
            "game_optimal_accel": game.optimal_accel,
            "game_cost_original": game.ev_cost_original_lane,
            "game_cost_target": game.ev_cost_target_lane,
            "game_cost_improvement": game.cost_improvement,
            "game_hv_style": game.hv_driving_style,
            "game_hv_speed": game.hv_predicted_speed,
            "game_success": game.game_success,
            "game_candidates": game.candidates_evaluated,

            # Safety
            "min_ttc": game.min_ttc,
            "min_gap": game.min_gap,
        }

    def reset(self) -> None:
        """Reset internal state (call at episode start)."""
        self.fsm.reset()
        self._last_decision_time = 0.0
        self._step_count = 0
        self._decision_history = []

    @property
    def fsm_state(self) -> FSMState:
        return self.fsm.state

    @property
    def decision_history(self) -> list:
        return self._decision_history


class StackelbergWrapper:
    """Gymnasium wrapper that replaces policy with Stackelberg expert.

    This wraps an environment so that step() calls go through the
    Stackelberg expert, useful for evaluating the game theory expert
    as a standalone baseline.
    """

    def __init__(self, env, config: Optional[GameConfig] = None):
        self.env = env
        self.expert = StackelbergExpert(config)
        self._last_info: Dict[str, Any] = {}

    def reset(self, **kwargs):
        self.expert.reset()
        return self.env.reset(**kwargs)

    def step(self, _action=None):
        """Ignore input action, use Stackelberg expert instead."""
        action, info = self.expert.decide(self.env)
        self._last_info = info
        obs, reward, terminated, truncated, env_info = self.env.step(action)
        if env_info is None:
            env_info = {}
        env_info["stackelberg"] = info
        return obs, reward, terminated, truncated, env_info

    def __getattr__(self, name):
        return getattr(self.env, name)
