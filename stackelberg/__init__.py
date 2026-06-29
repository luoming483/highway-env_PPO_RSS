"""Stackelberg Game Expert Module for Autonomous Lane-Change Decision-Making.

Implements "研究内容一" from the technical roadmap:
    基于主从博弈与FSM的交互式换道决策方法研究

Architecture:
    Perception → Stackelberg Game Solver (Tactical) → FSM (Execution) → Action

Reference:
    Shi B, Zhai L, Liu C. "Stackelberg Game Based on Trajectory Prediction
    for Lane Change in Mixed Traffic." IEEE Access.

Usage:
    from stackelberg import StackelbergExpert, GameConfig

    expert = StackelbergExpert(GameConfig())
    action, info = expert.decide(env)
"""

from .config import DRIVING_STYLE_WEIGHTS, GameConfig
from .fsm_executor import FSMExecutor, FSMState, RateLimiter
from .game_solver import GameResult, StackelbergSolver
from .stackelberg_expert import StackelbergExpert, StackelbergWrapper
from .trajectory_predictor import (
    TrajectoryPoint,
    VehicleState,
    predict_ev_candidate,
    predict_hv_response,
    predict_trajectory,
)
from .utility_functions import (
    UtilityResult,
    compute_ev_cost,
    compute_hv_utility,
)

__all__ = [
    "GameConfig",
    "DRIVING_STYLE_WEIGHTS",
    "StackelbergExpert",
    "StackelbergWrapper",
    "StackelbergSolver",
    "GameResult",
    "FSMExecutor",
    "FSMState",
    "RateLimiter",
    "VehicleState",
    "TrajectoryPoint",
    "predict_trajectory",
    "predict_hv_response",
    "predict_ev_candidate",
    "compute_hv_utility",
    "compute_ev_cost",
    "UtilityResult",
]
