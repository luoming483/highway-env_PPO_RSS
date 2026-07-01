"""RSS (Responsibility-Sensitive Safety) shield module.

Action-end RSS safety wrapper for highway-env discrete meta-actions.
Validates PPO-proposed actions against safety envelopes and overrides
unsafe ones.
"""

from .shield import RSSConfig, RSSSafetyWrapper, check_rss_safety

__all__ = ["RSSConfig", "RSSSafetyWrapper", "check_rss_safety"]
