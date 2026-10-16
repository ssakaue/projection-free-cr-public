from .base import ActionSetPlugin, ResidualBounds
from .m_out_of_n import MOutOfNActionSet


def build_action_set(action_dim: int, m_select: int, scale: float = 1.0) -> ActionSetPlugin:
    return MOutOfNActionSet(action_dim=action_dim, m_select=m_select, scale=scale)


__all__ = [
    "ActionSetPlugin",
    "ResidualBounds",
    "MOutOfNActionSet",
    "build_action_set",
]
