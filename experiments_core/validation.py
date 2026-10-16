"""Validation shared by experiment entry points."""
from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any


def finite_float(value: Any, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def finite_tree(value: Any, name: str = "config") -> None:
    """Reject nonfinite numeric inputs before comparisons can silently accept them."""
    if isinstance(value, dict):
        for key, item in value.items():
            finite_tree(item, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            finite_tree(item, f"{name}[{i}]")
    elif isinstance(value, Real) and not isinstance(value, bool) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def parse_grid(raw: str, name: str, *, positive: bool = False, probability: bool = False) -> list[float]:
    values = [finite_float(x.strip(), name, positive=positive, nonnegative=not positive)
              for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError(f"{name} must not be empty")
    if probability and any(x > 1 for x in values):
        raise ValueError(f"{name} values must be in [0, 1]")
    return sorted(set(values))


def value_tag(value: float) -> str:
    """Round-trippable value tags prevent distinct grid points sharing a directory."""
    value = finite_float(value, "grid value", nonnegative=True)
    return (str(int(value)) if value.is_integer() else repr(value)).replace(".", "p")


def target_only(raw: dict[str, Any], target: str) -> dict[str, Any]:
    from copy import deepcopy
    cfg = deepcopy(raw)
    for name in ("corectron", "ons", "ogd", "corectron_rbf", "kons"):
        cfg.setdefault("learners", {}).setdefault(name, {})["enabled"] = name == target
    return cfg
