from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RoundMetric:
    seed: int
    algorithm: str
    round_id: int
    instant_regret: float
    cumulative_regret: float
    runtime_sec: float
    cpu_time_sec: float


@dataclass
class SeedSummary:
    seed: int
    algorithm: str
    final_regret: float
    total_runtime_sec: float
    mean_runtime_sec: float
    total_cpu_time_sec: float
    mean_cpu_time_sec: float
    projection_events: int
    cumulative_suboptimality: float
