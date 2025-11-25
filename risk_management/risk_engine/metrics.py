"""Lightweight metrics registry for the risk engine."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Mapping, MutableMapping, Tuple


@dataclass
class MetricRegistry:
    """A minimal Prometheus-style collector used for in-process accounting."""

    counters: MutableMapping[Tuple[str, Tuple[Tuple[str, str], ...]], float] = field(
        default_factory=lambda: defaultdict(float)
    )
    histograms: MutableMapping[Tuple[str, Tuple[Tuple[str, str], ...]], list] = field(
        default_factory=lambda: defaultdict(list)
    )

    def inc(self, name: str, *, labels: Mapping[str, str] | None = None, amount: float = 1.0) -> None:
        key = self._key(name, labels)
        self.counters[key] += amount

    def observe(self, name: str, value: float, *, labels: Mapping[str, str] | None = None) -> None:
        key = self._key(name, labels)
        self.histograms[key].append(value)

    def _key(self, name: str, labels: Mapping[str, str] | None) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
        sorted_labels = tuple(sorted((labels or {}).items()))
        return name, sorted_labels


class Timer:
    """Context manager to record elapsed time into a histogram."""

    def __init__(self, registry: MetricRegistry, name: str, *, labels: Mapping[str, str] | None = None) -> None:
        self._registry = registry
        self._name = name
        self._labels = labels
        self._start: float | None = None

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._start is None:
            return
        duration = time.perf_counter() - self._start
        self._registry.observe(self._name, duration, labels=self._labels)
