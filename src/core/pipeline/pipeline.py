"""
Composable pipeline framework.

A Pipeline connects a source (any iterable or generator function) through a
chain of stages (filter/map/take/...) to sinks (collector, file, statistics).
Evaluation is lazy: values flow through Python generators one at a time.
"""

import math
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional


# ---- Stages ----

class PipelineStage(ABC):
    """Abstract base for pipeline stages. A stage transforms an iterator."""

    @abstractmethod
    def process(self, stream: Iterator[Any]) -> Iterator[Any]:
        """Transform an input iterator into an output iterator (lazy)."""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"


class FilterStage(PipelineStage):
    """Keep only values for which the predicate is true."""

    def __init__(self, predicate: Callable[[Any], bool], name: str = "filter"):
        self.predicate = predicate
        self.name = name

    def process(self, stream: Iterator[Any]) -> Iterator[Any]:
        return (x for x in stream if self.predicate(x))


class MapStage(PipelineStage):
    """Apply a function to every value."""

    def __init__(self, func: Callable[[Any], Any], name: str = "map"):
        self.func = func
        self.name = name

    def process(self, stream: Iterator[Any]) -> Iterator[Any]:
        return (self.func(x) for x in stream)


class TakeStage(PipelineStage):
    """Take at most n values, then stop pulling from the source."""

    def __init__(self, n: int):
        if n < 0:
            raise ValueError("n must be non-negative")
        self.n = n

    def process(self, stream: Iterator[Any]) -> Iterator[Any]:
        def gen():
            for i, x in enumerate(stream):
                if i >= self.n:
                    return
                yield x
        return gen()


class SkipStage(PipelineStage):
    """Skip the first n values."""

    def __init__(self, n: int):
        if n < 0:
            raise ValueError("n must be non-negative")
        self.n = n

    def process(self, stream: Iterator[Any]) -> Iterator[Any]:
        def gen():
            for i, x in enumerate(stream):
                if i >= self.n:
                    yield x
        return gen()


class DistinctStage(PipelineStage):
    """Drop duplicate values (keeps first occurrence)."""

    def process(self, stream: Iterator[Any]) -> Iterator[Any]:
        def gen():
            seen = set()
            for x in stream:
                if x not in seen:
                    seen.add(x)
                    yield x
        return gen()


# ---- Sinks ----

class CollectorSink:
    """Collects all values into a list."""

    def __init__(self):
        self.items: List[Any] = []

    def consume(self, stream: Iterator[Any]) -> List[Any]:
        self.items = list(stream)
        return self.items


class FileSink:
    """Writes values to a file, one per line."""

    def __init__(self, path: str):
        self.path = path
        self.count = 0

    def consume(self, stream: Iterator[Any]) -> int:
        self.count = 0
        with open(self.path, "w") as f:
            for x in stream:
                f.write(f"{x}\n")
                self.count += 1
        return self.count


class StatsSink:
    """Computes running statistics over consumed values (single pass)."""

    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.sum_sq = 0.0
        self.minimum: Optional[float] = None
        self.maximum: Optional[float] = None

    def consume(self, stream: Iterator[Any]) -> Dict[str, Any]:
        for x in stream:
            v = float(x)
            self.count += 1
            self.total += v
            self.sum_sq += v * v
            self.minimum = v if self.minimum is None else min(self.minimum, v)
            self.maximum = v if self.maximum is None else max(self.maximum, v)
        return self.report()

    def report(self) -> Dict[str, Any]:
        if self.count == 0:
            return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
        mean = self.total / self.count
        variance = max(0.0, self.sum_sq / self.count - mean * mean)
        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.minimum,
            "max": self.maximum,
        }


# ---- Pipeline ----

class Pipeline:
    """A source plus an ordered list of stages; lazily evaluated."""

    def __init__(self, source: Iterable[Any], stages: List[PipelineStage] = None):
        self._source = source
        self.stages = list(stages or [])

    def _stream(self) -> Iterator[Any]:
        stream = iter(self._source() if callable(self._source) else self._source)
        for stage in self.stages:
            stream = stage.process(stream)
        return stream

    def __iter__(self) -> Iterator[Any]:
        return self._stream()

    # Fluent chaining (returns new Pipeline; original unchanged)
    def add_stage(self, stage: PipelineStage) -> "Pipeline":
        return Pipeline(self._source, self.stages + [stage])

    def filter(self, predicate: Callable[[Any], bool]) -> "Pipeline":
        return self.add_stage(FilterStage(predicate))

    def map(self, func: Callable[[Any], Any]) -> "Pipeline":
        return self.add_stage(MapStage(func))

    def take(self, n: int) -> "Pipeline":
        return self.add_stage(TakeStage(n))

    def skip(self, n: int) -> "Pipeline":
        return self.add_stage(SkipStage(n))

    def distinct(self) -> "Pipeline":
        return self.add_stage(DistinctStage())

    # Terminal operations
    def collect(self) -> List[Any]:
        return CollectorSink().consume(self._stream())

    def to_file(self, path: str) -> int:
        return FileSink(path).consume(self._stream())

    def stats(self) -> Dict[str, Any]:
        return StatsSink().consume(self._stream())

    def first(self) -> Any:
        for x in self._stream():
            return x
        raise ValueError("Pipeline produced no values")

    def count(self) -> int:
        return sum(1 for _ in self._stream())


class PipelineBuilder:
    """Builder for pipelines: set a source, append stages, then build()."""

    def __init__(self):
        self._source: Optional[Iterable[Any]] = None
        self._stages: List[PipelineStage] = []

    def source(self, src: Iterable[Any]) -> "PipelineBuilder":
        self._source = src
        return self

    def stage(self, stage: PipelineStage) -> "PipelineBuilder":
        self._stages.append(stage)
        return self

    def filter(self, predicate: Callable[[Any], bool]) -> "PipelineBuilder":
        return self.stage(FilterStage(predicate))

    def map(self, func: Callable[[Any], Any]) -> "PipelineBuilder":
        return self.stage(MapStage(func))

    def take(self, n: int) -> "PipelineBuilder":
        return self.stage(TakeStage(n))

    def skip(self, n: int) -> "PipelineBuilder":
        return self.stage(SkipStage(n))

    def distinct(self) -> "PipelineBuilder":
        return self.stage(DistinctStage())

    def build(self) -> Pipeline:
        if self._source is None:
            raise ValueError("Pipeline requires a source")
        return Pipeline(self._source, self._stages)


# ---- Example pipeline factories ----

def _naturals() -> Iterator[int]:
    n = 0
    while True:
        yield n
        n += 1


def prime_pipeline(count: int) -> Pipeline:
    """Pipeline producing the first `count` primes from the natural numbers."""
    def is_prime(n: int) -> bool:
        if n < 2:
            return False
        i = 2
        while i * i <= n:
            if n % i == 0:
                return False
            i += 1
        return True

    return (PipelineBuilder()
            .source(_naturals)
            .filter(is_prime)
            .take(count)
            .build())


def chaos_uniform_pipeline(count: int, r: float = 3.99, x0: float = 0.5) -> Pipeline:
    """Pipeline of logistic-map values scaled to integers in [0, 100)."""
    from src.core.generators.chaos_generator import LogisticMap

    def source():
        return LogisticMap(r=r, x0=x0).iterate()

    return (PipelineBuilder()
            .source(source)
            .skip(100)  # discard transient
            .map(lambda x: int(x * 100))
            .take(count)
            .build())


def even_squares_pipeline(count: int) -> Pipeline:
    """Pipeline of squares of even natural numbers."""
    return (PipelineBuilder()
            .source(_naturals)
            .filter(lambda n: n % 2 == 0)
            .map(lambda n: n * n)
            .take(count)
            .build())
