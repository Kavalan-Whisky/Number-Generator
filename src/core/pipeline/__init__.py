"""Composable pipeline framework for number generation and processing."""

from src.core.pipeline.pipeline import (
    Pipeline,
    PipelineBuilder,
    PipelineStage,
    FilterStage,
    MapStage,
    TakeStage,
    SkipStage,
    DistinctStage,
    StatsSink,
    CollectorSink,
    FileSink,
    prime_pipeline,
    chaos_uniform_pipeline,
    even_squares_pipeline,
)

__all__ = [
    "Pipeline", "PipelineBuilder", "PipelineStage", "FilterStage", "MapStage",
    "TakeStage", "SkipStage", "DistinctStage", "StatsSink", "CollectorSink",
    "FileSink", "prime_pipeline", "chaos_uniform_pipeline", "even_squares_pipeline",
]
