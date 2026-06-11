"""Plugin system: generator registry with discovery of built-in generators."""

from src.core.plugins.registry import (
    GeneratorRegistry,
    GeneratorInfo,
    register_generator,
    get_generator,
    list_generators,
    discover_builtin_generators,
)

__all__ = [
    "GeneratorRegistry", "GeneratorInfo", "register_generator",
    "get_generator", "list_generators", "discover_builtin_generators",
]
