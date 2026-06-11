"""
Generator plugin registry.

Provides a GeneratorRegistry with a register decorator, discovery of all
built-in generators, get_generator(name), and list_generators() with metadata.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class GeneratorInfo:
    """Metadata for a registered generator."""
    name: str
    func: Callable[..., List]
    category: str = "general"
    description: str = ""
    params: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "params": dict(self.params),
        }


class GeneratorRegistry:
    """Registry of named generator functions.

    Each generator is a callable taking (count, **kwargs) and returning a list.
    """

    def __init__(self):
        self._generators: Dict[str, GeneratorInfo] = {}
        self._discovered = False

    def register(self, name: str, category: str = "general",
                 description: str = "", params: Dict[str, str] = None):
        """Decorator: register a generator function under `name`."""
        def decorator(func: Callable) -> Callable:
            if name in self._generators:
                raise ValueError(f"Generator '{name}' is already registered")
            self._generators[name] = GeneratorInfo(
                name=name, func=func, category=category,
                description=description or (func.__doc__ or "").strip().split("\n")[0],
                params=params or {},
            )
            return func
        return decorator

    def unregister(self, name: str) -> None:
        self._generators.pop(name, None)

    def get(self, name: str) -> GeneratorInfo:
        self.ensure_discovered()
        if name not in self._generators:
            raise KeyError(f"Unknown generator: {name}. "
                           f"Available: {sorted(self._generators)}")
        return self._generators[name]

    def generate(self, name: str, count: int, **kwargs) -> List:
        return self.get(name).func(count, **kwargs)

    def list(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        self.ensure_discovered()
        infos = sorted(self._generators.values(), key=lambda i: (i.category, i.name))
        if category:
            infos = [i for i in infos if i.category == category]
        return [i.to_dict() for i in infos]

    def names(self) -> List[str]:
        self.ensure_discovered()
        return sorted(self._generators)

    def __contains__(self, name: str) -> bool:
        self.ensure_discovered()
        return name in self._generators

    def __len__(self) -> int:
        self.ensure_discovered()
        return len(self._generators)

    def ensure_discovered(self) -> None:
        if not self._discovered:
            self._discovered = True
            _register_builtins(self)


# Global registry instance
_registry = GeneratorRegistry()


def register_generator(name: str, category: str = "general",
                       description: str = "", params: Dict[str, str] = None):
    """Module-level register decorator using the global registry."""
    return _registry.register(name, category, description, params)


def get_generator(name: str) -> GeneratorInfo:
    """Look up a generator by name in the global registry."""
    return _registry.get(name)


def list_generators(category: Optional[str] = None) -> List[Dict[str, Any]]:
    """List all registered generators with metadata."""
    return _registry.list(category)


def discover_builtin_generators() -> GeneratorRegistry:
    """Ensure all built-in generators are registered; return the registry."""
    _registry.ensure_discovered()
    return _registry


def _register_builtins(registry: GeneratorRegistry) -> None:
    """Register all built-in generators from the core modules."""
    from src.core.generators.prime_generator import PrimeGeneratorFactory
    from src.core.generators.fibonacci_generator import FibonacciGenerator
    from src.core.generators.random_generator import RandomGeneratorFactory
    from src.core.generators.chaos_generator import ChaosGeneratorFactory
    from src.core.generators.noise_generator import NoiseGeneratorFactory
    from src.core.generators.quasirandom_generator import QuasiRandomFactory
    from src.core.generators.digits_generator import DigitsGeneratorFactory
    from src.core.generators.numbertheory_generator import NumberTheoryFactory

    def reg(name, category, description, func, params=None):
        if name not in registry._generators:
            registry._generators[name] = GeneratorInfo(
                name=name, func=func, category=category,
                description=description, params=params or {})

    reg("prime", "classic", "Prime numbers via sieve",
        lambda count, start=2, **kw: PrimeGeneratorFactory.generate_primes(count, start),
        {"start": "first prime >= start"})
    reg("fibonacci", "classic", "Fibonacci sequence",
        lambda count, **kw: FibonacciGenerator().generate(count))
    reg("random", "random", "Pseudo-random integers",
        lambda count, algorithm="mersenne", seed=None, min_val=0, max_val=1000, **kw:
            RandomGeneratorFactory.create(algorithm, seed=seed).generate(count, min_val, max_val),
        {"algorithm": "PRNG algorithm", "seed": "random seed"})

    for m in ChaosGeneratorFactory.available_maps():
        reg(f"chaos_{m}", "chaos", f"Chaotic {m} map sequence",
            lambda count, _m=m, **kw: ChaosGeneratorFactory.generate(_m, count, **kw))
    for t in NoiseGeneratorFactory.available_types():
        reg(f"noise_{t}", "noise", f"{t.title()} noise sequence",
            lambda count, _t=t, **kw: NoiseGeneratorFactory.generate(_t, count, **kw))
    for q in QuasiRandomFactory.available_types():
        reg(f"quasi_{q}", "quasirandom", f"{q} low-discrepancy sequence",
            lambda count, _q=q, **kw: QuasiRandomFactory.generate(_q, count, **kw))
    for c in DigitsGeneratorFactory.available_constants():
        reg(f"digits_{c}", "digits", f"Digits of {c}",
            lambda count, _c=c, **kw: DigitsGeneratorFactory.generate(_c, count, **kw))
    for nt in NumberTheoryFactory.available_types():
        reg(f"special_{nt}", "numbertheory", f"{nt.title()} numbers",
            lambda count, _nt=nt, **kw: NumberTheoryFactory.generate(_nt, count, kw.get("start")))
