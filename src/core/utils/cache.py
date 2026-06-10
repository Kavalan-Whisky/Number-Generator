"""Caching utilities including LRU cache, TTL cache, and decorators."""

import time
import functools
from collections import OrderedDict
from typing import Any, Callable, Optional, TypeVar, Dict, Tuple

F = TypeVar('F', bound=Callable[..., Any])


class LRUCache:
    """Least Recently Used cache with configurable capacity."""

    def __init__(self, capacity: int = 128):
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        self.capacity = capacity
        self._cache: OrderedDict = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: Any) -> Optional[Any]:
        if key not in self._cache:
            self._misses += 1
            return None
        self._cache.move_to_end(key)
        self._hits += 1
        return self._cache[key]

    def put(self, key: Any, value: Any) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)

    def __contains__(self, key: Any) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "size": len(self._cache),
            "capacity": self.capacity,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }


class TTLCache:
    """Time-to-Live cache where entries expire after a given duration."""

    def __init__(self, capacity: int = 128, ttl: float = 60.0):
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if ttl <= 0:
            raise ValueError("TTL must be positive")
        self.capacity = capacity
        self.ttl = ttl
        self._cache: Dict[Any, Tuple[Any, float]] = {}
        self._order: OrderedDict = OrderedDict()

    def _is_expired(self, key: Any) -> bool:
        if key not in self._cache:
            return True
        _, timestamp = self._cache[key]
        return (time.monotonic() - timestamp) > self.ttl

    def get(self, key: Any) -> Optional[Any]:
        if key not in self._cache or self._is_expired(key):
            if key in self._cache:
                del self._cache[key]
                del self._order[key]
            return None
        self._order.move_to_end(key)
        value, _ = self._cache[key]
        return value

    def put(self, key: Any, value: Any) -> None:
        now = time.monotonic()
        if key in self._cache:
            self._order.move_to_end(key)
        else:
            self._order[key] = True
        self._cache[key] = (value, now)

        # Evict expired and LRU entries
        while len(self._cache) > self.capacity:
            oldest_key = next(iter(self._order))
            del self._cache[oldest_key]
            del self._order[oldest_key]

    def __contains__(self, key: Any) -> bool:
        return key in self._cache and not self._is_expired(key)

    def clear(self) -> None:
        self._cache.clear()
        self._order.clear()

    def __len__(self) -> int:
        return sum(1 for k in list(self._cache) if not self._is_expired(k))


def memoize(func: F = None, *, maxsize: int = 128) -> F:
    """Memoization decorator using LRU cache."""
    def decorator(f: F) -> F:
        cache = LRUCache(capacity=maxsize)

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            cached = cache.get(key)
            if cached is not None:
                return cached
            # Use sentinel to distinguish None result from cache miss
            sentinel = object()
            result = cache.get((key, 'result'))
            if result is sentinel or result is None:
                result = f(*args, **kwargs)
                cache.put(key, result)
                cache.put((key, 'result'), result)
            return result

        wrapper.cache = cache
        wrapper.cache_clear = cache.clear
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def memoize_simple(func: F) -> F:
    """Simple memoization using a plain dict (unlimited size)."""
    _cache: Dict = {}

    @functools.wraps(func)
    def wrapper(*args):
        if args not in _cache:
            _cache[args] = func(*args)
        return _cache[args]

    wrapper.cache = _cache
    wrapper.cache_clear = _cache.clear
    return wrapper


class cached_property:
    """Descriptor that computes the value once and caches it on the instance."""

    def __init__(self, func: Callable):
        self.func = func
        self.attrname = None
        self.__doc__ = func.__doc__

    def __set_name__(self, owner: type, name: str) -> None:
        self.attrname = name

    def __get__(self, instance: Any, owner: type = None) -> Any:
        if instance is None:
            return self
        name = self.attrname or self.func.__name__
        cache_attr = f'_cached_{name}'
        if not hasattr(instance, cache_attr):
            setattr(instance, cache_attr, self.func(instance))
        return getattr(instance, cache_attr)


class FibonacciMemoCache:
    """Specialized cache for Fibonacci-like recursive computations."""

    def __init__(self):
        self._memo: Dict[int, int] = {0: 0, 1: 1}

    def get(self, n: int) -> Optional[int]:
        return self._memo.get(n)

    def put(self, n: int, value: int) -> None:
        self._memo[n] = value

    def __contains__(self, n: int) -> bool:
        return n in self._memo

    def clear(self) -> None:
        self._memo = {0: 0, 1: 1}

    def size(self) -> int:
        return len(self._memo)
