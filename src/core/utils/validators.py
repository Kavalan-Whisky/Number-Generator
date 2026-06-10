"""Input validation utilities."""

from typing import Any, Optional, Union, List


def validate_positive_int(value: Any, name: str = "value") -> int:
    """Validate that value is a positive integer."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def validate_non_negative_int(value: Any, name: str = "value") -> int:
    """Validate that value is a non-negative integer."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def validate_range(value: Any, min_val: Any, max_val: Any, name: str = "value") -> Any:
    """Validate that value is within [min_val, max_val]."""
    if value < min_val or value > max_val:
        raise ValueError(f"{name} must be in [{min_val}, {max_val}], got {value}")
    return value


def validate_count(count: Any, name: str = "count") -> int:
    """Validate count parameter."""
    return validate_positive_int(count, name)


def validate_seed(seed: Any) -> Optional[int]:
    """Validate optional random seed."""
    if seed is None:
        return None
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"seed must be an integer or None, got {type(seed).__name__}")
    return seed


def validate_probability(p: Any, name: str = "probability") -> float:
    """Validate that p is a valid probability [0, 1]."""
    try:
        p = float(p)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be numeric")
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {p}")
    return p


def validate_distribution_params(**params) -> None:
    """Generic distribution parameter validator."""
    for name, (value, constraints) in params.items():
        for constraint, arg in constraints:
            if constraint == "positive" and value <= 0:
                raise ValueError(f"{name} must be positive")
            elif constraint == "non_negative" and value < 0:
                raise ValueError(f"{name} must be non-negative")
            elif constraint == "range" and not (arg[0] <= value <= arg[1]):
                raise ValueError(f"{name} must be in {arg}")


def validate_list_of_numbers(values: Any, name: str = "values") -> List[float]:
    """Validate a list of numbers."""
    if not hasattr(values, '__iter__'):
        raise TypeError(f"{name} must be iterable")
    result = []
    for i, v in enumerate(values):
        try:
            result.append(float(v))
        except (TypeError, ValueError):
            raise ValueError(f"{name}[{i}] must be numeric, got {repr(v)}")
    return result


def validate_min_max(min_val: Any, max_val: Any) -> tuple:
    """Validate min < max."""
    try:
        min_val = float(min_val)
        max_val = float(max_val)
    except (TypeError, ValueError):
        raise TypeError("min and max must be numeric")
    if min_val >= max_val:
        raise ValueError(f"min ({min_val}) must be less than max ({max_val})")
    return min_val, max_val


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to [min_val, max_val]."""
    return max(min_val, min(max_val, value))
