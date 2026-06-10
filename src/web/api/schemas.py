"""
Request/Response schemas using marshmallow.
"""

from marshmallow import Schema, fields, validate, ValidationError, post_load
from typing import Any, Dict, Optional


class RandomGenerateSchema(Schema):
    algorithm = fields.Str(
        missing="mersenne",
        validate=validate.OneOf(["lcg", "xorshift32", "xorshift64", "pcg", "lfsr", "bbs", "middle_square", "mersenne"])
    )
    count = fields.Int(missing=10, validate=validate.Range(min=1, max=10000))
    min = fields.Int(missing=0)
    max = fields.Int(missing=1000)
    seed = fields.Int(missing=None, allow_none=True)


class PrimeGenerateSchema(Schema):
    count = fields.Int(missing=20, validate=validate.Range(min=1, max=10000))
    start = fields.Int(missing=2, validate=validate.Range(min=2))


class FibonacciSchema(Schema):
    count = fields.Int(missing=20, validate=validate.Range(min=1, max=1000))
    variant = fields.Str(
        missing="iterative",
        validate=validate.OneOf(["iterative", "recursive", "matrix", "closed_form"])
    )
    type = fields.Str(
        missing="fibonacci",
        validate=validate.OneOf(["fibonacci", "lucas", "tribonacci", "tetranacci"])
    )


class SequenceSchema(Schema):
    type = fields.Str(missing="arithmetic")
    count = fields.Int(missing=15, validate=validate.Range(min=1, max=1000))
    start = fields.Float(missing=None, allow_none=True)
    difference = fields.Float(missing=None, allow_none=True)
    ratio = fields.Float(missing=None, allow_none=True)
    n = fields.Int(missing=None, allow_none=True)


class StatisticalSchema(Schema):
    distribution = fields.Str(missing="normal")
    count = fields.Int(missing=100, validate=validate.Range(min=1, max=100000))
    seed = fields.Int(missing=None, allow_none=True)
    mean = fields.Float(missing=0.0)
    std = fields.Float(missing=1.0, validate=validate.Range(min=0.0001))
    rate = fields.Float(missing=1.0, validate=validate.Range(min=0.0001))
    low = fields.Float(missing=0.0)
    high = fields.Float(missing=1.0)
    lam = fields.Float(missing=1.0, validate=validate.Range(min=0.0001))


class AnalyzeSchema(Schema):
    data = fields.List(fields.Float(), required=True, validate=validate.Length(min=2))
    tests = fields.Str(
        missing="statistical",
        validate=validate.OneOf(["statistical", "randomness", "patterns", "distributions", "all"])
    )


class TransformSchema(Schema):
    data = fields.List(fields.Float(), required=True, validate=validate.Length(min=1))
    operation = fields.Str(required=True)
    params = fields.Dict(missing={})


def validate_request(schema_class, data: Dict) -> tuple[Optional[Dict], Optional[str]]:
    """
    Validate request data against a schema.
    Returns (validated_data, error_message).
    """
    try:
        validated = schema_class().load(data)
        return validated, None
    except ValidationError as e:
        return None, str(e.messages)


# Schema registry
SCHEMAS = {
    "random": RandomGenerateSchema,
    "prime": PrimeGenerateSchema,
    "fibonacci": FibonacciSchema,
    "sequence": SequenceSchema,
    "statistical": StatisticalSchema,
    "analyze": AnalyzeSchema,
    "transform": TransformSchema,
}
