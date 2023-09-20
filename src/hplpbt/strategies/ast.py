# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

###############################################################################
# Imports
###############################################################################

from typing import Any, Iterable, List, Optional

from enum import auto, Enum

from attrs import field, frozen
from attrs.validators import deep_iterable, instance_of
from hpl.ast import HplExpression
from typeguard import check_type

################################################################################
# Strategy AST - Value Expressions
################################################################################


class ExpressionType(Enum):
    LITERAL = auto()
    REFERENCE = auto()
    RANDOM_BOOL = auto()
    RANDOM_INT = auto()
    RANDOM_FLOAT = auto()
    RANDOM_STRING = auto()
    RANDOM_ARRAY = auto()
    RANDOM_SAMPLE = auto()

    @property
    def is_random_value(self) -> bool:
        return self in (
            ExpressionType.RANDOM_BOOL,
            ExpressionType.RANDOM_INT,
            ExpressionType.RANDOM_FLOAT,
            ExpressionType.RANDOM_STRING,
            ExpressionType.RANDOM_ARRAY,
            ExpressionType.RANDOM_SAMPLE,
        )


@frozen
class Expression:
    @property
    def type(self) -> ExpressionType:
        raise NotImplementedError()

    @property
    def is_literal(self) -> bool:
        return self.type == ExpressionType.LITERAL

    @property
    def is_reference(self) -> bool:
        return self.type == ExpressionType.REFERENCE

    @property
    def is_random_value(self) -> bool:
        return self.type.is_random_value

    @property
    def is_random_bool(self) -> bool:
        return self.type == ExpressionType.RANDOM_BOOL

    @property
    def is_random_int(self) -> bool:
        return self.type == ExpressionType.RANDOM_INT

    @property
    def is_random_float(self) -> bool:
        return self.type == ExpressionType.RANDOM_FLOAT

    @property
    def is_random_string(self) -> bool:
        return self.type == ExpressionType.RANDOM_STRING

    @property
    def is_random_array(self) -> bool:
        return self.type == ExpressionType.RANDOM_ARRAY

    @property
    def is_random_sample(self) -> bool:
        return self.type == ExpressionType.RANDOM_SAMPLE


@frozen
class Literal(Expression):
    value: Any

    @property
    def type(self) -> ExpressionType:
        return ExpressionType.LITERAL

    @property
    def is_none(self) -> bool:
        return self.value is None

    @property
    def is_bool(self) -> bool:
        return isinstance(self.value, bool)

    @property
    def is_int(self) -> bool:
        return isinstance(self.value, int)

    @property
    def is_float(self) -> bool:
        return isinstance(self.value, float)

    @property
    def is_string(self) -> bool:
        return isinstance(self.value, str)

    @property
    def is_array(self) -> bool:
        return isinstance(self.value, (list, tuple))

    @classmethod
    def none(cls) -> 'Literal':
        return cls(None)

    @classmethod
    def true(cls) -> 'Literal':
        return cls(True)

    @classmethod
    def false(cls) -> 'Literal':
        return cls(False)

    @classmethod
    def zero(cls) -> 'Literal':
        return cls(0)

    @classmethod
    def empty_string(cls) -> 'Literal':
        return cls('')

    @classmethod
    def empty_list(cls) -> 'Literal':
        return cls([])

    def __str__(self) -> str:
        return repr(self.value)


@frozen
class Reference(Expression):
    variable: str

    @property
    def type(self) -> ExpressionType:
        return ExpressionType.REFERENCE

    def __str__(self) -> str:
        return self.variable


@frozen
class RandomBool(Expression):
    @property
    def type(self) -> ExpressionType:
        return ExpressionType.RANDOM_BOOL

    def __str__(self) -> str:
        return 'booleans()'


@frozen
class RandomInt(Expression):
    min_value: Expression = field(factory=Literal.none)
    max_value: Expression = field(factory=Literal.none)

    @property
    def type(self) -> ExpressionType:
        return ExpressionType.RANDOM_INT

    def __str__(self) -> str:
        return f'integers(min_value={self.min_value}, max_value={self.max_value})'


@frozen
class RandomFloat(Expression):
    min_value: Expression = field(factory=Literal.none)
    max_value: Expression = field(factory=Literal.none)
    # allow_nan: bool = True
    # allow_infinity: bool = True
    # allow_subnormal: bool = True
    # width: int = 64
    # exclude_min: bool = False
    # exclude_max: bool = False

    @property
    def type(self) -> ExpressionType:
        return ExpressionType.RANDOM_FLOAT

    def __str__(self) -> str:
        # min_value=None
        # max_value=None
        # allow_nan=None
        # allow_infinity=None
        # allow_subnormal=None
        # width=64
        # exclude_min=False
        # exclude_max=False
        return f'floats(min_value={self.min_value}, max_value={self.max_value})'


@frozen
class RandomString(Expression):
    min_size: Expression = field(factory=Literal.zero)
    max_size: Expression = field(factory=Literal.none)
    # alphabet: Optional[Expression] = None

    @property
    def type(self) -> ExpressionType:
        return ExpressionType.RANDOM_STRING

    def __str__(self) -> str:
        # alphabet=characters(codec='utf-8')
        # min_size=0
        # max_size=None
        return f'text(min_size={self.min_size}, max_size={self.max_size})'


@frozen
class RandomArray(Expression):
    elements: Expression = field(factory=RandomBool, validator=instance_of(Expression))
    min_size: Expression = field(factory=Literal.zero)
    max_size: Expression = field(factory=Literal.none)
    unique: bool = False

    @elements.validator
    def _check_elements(self, _attribute, value: Expression):
        if not value.is_random_value:
            raise ValueError(f'expected a value generator, got {value!r}')

    @property
    def type(self) -> ExpressionType:
        return ExpressionType.RANDOM_ARRAY

    def __str__(self) -> str:
        # min_size=0
        # max_size=None
        # unique_by=None
        # unique=False
        return (
            f'lists({self.elements}'
            f', min_size={self.min_size}'
            f', max_size={self.max_size}'
            f', unique={self.unique})'
        )


@frozen
class RandomSample(Expression):
    elements: Iterable[Expression] = field(factory=tuple, converter=tuple)

    @elements.validator
    def _check_elements(self, _attribute, values: Iterable[Expression]):
        for value in values:
            value: Expression = check_type(value, Expression)
            if value.is_literal:
                continue
            if value.is_reference:
                continue
            raise ValueError(f'expression must not be a generator: {value!r}')

    @property
    def type(self) -> ExpressionType:
        return ExpressionType.RANDOM_SAMPLE

    def __str__(self) -> str:
        return f'sampled_from({self.elements})'


################################################################################
# Strategy AST
################################################################################


@frozen
class StrategyCondition:
    pass


@frozen
class Statement:
    @property
    def is_assignment(self) -> bool:
        return False

    @property
    def is_assumption(self) -> bool:
        return False

    @property
    def is_loop(self) -> bool:
        return False

    @property
    def is_block(self) -> bool:
        return False

    def merge(self, other: 'Statement') -> 'Statement':
        c1 = type(self).__name__
        c2 = type(other).__name__
        raise TypeError(f'unable to merge {c1} with {c2}')


################################################################################
# Strategy AST - Assignment
################################################################################


@frozen
class Assignment(Statement):
    variable: str
    expression: HplExpression

    @property
    def is_assignment(self) -> bool:
        return True

    def __str__(self) -> str:
        return f'{self.variable} = {self.expression}'


################################################################################
# Strategy AST - Assumption
################################################################################


@frozen
class Assumption(Statement):
    expression: HplExpression

    @property
    def is_assumption(self) -> bool:
        return True

    def __str__(self) -> str:
        return f'assume({self.expression})'
