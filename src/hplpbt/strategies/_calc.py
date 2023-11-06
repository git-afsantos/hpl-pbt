# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

###############################################################################
# Imports
###############################################################################

from typing import Final, Set, Union

from enum import auto, Enum
from attrs import field, frozen

################################################################################
# Strategy AST - Value Expressions
################################################################################


class ExpressionType(Enum):
    LITERAL = auto()
    REFERENCE = auto()
    UNARY_OPERATOR = auto()
    BINARY_OPERATOR = auto()
    CALL = auto()
    ITERATOR = auto()
    DOT_ACCESS = auto()
    KEY_ACCESS = auto()
    DRAW = auto()


@frozen
class Expression:
    """
    Base class that represents concrete values in the AST, for example, values
    given as function arguments and values used within other expressions.
    """

    @property
    def type(self) -> ExpressionType:
        raise NotImplementedError()

    @property
    def is_boolean(self) -> bool:
        return False

    @property
    def is_number(self) -> bool:
        return False

    @property
    def is_string(self) -> bool:
        return False

    @property
    def is_literal(self) -> bool:
        return self.type == ExpressionType.LITERAL

    @property
    def is_reference(self) -> bool:
        return self.type == ExpressionType.REFERENCE

    @property
    def is_operator(self) -> bool:
        return self.type == ExpressionType.UNARY_OPERATOR

    @property
    def is_function_call(self) -> bool:
        return self.type == ExpressionType.CALL

    def references(self) -> Set[str]:
        raise NotImplementedError()


################################################################################
# Number Expressions
################################################################################


@frozen
class NumericExpression(Expression):
    @property
    def is_number(self) -> bool:
        return True

    @property
    def is_int(self) -> bool:
        return False

    @property
    def is_float(self) -> bool:
        return False


@frozen
class NumberLiteral(NumericExpression):
    value: Union[int, float]

    @property
    def is_int(self) -> bool:
        return isinstance(self.value, int)

    @property
    def is_float(self) -> bool:
        return isinstance(self.value, float)

    @classmethod
    def zero(cls) -> 'NumberLiteral':
        return cls(0)

    @classmethod
    def one(cls) -> 'NumberLiteral':
        return cls(1)


ZERO: Final[NumberLiteral] = NumberLiteral(0)
ONE: Final[NumberLiteral] = NumberLiteral(1)
MINUS_ONE: Final[NumberLiteral] = NumberLiteral(-1)


@frozen
class CompoundNumericExpression(Expression):
    min_value: NumericExpression
    max_value: NumericExpression
    exclude_min: bool = False
    exclude_max: bool = False
    negative: bool = False
    exponent: NumericExpression = ONE

    @property
    def is_sum(self) -> bool:
        return False

    @property
    def is_product(self) -> bool:
        return False
