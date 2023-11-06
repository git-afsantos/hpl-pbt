# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

###############################################################################
# Imports
###############################################################################

from typing import Final, List, Set, Union

from enum import auto, Enum
from attrs import evolve, field, frozen
from attrs.validators import instance_of

################################################################################
# Strategy AST - Value Expressions
################################################################################


class ExpressionType(Enum):
    LITERAL = auto()
    REFERENCE = auto()
    OPERATOR = auto()
    CALL = auto()


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
        return self.type == ExpressionType.OPERATOR

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

    def negative(self) -> 'NumericExpression':
        raise NotImplementedError()

    def add(self, value: 'NumericExpression') -> 'NumericExpression':
        raise NotImplementedError()


@frozen
class NumberLiteral(NumericExpression):
    value: Union[int, float]

    @property
    def type(self) -> ExpressionType:
        return ExpressionType.LITERAL

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
    
    def negative(self) -> 'NumberLiteral':
        return NumberLiteral(-self.value)

    def add(self, value: NumericExpression) -> NumericExpression:
        if value.is_literal:
            assert isinstance(value, NumberLiteral)
            return NumberLiteral(self.value + value.value)
        return value.add(self)



ZERO: Final[NumberLiteral] = NumberLiteral(0)
ONE: Final[NumberLiteral] = NumberLiteral(1)
MINUS_ONE: Final[NumberLiteral] = NumberLiteral(-1)
INFINITY: Final[NumberLiteral] = NumberLiteral(float('inf'))
MINUS_INFINITY: Final[NumberLiteral] = NumberLiteral(-float('inf'))


@frozen
class CompoundNumericExpression(NumericExpression):
    min_value: NumericExpression = MINUS_INFINITY
    max_value: NumericExpression = INFINITY
    exclude_min: bool = False
    exclude_max: bool = False
    exponent: NumericExpression = ONE

    @property
    def type(self) -> ExpressionType:
        return ExpressionType.OPERATOR

    @property
    def is_sum(self) -> bool:
        return False

    @property
    def is_product(self) -> bool:
        return False

    def negative(self) -> NumericExpression:
        raise NotImplementedError()


@frozen
class Sum(CompoundNumericExpression):
    products: List[NumericExpression] = field(factory=list)
    symbols: List[NumericExpression] = field(factory=list)
    constant: NumberLiteral = field(default=ZERO, validator=instance_of(NumberLiteral))

    @constant.validator
    def _check_constant(self, _attr, value: NumberLiteral):
        if value == INFINITY or value == MINUS_INFINITY:
            raise ValueError('sum of infinite values')

    @property
    def is_sum(self) -> bool:
        return True

    def negative(self) -> NumericExpression:
        return evolve(
            self,
            min_value=self.max_value.negative(),
            max_value=self.min_value.negative(),
            exclude_min=self.exclude_max,
            exclude_max=self.exclude_min,
            products=list(p.negative() for p in self.products),
            symbols=list(s.negative() for s in self.symbols),
            constant=self.constant.negative(),
        )

    def add(self, value: NumericExpression) -> NumericExpression:
        if value.is_literal:
            assert isinstance(value, NumberLiteral)
            if value == INFINITY or value == MINUS_INFINITY:
                return value
            return evolve(self, constant=self.constant.add(value))
        if value.is_operator:
            assert isinstance(value, CompoundNumericExpression)
            if value.is_sum:
                assert isinstance(value, Sum)
                if self.is_negative != value.is_negative:
                    pass
                return evolve(
                    self,
                    min_value=self.min_value.add(value.min_value),
                    max_value=self.max_value.add(value.max_value),
                    exclude_min=(self.exclude_min or value.exclude_min),
                    exclude_max=(self.exclude_max or value.exclude_max),
                    products=(self.products + value.products),
                    symbols=list(sorted(self.symbols + value.symbols)),
                    constant=(self.constant.add(value.constant)),
                )
        raise NotImplementedError()
