# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

###############################################################################
# Imports
###############################################################################

from typing import Final, Iterable, Mapping, Optional, Set, Union

from enum import auto, Enum
from attrs import evolve, field, frozen
from attrs.validators import instance_of
from hplpbt.errors import ContradictionError

################################################################################
# Strategy AST - Value Expressions
################################################################################


class ExpressionType(Enum):
    LITERAL = auto()
    REFERENCE = auto()
    SYMBOL = auto()
    SUM = auto()
    PRODUCT = auto()
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
    def is_symbol(self) -> bool:
        return self.type == ExpressionType.SYMBOL

    @property
    def is_sum(self) -> bool:
        return self.type == ExpressionType.SUM

    @property
    def is_product(self) -> bool:
        return self.type == ExpressionType.PRODUCT

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

    @property
    def has_minus_sign(self) -> bool:
        raise NotImplementedError()

    @property
    def has_exponent(self) -> bool:
        raise NotImplementedError()

    def negative(self) -> 'NumericExpression':
        raise NotImplementedError()

    def add(self, value: 'NumericExpression') -> 'NumericExpression':
        raise NotImplementedError()

    def multiply(self, value: 'NumericExpression') -> 'NumericExpression':
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

    @property
    def is_negative(self) -> bool:
        return self.value < 0

    @property
    def has_minus_sign(self) -> bool:
        return self.value < 0

    @property
    def has_exponent(self) -> bool:
        return False

    @property
    def min_value(self) -> NumericExpression:
        return self

    @property
    def max_value(self) -> NumericExpression:
        return self

    @property
    def exclude_min(self) -> bool:
        return False

    @property
    def exclude_max(self) -> bool:
        return False

    @property
    def exponent(self) -> NumericExpression:
        return NumberLiteral(1)

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
        if self.value == 0:
            return value
        return value.add(self)

    def multiply(self, value: NumericExpression) -> NumericExpression:
        if value.is_literal:
            assert isinstance(value, NumberLiteral)
            return NumberLiteral(self.value * value.value)
        if self.value == 0:
            return self
        if self.value == 1:
            return value
        if self.value == -1:
            return value.negative()
        return value.multiply(self)


ZERO: Final[NumberLiteral] = NumberLiteral(0)
ONE: Final[NumberLiteral] = NumberLiteral(1)
MINUS_ONE: Final[NumberLiteral] = NumberLiteral(-1)
INFINITY: Final[NumberLiteral] = NumberLiteral(float('inf'))
MINUS_INFINITY: Final[NumberLiteral] = NumberLiteral(-float('inf'))


@frozen
class UnknownNumber(NumericExpression):
    min_value: NumericExpression = MINUS_INFINITY
    max_value: NumericExpression = INFINITY
    exclude_min: bool = False
    exclude_max: bool = False
    is_negative: bool = False
    exponent: NumericExpression = ONE

    @property
    def has_minus_sign(self) -> bool:
        return self.is_negative

    @property
    def has_exponent(self) -> bool:
        return self.exponent != ONE

    def negative(self) -> NumericExpression:
        return evolve(
            self,
            min_value=self.max_value,
            max_value=self.min_value,
            exclude_min=self.exclude_max,
            exclude_max=self.exclude_min,
            is_negative=(not self.is_negative),
        )

    def solve(self, **symbols: Mapping[str, NumericExpression]) -> NumericExpression:
        raise NotImplementedError()


_symbol_counter = 0


def _default_symbol_name() -> str:
    global _symbol_counter
    _symbol_counter += 1
    return f'x{_symbol_counter}'


@frozen
class Symbol(UnknownNumber):
    name: str = field(factory=_default_symbol_name)

    @property
    def type(self) -> ExpressionType:
        return ExpressionType.SYMBOL

    def solve(self, **symbols: Mapping[str, NumericExpression]) -> NumericExpression:
        # solve constituent bits of information
        lower = self.min_value if self.min_value.is_literal else self.min_value.solve(**symbols)
        upper = self.max_value if self.max_value.is_literal else self.max_value.solve(**symbols)
        power = self.exponent if self.exponent.is_literal else self.exponent.solve(**symbols)
        if not lower.is_literal or not upper.is_literal or not power.is_literal:
            # cannot ever get to an actual value
            return evolve(self, min_value=lower, max_value=upper, exponent=power)
        assert isinstance(lower, NumberLiteral)
        assert isinstance(upper, NumberLiteral)
        assert isinstance(power, NumberLiteral)
        a = lower.value
        b = upper.value
        e = power.value

        # recursively substitute until reaching a dead end
        visited = {self.name}
        value = symbols.get(self.name, self)
        while value.is_symbol:
            if value.name in visited:
                break  # undefined or cyclic
            visited.add(value.name)
            value = symbols.get(value.name, value)
        if value.is_operator:
            value = value.solve(**symbols)

        if value.is_literal:
            # fully resolved, calculate value
            assert isinstance(value, NumberLiteral)
            x = value.value
            x = x ** e
            if self.is_negative:
                x = -x
            # validation
            if self.exclude_min:
                if x <= a:
                    raise ContradictionError(f'{a} < {x}')
            elif x < a:
                raise ContradictionError(f'{a} <= {x}')
            if self.exclude_max:
                if x >= b:
                    raise ContradictionError(f'{b} > {x}')
            elif x > b:
                raise ContradictionError(f'{b} >= {x}')
            return NumberLiteral(x)

        return evolve(
            value,
            min_value=lower,
            max_value=upper,
            exclude_min=self.exclude_min,
            exclude_max=self.exclude_max,
            is_negative=(self.is_negative ^ value.is_negative),
            exponent=power.multiply(value.exponent),
        )


@frozen
class Sum(UnknownNumber):
    parts: Iterable[NumericExpression] = field(factory=tuple, converter=tuple)

    @property
    def type(self) -> ExpressionType:
        return ExpressionType.SUM

    @property
    def is_int(self) -> bool:
        return all(p.is_int for p in self.parts)

    @property
    def is_float(self) -> bool:
        return not self.is_int

    def add(self, value: NumericExpression) -> NumericExpression:
        if self.has_exponent:
            return Sum(parts=(self, value))
        if value.is_literal:
            assert isinstance(value, NumberLiteral)
            if value == INFINITY or value == MINUS_INFINITY:
                return value
            parts = []
            constant = value
            for part in self.parts:
                if part.is_literal:
                    constant = constant.add(part)
                else:
                    parts.append(part)
            parts.append(constant)
            return evolve(self, parts=parts)
        if value.is_sum and not value.has_exponent:
            assert isinstance(value, Sum)
            if self.is_negative != value.is_negative:
                value = value.negative()
            return evolve(
                self,
                min_value=self.min_value.add(value.min_value),
                max_value=self.max_value.add(value.max_value),
                exclude_min=(self.exclude_min or value.exclude_min),
                exclude_max=(self.exclude_max or value.exclude_max),
                parts=(self.parts + value.parts),
            )
        parts = self.parts + (value,)
        return evolve(self, parts=parts)

    def solve(self, **symbols: Mapping[str, NumericExpression]) -> NumericExpression:
        parts = []
        constant = ZERO
        for part in self.parts:
            if part.is_symbol:
                # recursively substitute until reaching a dead end
                assert isinstance(part, Symbol)
                visited = {part.name}
                value = symbols.get(part.name, part)
                while value.is_symbol:
                    if value.name in visited:
                        break  # undefined or cyclic
                    visited.add(value.name)
                    value = symbols.get(value.name, value)
                else:
                    part = value
            if part.is_sum or part.is_product:
                part = part.solve(**symbols)
            if part.is_literal:
                constant = constant.add(part)
            else:
                parts.append(part)
        if not parts:
            # fully resolved

            return constant
        if constant != ZERO:
            parts.append(constant)
        return 


def try_solve(
    expr: NumericExpression,
    **symbols: Mapping[str, NumericExpression],
) -> NumericExpression:
    if expr.is_literal:
        return expr
    if expr.is_sum:
        assert isinstance(expr, Sum)
        return expr.solve(**symbols)
    raise NotImplementedError()


def add(a: NumericExpression, b: NumericExpression) -> NumericExpression:
    if a.is_literal:
        assert isinstance(a, NumberLiteral)
        if a.value == 0:
            return b
        if b.is_literal:
            assert isinstance(b, NumberLiteral)
            return NumberLiteral(a.value + b.value)
    elif b.is_literal:
        assert isinstance(b, NumberLiteral)
        if b.value == 0:
            return a
        return NumberLiteral(a.value + b.value)
    else:
        assert isinstance(a, UnknownNumber)
        assert isinstance(b, UnknownNumber)
        min_value = add(a.min_value, b.min_value)
        max_value = add(a.max_value, b.max_value)
        exclude_max = a.exclude_max or b.exclude_max
        exclude_min = a.exclude_min or b.exclude_min
        return Sum(
            min_value=min_value,
            max_value=max_value,
            exclude_min=exclude_min,
            exclude_max=exclude_max,
            parts=(a, b),
        )
