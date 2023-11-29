# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

###############################################################################
# Imports
###############################################################################

from typing import Final, Iterable, Mapping, Set, Union

from enum import auto, Enum
from attrs import evolve, field, frozen
from attrs.validators import deep_iterable, instance_of, min_len
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
    def min_value(self) -> 'NumericExpression':
        raise NotImplementedError()

    @property
    def max_value(self) -> 'NumericExpression':
        raise NotImplementedError()

    @property
    def exclude_min(self) -> bool:
        raise NotImplementedError()

    @property
    def exclude_max(self) -> bool:
        raise NotImplementedError()

    @property
    def minus_sign(self) -> bool:
        raise NotImplementedError()

    def negative(self) -> 'NumericExpression':
        raise NotImplementedError()

    def solve(self, **symbols: Mapping[str, 'NumericExpression']) -> 'NumericExpression':
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
    def minus_sign(self) -> bool:
        return self.value < 0

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

    def solve(self, **_symbols: Mapping[str, NumericExpression]) -> NumericExpression:
        return self

    def __str__(self) -> str:
        return str(self.value)


ZERO: Final[NumberLiteral] = NumberLiteral(0)
ONE: Final[NumberLiteral] = NumberLiteral(1)
MINUS_ONE: Final[NumberLiteral] = NumberLiteral(-1)
INFINITY: Final[NumberLiteral] = NumberLiteral(float('inf'))
MINUS_INFINITY: Final[NumberLiteral] = NumberLiteral(-float('inf'))


def _ensure_num_expr(x: Union[int, float, NumericExpression]) -> NumericExpression:
    return x if isinstance(x, NumericExpression) else NumberLiteral(x)


_symbol_counter = 0


def _default_symbol_name() -> str:
    global _symbol_counter
    _symbol_counter += 1
    return f'x{_symbol_counter}'


@frozen
class Symbol(NumericExpression):
    name: str = field(factory=_default_symbol_name)
    exponent: NumericExpression = field(default=ONE, converter=_ensure_num_expr)
    # overrides
    minus_sign: bool = False 
    min_value: NumericExpression = field(default=MINUS_INFINITY, converter=_ensure_num_expr)
    max_value: NumericExpression = field(default=INFINITY, converter=_ensure_num_expr)
    exclude_min: bool = False
    exclude_max: bool = False

    @property
    def type(self) -> ExpressionType:
        return ExpressionType.SYMBOL

    def negative(self) -> NumericExpression:
        return evolve(
            self,
            minus_sign=(not self.minus_sign),
            min_value=self.max_value.negative(),
            max_value=self.min_value.negative(),
            exclude_min=self.exclude_max,
            exclude_max=self.exclude_min,
        )

    def solve(self, **symbols: Mapping[str, NumericExpression]) -> NumericExpression:
        backup = self
        # solve constituent bits of information
        min_value = self.min_value.solve(**symbols)
        max_value = self.max_value.solve(**symbols)
        exponent = self.exponent.solve(**symbols)
        if not min_value.is_literal or not max_value.is_literal or not exponent.is_literal:
            # cannot ever get to an actual value
            return evolve(self, min_value=min_value, max_value=max_value, exponent=exponent)
        assert isinstance(min_value, NumberLiteral)
        assert isinstance(max_value, NumberLiteral)
        assert isinstance(exponent, NumberLiteral)
        if min_value is not self.min_value:
            backup = evolve(self, min_value=min_value, max_value=max_value, exponent=exponent)
        elif max_value is not self.max_value:
            backup = evolve(self, min_value=min_value, max_value=max_value, exponent=exponent)
        elif exponent is not self.exponent:
            backup = evolve(self, min_value=min_value, max_value=max_value, exponent=exponent)

        # recursively substitute until reaching a dead end
        visited = {self.name}
        value = symbols.get(self.name, self)
        while value.is_symbol:
            if value.name in visited:
                break  # undefined or cyclic
            visited.add(value.name)
            value = symbols.get(value.name, value)

        # got to another symbol?
        if value.is_symbol:
            assert isinstance(value, Symbol)
            if value.name == self.name:
                return backup
            return evolve(
                value,
                exponent=multiply(exponent, value.exponent),
                min_value=min_value,
                max_value=max_value,
                exclude_min=self.exclude_min,
                exclude_max=self.exclude_max,
                minus_sign=(self.minus_sign ^ value.minus_sign),
            )

        # got to a literal?
        if value.is_literal:
            # fully resolved, calculate value
            assert isinstance(value, NumberLiteral)
            x = value.value
            x = x ** exponent.value
            if self.minus_sign:
                x = -x
            # validation
            a = min_value.value
            b = max_value.value
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

        # something else
        return value.solve(**symbols)

    def __str__(self) -> str:
        if self.minus_sign:
            if self.exponent == ONE:
                return f'-{self.name}'
            else:
                return f'-({self.name} ** {self.exponent})'
        elif self.exponent == ONE:
            return self.name
        else:
            return f'({self.name} ** {self.exponent})'


@frozen
class Sum(NumericExpression):
    parts: Iterable[NumericExpression] = field(
        factory=tuple,
        converter=tuple,
        validator=deep_iterable(
            instance_of(NumericExpression),
            iterable_validator=min_len(2),
        )
    )

    @property
    def type(self) -> ExpressionType:
        return ExpressionType.SUM

    @property
    def is_int(self) -> bool:
        return all(p.is_int for p in self.parts)

    @property
    def min_value(self) -> NumericExpression:
        x = self.parts[0].min_value
        for i in range(1, len(self.parts)):
            x = add(x, self.parts[i].min_value)
        return x

    @property
    def max_value(self) -> NumericExpression:
        x = self.parts[0].max_value
        for i in range(1, len(self.parts)):
            x = add(x, self.parts[i].max_value)
        return x

    @property
    def exclude_min(self) -> bool:
        return any(p.exclude_min for p in self.parts)

    @property
    def exclude_max(self) -> bool:
        return any(p.exclude_max for p in self.parts)

    @property
    def minus_sign(self) -> bool:
        return False

    def add(self, value: NumericExpression) -> NumericExpression:
        parts = []
        constant = ZERO
        for part in self.parts:
            if part.is_literal:
                constant = add(constant, part)
            else:
                parts.append(part)
        if value.is_literal:
            assert isinstance(value, NumberLiteral)
            constant = add(constant, value)
        elif value.is_sum:
            assert isinstance(value, Sum)
            for part in value.parts:
                if part.is_literal:
                    constant = add(constant, part)
                else:
                    parts.append(part)
        else:
            parts.append(value)
        if not parts:
            return constant
        if constant.value == 0:
            return Sum(parts=parts) if len(parts) > 1 else parts[0]
        parts.append(constant)
        return Sum(parts=parts)

    def solve(self, **symbols: Mapping[str, NumericExpression]) -> NumericExpression:
        parts = []
        variables = []
        constant = ZERO
        # reduce literal values
        for part in self.parts:
            value = part.solve(**symbols)
            if value.is_literal:
                constant = add(constant, value)
            elif value.is_symbol:
                variables.append(value)
            else:
                parts.append(value)
        # cancel opposite symbols
        while variables:
            x = variables.pop()
            for i in range(len(variables)):
                y = variables[i]
                z = add(x, y)
                if z == ZERO:
                    del variables[i]
                    break
            else:
                parts.append(x)
        # fully resolved?
        if not parts:
            return constant
        if constant.value != 0:
            parts.append(constant)
        return Sum(parts=parts) if len(parts) > 1 else parts[0]

    def __str__(self) -> str:
        parts = [str(self.parts[0])]
        for i in range(1, len(self.parts)):
            part: NumericExpression = self.parts[i]
            if part.minus_sign:
                parts.append('-')
                s = str(part)
                assert s[0] == '-'
                parts.append(s[1:])
            else:
                parts.append(str(part))
        return f'({" + ".join(parts)})'


@frozen
class Product(NumericExpression):
    factors: Iterable[NumericExpression] = field(
        factory=tuple,
        converter=tuple,
        validator=deep_iterable(
            instance_of(NumericExpression),
            iterable_validator=min_len(2),
        )
    )

    @property
    def type(self) -> ExpressionType:
        return ExpressionType.PRODUCT

    @property
    def is_int(self) -> bool:
        return all(p.is_int for p in self.factors)

    @property
    def min_value(self) -> NumericExpression:
        x = self.factors[0].min_value
        for i in range(1, len(self.factors)):
            x = add(x, self.factors[i].min_value)
        return x

    @property
    def max_value(self) -> NumericExpression:
        x = self.factors[0].max_value
        for i in range(1, len(self.factors)):
            x = add(x, self.factors[i].max_value)
        return x

    @property
    def exclude_min(self) -> bool:
        return any(p.exclude_min for p in self.factors)

    @property
    def exclude_max(self) -> bool:
        return any(p.exclude_max for p in self.factors)

    @property
    def minus_sign(self) -> bool:
        return False

    def multiply(self, value: NumericExpression) -> NumericExpression:
        factors = []
        constant = ONE
        for factor in self.factors:
            if factor.is_literal:
                constant = multiply(constant, factor)
            else:
                factors.append(factor)
        if value.is_literal:
            assert isinstance(value, NumberLiteral)
            constant = multiply(constant, value)
        elif value.is_product:
            assert isinstance(value, Product)
            for factor in value.factors:
                if factor.is_literal:
                    constant = multiply(constant, factor)
                else:
                    factors.append(factor)
        else:
            factors.append(value)
        if not factors:
            return constant
        if constant.value == 0:
            return constant
        if constant.value != 1:
            factors.append(constant)
        return Product(factors=factors) if len(factors) > 1 else factors[0]

    def solve(self, **symbols: Mapping[str, NumericExpression]) -> NumericExpression:
        factors = []
        variables = []
        constant = ONE
        # reduce literal values
        for factor in self.factors:
            value = factor.solve(**symbols)
            if value.is_literal:
                constant = multiply(constant, value)
            elif value.is_symbol:
                variables.append(value)
            else:
                factors.append(value)
        # cancel opposite symbols
        while variables:
            x = variables.pop()
            for i in range(len(variables)):
                y = variables[i]
                p = multiply(x, y)
                if p == ONE:
                    del variables[i]
                    break
            else:
                factors.append(x)
        # fully resolved?
        if not factors or constant.value == 0:
            return constant
        if constant.value != 1:
            factors.append(constant)
        return Product(factors=factors) if len(factors) > 1 else factors[0]

    def __str__(self) -> str:
        return f'({" * ".join(map(str, self.factors))})'


def add(a: NumericExpression, b: NumericExpression) -> NumericExpression:
    if a.is_literal:
        assert isinstance(a, NumberLiteral)
        if a.value == 0:
            return b
        if b.is_literal:
            assert isinstance(b, NumberLiteral)
            return NumberLiteral(a.value + b.value)
        elif b.is_sum:
            assert isinstance(b, Sum)
            return b.add(a)
        return Sum((b, a))

    if b.is_literal:
        assert isinstance(b, NumberLiteral)
        if b.value == 0:
            return a
    elif a.is_symbol and b.is_symbol:
        assert isinstance(a, Symbol)
        assert isinstance(b, Symbol)
        if a.name == b.name and a.exponent == b.exponent and a.minus_sign != b.minus_sign:
            return ZERO
    elif a.is_sum:
        assert isinstance(a, Sum)
        return a.add(b)
    elif b.is_sum:
        assert isinstance(b, Sum)
        return b.add(a)
    return Sum((a, b))


def multiply(a: NumericExpression, b: NumericExpression) -> NumericExpression:
    if a.is_literal:
        assert isinstance(a, NumberLiteral)
        if a.value == 0:
            return a
        if a.value == 1:
            return b
        if a.value == -1:
            return b.negative()
        if b.is_literal:
            assert isinstance(b, NumberLiteral)
            return NumberLiteral(a.value * b.value)
        return Product((a, b))

    if b.is_literal:
        assert isinstance(b, NumberLiteral)
        if b.value == 0:
            return b
        if b.value == 1:
            return a
        if b.value == -1:
            return a.negative()
    elif a.is_symbol and b.is_symbol:
        assert isinstance(a, Symbol)
        assert isinstance(b, Symbol)
        if a.name == b.name:
            e: NumberLiteral = add(a.exponent, b.exponent)
            if e.is_literal and e.value == 0:
                return ONE
            return Symbol(
                name=a.name,
                exponent=e,
                minus_sign=(a.minus_sign ^ b.minus_sign),
                min_value=multiply(a.min_value, b.min_value),
                max_value=multiply(a.max_value, b.max_value),
                exclude_min=(a.exclude_min or b.exclude_min),
                exclude_max=(a.exclude_max or b.exclude_max),
            )
    elif a.is_product:
        assert isinstance(a, Product)
        return a.multiply(b)
    elif b.is_product:
        assert isinstance(b, Product)
        return b.multiply(a)
    return Product((a, b))
