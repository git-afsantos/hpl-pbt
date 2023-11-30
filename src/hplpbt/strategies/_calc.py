# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

###############################################################################
# Imports
###############################################################################

from typing import Final, Iterable, List, Mapping, Set, Union

from enum import auto, Enum

from attrs import evolve, field, frozen
from attrs.validators import deep_iterable, instance_of, min_len
from hpl.ast import (
    HplBinaryOperator,
    HplDataAccess,
    HplExpression,
    HplFunctionCall,
    HplQuantifier,
    HplUnaryOperator,
    HplValue,
    Not,
    Or,
)
from typeguard import typechecked

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
    POWER = auto()
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
    def is_power(self) -> bool:
        return self.type == ExpressionType.POWER

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

    @classmethod
    def zero(cls) -> 'NumberLiteral':
        return cls(0)

    @classmethod
    def one(cls) -> 'NumberLiteral':
        return cls(1)

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
    # overrides
    minus_sign: bool = False 
    min_value: NumericExpression = field(
        default=MINUS_INFINITY,
        converter=_ensure_num_expr,
        eq=False,
    )
    max_value: NumericExpression = field(default=INFINITY, converter=_ensure_num_expr, eq=False)
    exclude_min: bool = field(default=False, eq=False)
    exclude_max: bool = field(default=False, eq=False)

    @property
    def type(self) -> ExpressionType:
        return ExpressionType.SYMBOL

    def solve(self, **symbols: Mapping[str, NumericExpression]) -> NumericExpression:
        backup = self
        # solve constituent bits of information
        min_value = self.min_value.solve(**symbols)
        max_value = self.max_value.solve(**symbols)
        if not min_value.is_literal or not max_value.is_literal:
            # cannot ever get to an actual value
            return evolve(self, min_value=min_value, max_value=max_value)
        assert isinstance(min_value, NumberLiteral)
        assert isinstance(max_value, NumberLiteral)
        if min_value is not self.min_value:
            backup = evolve(self, min_value=min_value, max_value=max_value)
        elif max_value is not self.max_value:
            backup = evolve(self, min_value=min_value, max_value=max_value)

        # recursively substitute until reaching a dead end
        visited = {self.name}
        value = symbols.get(self.name, self)
        while value.is_symbol or value.is_reference:
            if value.name in visited:
                break  # undefined or cyclic
            visited.add(value.name)
            value = symbols.get(value.name, value)

        # got to another symbol?
        if value.is_reference:
            return backup if value.name == self.name else Symbol(value.name)
        if value.is_symbol:
            assert isinstance(value, Symbol)
            if value.name == self.name:
                return backup
            return evolve(
                value,
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
            x = -value.value if self.minus_sign else value.value
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
        return f'-{self.name}' if self.minus_sign else self.name


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
        parts = list(self.parts)
        parts.append(value)
        return Sum(parts).solve()

    def solve(self, **symbols: Mapping[str, NumericExpression]) -> NumericExpression:
        stack = []
        constant = ZERO
        # reduce literal values
        for part in self.parts:
            value = part.solve(**symbols)
            if value.is_literal:
                constant = add(constant, value)
            elif value.is_sum:
                stack.extend(value.parts)
            else:
                stack.append(value)
        # cancel opposite symbols
        parts = []
        while stack:
            x = stack.pop()
            for i in range(len(stack)):
                y = stack[i]
                z = add(x, y)
                if z == ZERO:
                    del stack[i]
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
        factors = list(self.factors)
        factors.append(value)
        return Product(factors).solve()

    def solve(self, **symbols: Mapping[str, NumericExpression]) -> NumericExpression:
        stack = []
        constant = ONE
        # reduce literal values
        for factor in self.factors:
            value = factor.solve(**symbols)
            if value.is_literal:
                constant = multiply(constant, value)
            elif value.is_product:
                stack.extend(value.factors)
            else:
                stack.append(value)
        # fully resolved?
        if constant.value == 0:
            return constant
        # cancel opposite symbols
        factors = []
        while stack:
            x = stack.pop()
            for i in range(len(stack)):
                y = stack[i]
                p = multiply(x, y)
                if p == ONE:
                    del stack[i]
                    break
            else:
                factors.append(x)
        # fully resolved?
        if not factors:
            return constant
        if constant.value != 1:
            factors.append(constant)
        return Product(factors=factors) if len(factors) > 1 else factors[0]

    def __str__(self) -> str:
        return f'({" * ".join(map(str, self.factors))})'


@frozen
class Power(NumericExpression):
    base: NumericExpression
    exponent: NumericExpression

    @property
    def type(self) -> ExpressionType:
        return ExpressionType.POWER

    @property
    def is_int(self) -> bool:
        if not self.exponent.is_int or not self.exponent.is_literal:
            return False
        assert isinstance(self.exponent, NumberLiteral)
        return self.base.is_int and self.exponent.value >= 0

    @property
    def min_value(self) -> NumericExpression:
        return Power(self.base.min_value, self.exponent.min_value).solve()

    @property
    def max_value(self) -> NumericExpression:
        return Power(self.base.max_value, self.exponent.max_value).solve()

    @property
    def exclude_min(self) -> bool:
        return self.base.exclude_min or self.exponent.exclude_min

    @property
    def exclude_max(self) -> bool:
        return self.base.exclude_max or self.exponent.exclude_max

    @property
    def minus_sign(self) -> bool:
        return False

    def solve(self, **symbols: Mapping[str, NumericExpression]) -> NumericExpression:
        base = self.base.solve(**symbols)
        exponent = self.exponent.solve(**symbols)

        if base.is_literal:
            assert isinstance(base, NumberLiteral)
            if base.value == 1:
                return base
            if exponent.is_literal:
                assert isinstance(exponent, NumberLiteral)
                return _literal(base.value ** exponent.value)

        elif exponent.is_literal:
            assert isinstance(exponent, NumberLiteral)
            if exponent.value == 0:
                return ONE
            if exponent.value == 1:
                return base

        elif base.is_power:
            assert isinstance(base, Power)
            exponent = multiply(base.exponent, exponent)
            base = base.base
            return Power(base, exponent)

        return self if base is self.base and exponent is self.exponent else Power(base, exponent)

    def __str__(self) -> str:
        return f'({self.base} ** {self.exponent})'


def _literal(x: Union[int, float, NumberLiteral]) -> NumberLiteral:
    return x if isinstance(x, NumberLiteral) else NumberLiteral(x)


def negative(x: NumericExpression) -> NumericExpression:
    # does not use other functions to avoid loops
    if x.is_literal:
        assert isinstance(x, NumberLiteral)
        return NumberLiteral(-x.value)
    if x.is_symbol:
        assert isinstance(x, Symbol)
        return evolve(
            x,
            minus_sign=(not x.minus_sign),
            min_value=negative(x.max_value),
            max_value=negative(x.min_value),
            exclude_min=x.exclude_max,
            exclude_max=x.exclude_min,
        )
    if x.is_sum:
        assert isinstance(x, Sum)
        return Sum(map(negative, x.parts))
    return Product(x, MINUS_ONE)


def inverse(x: NumericExpression) -> NumericExpression:
    # does not use other functions to avoid loops
    if x.is_literal:
        assert isinstance(x, NumberLiteral)
        return NumberLiteral(1 / x.value)
    if x.is_power:
        assert isinstance(x, Power)
        return Power(x.base, negative(x.exponent))
    return Power(x, MINUS_ONE)


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
        if a.name == b.name and a.minus_sign != b.minus_sign:
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
            return negative(b)
        if b.is_literal:
            assert isinstance(b, NumberLiteral)
            return NumberLiteral(a.value * b.value)
        return Product((b, a))

    if b.is_literal:
        assert isinstance(b, NumberLiteral)
        if b.value == 0:
            return b
        if b.value == 1:
            return a
        if b.value == -1:
            return negative(a)
        return Product((a, b))

    if a.is_product:
        assert isinstance(a, Product)
        return a.multiply(b)
    if b.is_product:
        assert isinstance(b, Product)
        return b.multiply(a)

    e1 = ONE
    e2 = ONE
    if a.is_power:
        assert isinstance(a, Power)
        e1 = a.exponent
        a = a.base
    if b.is_power:
        assert isinstance(b, Power)
        e2 = b.exponent
        b = b.base

    if a == b:
        return Power(a, add(e1, e2)).solve()
    if a.is_symbol and b.is_symbol:
        assert isinstance(a, Symbol)
        assert isinstance(b, Symbol)
        if a.name == b.name:
            x = Symbol(
                name=a.name,
                minus_sign=(a.minus_sign ^ b.minus_sign),
                min_value=multiply(a.min_value, b.min_value),
                max_value=multiply(a.max_value, b.max_value),
                exclude_min=(a.exclude_min or b.exclude_min),
                exclude_max=(a.exclude_max or b.exclude_max),
            )
            return Power(x, add(e1, e2)).solve()
    p = Product((a, b))
    return Power(p, e1) if e1 == e2 and e1 != ONE else p


################################################################################
# Interface
################################################################################


@typechecked
def solve_constraints(conditions: Iterable[HplExpression]) -> List[HplExpression]:
    # assumes that `conditions` is a list of expressions in canonical/simple form
    # e.g., 'x > y + 20', or 'z = w'
    new_conditions = []
    symbols = _build_symbol_table(conditions)
    for phi in conditions:
        assert phi.can_be_bool, str(phi)
        if phi.is_operator:
            if isinstance(phi, HplBinaryOperator):
                assert not phi.operator.is_implies, str(phi)
                assert not phi.operator.is_iff, str(phi)
                if phi.operator.is_and:
                    new_conditions.extend(solve_constraints((phi.operand1, phi.operand2)))
                elif phi.operator.is_or:
                    a = solve_constraints((phi.operand1,))
                    b = solve_constraints((phi.operand2,))
                    new_conditions.append(Or(a, b))
                else:
                    if not phi.operator.is_comparison and not phi.operator.is_inclusion:
                        raise TypeError(f'unknown expression type: {phi}')
                    if phi.operator.is_inequality:
                        continue

                    a = _convert_arithmetic(phi.operand1)
                    b = _convert_arithmetic(phi.operand2)
                    if not a.is_symbol:
                        continue
                    assert isinstance(a, Symbol)

                    value = symbols.get(a.name)
                    if value is None:
                        value = a
                        symbols[a.name] = a
                    if phi.operator.is_equality:
                        # TODO aliases etc
                        if not value.is_symbol or value.name != a.name:
                            raise ContradictionError(str(phi))
                        symbols[a.name] = b
                    elif phi.operator.is_less_than:
                        if value.is_symbol:
                            value = evolve(value, max_value=b, exclude_max=True)
                            symbols[a.name] = value
                    elif phi.operator.is_less_than_eq:
                        if value.is_symbol:
                            value = evolve(value, max_value=b, exclude_max=False)
                            symbols[a.name] = value
                    elif phi.operator.is_greater_than:
                        if value.is_symbol:
                            value = evolve(value, min_value=b, exclude_min=True)
                            symbols[a.name] = value
                    elif phi.operator.is_greater_than_eq:
                        if value.is_symbol:
                            value = evolve(value, min_value=b, exclude_min=False)
                            symbols[a.name] = value
            else:
                assert isinstance(phi, HplUnaryOperator), str(phi)
                assert phi.operator.is_not, str(phi)
                new_conditions.append(Not(solve_constraints((phi.operand))))
    return new_conditions


def _build_symbol_table(conditions: Iterable[HplExpression]) -> Mapping[str, Symbol]:
    symbols = {}
    for phi in conditions:
        _find_symbols(phi, symbols)
    return symbols


def _find_symbols(expr: HplExpression, symbols: Mapping[str, Symbol]):
    if expr.is_operator:
        if isinstance(expr, HplBinaryOperator):
            _find_symbols(expr.operand1, symbols)
            _find_symbols(expr.operand2, symbols)
        elif isinstance(expr, HplUnaryOperator):
            _find_symbols(expr.operand, symbols)
    elif expr.is_quantifier:
        assert isinstance(expr, HplQuantifier)
        _find_symbols(expr.condition, symbols)
    elif not expr.can_be_number:
        return

    if expr.is_function_call:
        assert isinstance(expr, HplFunctionCall)
        for arg in expr.arguments:
            _find_symbols(arg, symbols)
    elif expr.is_value:
        assert isinstance(expr, HplValue)
        if expr.is_variable:
            symbols[expr.token] = Symbol(expr.token)
    elif expr.is_accessor:
        assert isinstance(expr, HplDataAccess)
        name = str(expr)
        symbols[name] = Symbol(name)


def _convert_arithmetic(expr: HplExpression) -> NumericExpression:
    assert expr.can_be_number, str(expr)
    if isinstance(expr, HplBinaryOperator):
        a = _convert_arithmetic(expr.operand1)
        b = _convert_arithmetic(expr.operand2)
        if expr.operator.is_plus:
            return Sum((a, b))
        if expr.operator.is_minus:
            return Sum((a, negative(b)))
        if expr.operator.is_times:
            return Product((a, b))
        if expr.operator.is_division:
            return Product(a, inverse(b))
        if expr.operator.is_power:
            return Power(a, b)
        raise TypeError(f'unknown arithmetic operator: {expr}')
    elif isinstance(expr, HplUnaryOperator):
        assert expr.operator.is_minus, str(expr)
        a = _convert_arithmetic(expr.operand)
        return negative(a)
    elif isinstance(expr, HplValue):
        if expr.is_literal:
            return NumberLiteral(expr.value)
        if expr.is_variable:
            return Symbol(expr.token)
    elif isinstance(expr, HplDataAccess):
        return Symbol(str(expr))
    raise TypeError(f'unknown arithmetic expression: {expr}')
