# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

###############################################################################
# Imports
###############################################################################

from typing import Any, Final, Iterable, Optional, Set, Tuple

from enum import auto, Enum

from attrs import field, frozen
from attrs.validators import deep_iterable, instance_of, optional
from hpl.ast import (
    HplArrayAccess,
    HplBinaryOperator,
    HplDataAccess,
    HplExpression,
    HplFieldAccess,
    HplFunctionCall,
    HplQuantifier,
    HplUnaryOperator,
)
from hplrv.gen import TemplateRenderer
from typeguard import check_type, typechecked

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
    def is_literal(self) -> bool:
        return self.type == ExpressionType.LITERAL

    @property
    def is_reference(self) -> bool:
        return self.type == ExpressionType.REFERENCE

    @property
    def is_unary_operator(self) -> bool:
        return self.type == ExpressionType.UNARY_OPERATOR

    @property
    def is_binary_operator(self) -> bool:
        return self.type == ExpressionType.BINARY_OPERATOR

    @property
    def is_function_call(self) -> bool:
        return self.type == ExpressionType.CALL

    @property
    def is_iterator(self) -> bool:
        return self.type == ExpressionType.ITERATOR

    @property
    def is_dot_access(self) -> bool:
        return self.type == ExpressionType.DOT_ACCESS

    @property
    def is_key_access(self) -> bool:
        return self.type == ExpressionType.KEY_ACCESS

    @property
    def is_value_draw(self) -> bool:
        return self.type == ExpressionType.DRAW

    @property
    def can_be_number(self) -> bool:
        return True

    def references(self) -> Set[str]:
        raise NotImplementedError()

    def eq(self, other: 'Expression') -> bool:
        return self == other


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

    @property
    def can_be_number(self) -> bool:
        return self.is_int or self.is_float

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

    def references(self) -> Set[str]:
        return set()

    def __str__(self) -> str:
        return repr(self.value)


@frozen
class Reference(Expression):
    variable: str

    @property
    def type(self) -> ExpressionType:
        return ExpressionType.REFERENCE

    def references(self) -> Set[str]:
        return {self.variable}

    def __str__(self) -> str:
        return self.variable


@frozen
class UnaryOperator(Expression):
    token: str
    operand: Expression

    @property
    def type(self) -> ExpressionType:
        return ExpressionType.UNARY_OPERATOR

    @property
    def can_be_number(self) -> bool:
        return self.token == '-' or self.token == '+' or self.token == '~'

    def references(self) -> Set[str]:
        return self.operand.references()

    def eq(self, other: Expression) -> bool:
        if super().eq(other):
            return True
        if isinstance(other, UnaryOperator):
            return self.token == other.token and self.operand.eq(other.operand)
        return False

    def __str__(self) -> str:
        ws: str = ' ' if self.token.isalpha() else ''
        return f'({self.token}{ws}{self.operand})'


_COMMUTATIVE_OPERATORS: Final[Iterable[str]] = ('+', '*', '==', '!=')


@frozen
class BinaryOperator(Expression):
    token: str
    operand1: Expression
    operand2: Expression

    @property
    def type(self) -> ExpressionType:
        return ExpressionType.BINARY_OPERATOR

    @property
    def can_be_number(self) -> bool:
        return (
            self.token == '+'
            or self.token == '-'
            or self.token == '*'
            or self.token == '/'
            or self.token == '%'
            or self.token == '**'
            or self.token == '=='
            or self.token == '!='
            or self.token == '<'
            or self.token == '<='
            or self.token == '>'
            or self.token == '>='
            or self.token == '&'
            or self.token == '|'
            or self.token == '^'
        )

    def references(self) -> Set[str]:
        return self.operand1.references() | self.operand2.references()

    def eq(self, other: Expression) -> bool:
        if super().eq(other):
            return True
        if isinstance(other, BinaryOperator):
            if self.token != other.token:
                return False
            if self.operand1.eq(other.operand1) and self.operand2.eq(other.operand2):
                return True
            if self.token not in _COMMUTATIVE_OPERATORS:
                return False
            return self.operand1.eq(other.operand2) and self.operand2.eq(other.operand1)
        return False

    def __str__(self) -> str:
        return f'({self.operand1} {self.token} {self.operand2})'


@frozen
class FunctionCall(Expression):
    function: str
    arguments: Iterable[Expression] = field(factory=tuple, converter=tuple)
    keyword_arguments: Iterable[Tuple[str, Expression]] = field(factory=tuple, converter=tuple)

    @property
    def type(self) -> ExpressionType:
        return ExpressionType.CALL

    @property
    def can_be_number(self) -> bool:
        return (
            self.function == 'len'
            or self.function == 'abs'
            or self.function == 'max'
            or self.function == 'min'
            or self.function == 'sin'
            or self.function == 'cos'
            or self.function == 'tan'
            or self.function == 'asin'
            or self.function == 'acos'
            or self.function == 'atan'
            or self.function == 'atan2'
        )

    def references(self) -> Set[str]:
        names = set()
        for arg in self.arguments:
            names.update(arg.references())
        for _name, arg in self.keyword_arguments:
            names.update(arg.references())
        return names

    def eq(self, other: Expression) -> bool:
        if super().eq(other):
            return True
        if isinstance(other, FunctionCall):
            if self.function != other.function:
                return False
            if len(self.arguments) != len(other.arguments):
                return False
            if len(self.keyword_arguments) != len(other.keyword_arguments):
                return False
            for i in range(len(self.arguments)):
                if not self.arguments[i].eq(other.arguments[i]):
                    return False
            kwargs = dict(other.keyword_arguments)
            for name, value1 in self.keyword_arguments:
                value2 = kwargs.get(name)
                if not value1.eq(value2):
                    return False
            return True
        return False

    def __str__(self) -> str:
        args = list(map(str, self.arguments))
        args.extend(f'{key}={arg}' for key, arg in self.keyword_arguments)
        args = ', '.join(args)
        return f'{self.function}({args})'


@frozen
class IteratorExpression(Expression):
    expression: Expression
    variable: str
    domain: Expression
    condition: Optional[Expression] = None

    @property
    def type(self) -> ExpressionType:
        return ExpressionType.ITERATOR

    @property
    def can_be_number(self) -> bool:
        return False

    def eq(self, other: Expression) -> bool:
        if super().eq(other):
            return True
        if isinstance(other, IteratorExpression):
            if not self.expression.eq(other.expression):
                return False
            if self.variable != other.variable:
                return False
            if not self.domain.eq(other.domain):
                return False
            if (self.condition is None) is not (other.condition is None):
                return False
            if self.condition is not None and not self.condition.eq(other.condition):
                return False
            return True
        return False

    def __str__(self) -> str:
        if self.condition is None:
            return f'{self.expression} for {self.variable} in {self.domain}'
        return f'{self.expression} for {self.variable} in {self.domain} if {self.condition}'


@frozen
class DotAccess(Expression):
    expression: Expression
    name: str

    @property
    def type(self) -> ExpressionType:
        return ExpressionType.DOT_ACCESS

    def eq(self, other: Expression) -> bool:
        if super().eq(other):
            return True
        if isinstance(other, DotAccess):
            if not self.expression.eq(other.expression):
                return False
            if self.name != other.name:
                return False
            return True
        return False

    def __str__(self) -> str:
        return f'{self.expression}.{self.name}'


@frozen
class KeyAccess(Expression):
    expression: Expression
    key: Expression

    @property
    def type(self) -> ExpressionType:
        return ExpressionType.KEY_ACCESS

    def eq(self, other: Expression) -> bool:
        if super().eq(other):
            return True
        if isinstance(other, DotAccess):
            if not self.expression.eq(other.expression):
                return False
            if not self.key.eq(other.key):
                return False
            return True
        return False

    def __str__(self) -> str:
        return f'{self.expression}[{self.key}]'


@frozen
class ValueDraw(Expression):
    strategy: 'DataStrategy'

    @property
    def type(self) -> ExpressionType:
        return ExpressionType.DRAW

    @property
    def can_be_number(self) -> bool:
        if self.strategy.is_int or self.strategy.is_float:
            return True
        if self.strategy.is_constant:
            assert isinstance(self.strategy, ConstantValue)
            return self.strategy.expression.can_be_number
        if self.strategy.is_sample:
            assert isinstance(self.strategy, RandomSample)
            return any(value.can_be_number for value in self.strategy.elements)
        return False

    def references(self) -> Set[str]:
        return self.strategy.dependencies()

    def __str__(self) -> str:
        return f'draw({self.strategy})'


@typechecked
def expression_from_hpl(expr: HplExpression) -> Expression:
    if expr.is_value:
        if expr.is_literal:
            return Literal(expr.value)
        if expr.is_set:
            return Literal(tuple(map(expression_from_hpl, expr.values)))
        if expr.is_range:
            lb = convert_to_int(expression_from_hpl(expr.min_value))
            if expr.exclude_min:
                lb = BinaryOperator('+', lb, Literal(1))
            ub = convert_to_int(expression_from_hpl(expr.max_value))
            if not expr.exclude_max:
                ub = BinaryOperator('+', ub, Literal(1))
            return FunctionCall('range', (lb, ub))
        if expr.is_variable:
            return Reference(expr.name)
        if expr.is_this_msg:
            return Reference('msg')
    elif expr.is_operator:
        if isinstance(expr, HplUnaryOperator):
            a = expression_from_hpl(expr.operand)
            return UnaryOperator(expr.operator.token, a)
        elif isinstance(expr, HplBinaryOperator):
            op = '==' if expr.operator.is_equality else expr.operator.token
            a = expression_from_hpl(expr.operand1)
            b = expression_from_hpl(expr.operand2)
            return BinaryOperator(op, a, b)
    elif expr.is_function_call:
        assert isinstance(expr, HplFunctionCall)
        args = tuple(map(expression_from_hpl, expr.arguments))
        return FunctionCall(expr.function.name, arguments=args)
    elif expr.is_quantifier:
        assert isinstance(expr, HplQuantifier)
        phi = expression_from_hpl(expr.condition)
        domain = expression_from_hpl(expr.domain)
        it = IteratorExpression(phi, expr.variable, domain)
        if expr.is_universal:
            return FunctionCall('all', arguments=(it,))
        else:
            return FunctionCall('any', arguments=(it,))
    elif expr.is_accessor:
        assert isinstance(expr, HplDataAccess)
        obj = expression_from_hpl(expr.object)
        if expr.is_field:
            assert isinstance(expr, HplFieldAccess)
            return DotAccess(obj, expr.field)
        elif expr.is_indexed:
            assert isinstance(expr, HplArrayAccess)
            return KeyAccess(obj, expression_from_hpl(expr.index))
    raise ValueError(f'unable to handle HplExpression: {expr!r}')


@typechecked
def convert_to_int(expr: Expression) -> Expression:
    if expr.is_literal:
        assert isinstance(expr, Literal)
        if expr.is_int:
            return expr
        if expr.is_float and expr.value == int(expr.value):
            return Literal(int(expr.value))
    if expr.is_function_call:
        if expr.function == 'len':
            return expr
    if expr.is_unary_operator:
        assert isinstance(expr, UnaryOperator)
        if expr.token == '-':
            if expr.operand.is_literal:
                assert isinstance(expr.operand, Literal)
                return convert_to_int(Literal(-expr.operand.value))
        if expr.token == '+':
            if expr.operand.is_literal:
                assert isinstance(expr.operand, Literal)
                return convert_to_int(Literal(+expr.operand.value))
        if expr.token == '~':
            if expr.operand.is_literal:
                assert isinstance(expr.operand, Literal)
                return convert_to_int(Literal(~expr.operand.value))
    if expr.is_binary_operator:
        assert isinstance(expr, BinaryOperator)
        if expr.operand1.is_literal and expr.operand2.is_literal:
            assert isinstance(expr.operand1, Literal)
            assert isinstance(expr.operand2, Literal)
            f = lambda x: convert_to_int(Literal(x))
            if expr.token == '+':
                return f(expr.operand1.value + expr.operand2.value)
            if expr.token == '-':
                return f(expr.operand1.value - expr.operand2.value)
            if expr.token == '*':
                return f(expr.operand1.value * expr.operand2.value)
            if expr.token == '/':
                return f(expr.operand1.value / expr.operand2.value)
            if expr.token == '%':
                return f(expr.operand1.value % expr.operand2.value)
            if expr.token == '**':
                return f(expr.operand1.value ** expr.operand2.value)
            if expr.token == '&':
                return f(expr.operand1.value & expr.operand2.value)
            if expr.token == '|':
                return f(expr.operand1.value | expr.operand2.value)
            if expr.token == '^':
                return f(expr.operand1.value ^ expr.operand2.value)
    return FunctionCall('int', (expr,))


# @classmethod
# def plus(cls, a: Expression, b: Expression) -> Expression:
#     if a.is_literal:
#         assert isinstance(a, Literal)
#         assert a.is_int or a.is_float
#         if a.value == 0:
#             return b
#         if b.is_literal:
#             assert isinstance(b, Literal)
#             assert b.is_int or b.is_float
#             if a.value == -b.value:
#                 return Literal.zero()
#     if b.is_literal:
#         assert isinstance(b, Literal)
#         if b.value == 0:
#             return a
#         if a.is_literal:
#             assert isinstance(a, Literal)
#             assert a.is_int or a.is_float
#             if a.value == -b.value:
#                 return Literal.zero()
#     return cls('+', a, b)


################################################################################
# Strategy AST - Value Generators
################################################################################


class DataStrategyType(Enum):
    CONSTANT = auto()
    BOOL = auto()
    INT = auto()
    FLOAT = auto()
    STRING = auto()
    ARRAY = auto()
    SAMPLE = auto()
    SPECIAL = auto()


@frozen
class DataStrategy:
    """
    Base class that represents calls to strategies within a function's body,
    for example, to generate arguments and other variables.
    """

    @property
    def type(self) -> DataStrategyType:
        raise NotImplementedError()

    @property
    def is_constant(self) -> bool:
        return self.type == DataStrategyType.CONSTANT

    @property
    def is_bool(self) -> bool:
        return self.type == DataStrategyType.BOOL

    @property
    def is_int(self) -> bool:
        return self.type == DataStrategyType.INT

    @property
    def is_float(self) -> bool:
        return self.type == DataStrategyType.FLOAT

    @property
    def is_string(self) -> bool:
        return self.type == DataStrategyType.STRING

    @property
    def is_array(self) -> bool:
        return self.type == DataStrategyType.ARRAY

    @property
    def is_sample(self) -> bool:
        return self.type == DataStrategyType.SAMPLE

    def dependencies(self) -> Set[str]:
        return set()

    def is_value_impossible(self, expr: Expression) -> bool:
        """Return True only if absolutely sure."""
        raise NotImplementedError(f'is_value_impossible({expr!r})')


@frozen
class ConstantValue(DataStrategy):
    expression: Expression = field(validator=instance_of(Expression))

    @property
    def type(self) -> DataStrategyType:
        return DataStrategyType.CONSTANT

    @property
    def value(self) -> Expression:
        return self.expression  # alias

    def dependencies(self) -> Set[str]:
        return self.expression.references()

    def is_value_impossible(self, expr: Expression) -> bool:
        if expr.is_iterator:
            return True
        if expr.is_literal:
            assert isinstance(expr, Literal)
            if self.expression.is_literal:
                assert isinstance(self.expression, Literal)
                return self.expression.value != expr.value
        elif expr.is_reference:
            pass
        elif expr.is_function_call:
            assert isinstance(expr, FunctionCall)
            if self.expression.is_literal:
                assert isinstance(self.expression, Literal)
                x = self.expression
                if expr.function == 'len':
                    return not x.is_int or x.value < 0
                if expr.function == 'abs':
                    return not (x.is_int or x.is_float) or x.value < 0
        elif expr.is_value_draw:
            pass
        return False

    def __str__(self) -> str:
        return f'just({self.expression})'


@frozen
class RandomBool(DataStrategy):
    @property
    def type(self) -> DataStrategyType:
        return DataStrategyType.BOOL

    def dependencies(self) -> Set[str]:
        return set()

    def is_value_impossible(self, expr: Expression) -> bool:
        if expr.is_iterator:
            return True
        if expr.is_literal:
            return not expr.is_bool
        return False

    def __str__(self) -> str:
        return 'booleans()'


@typechecked
def _maybe_convert_to_int(expr: Optional[Expression]) -> Optional[Expression]:
    if expr is None:
        return None
    return convert_to_int(expr)


@frozen
class RandomInt(DataStrategy):
    # min_value: Expression = field(factory=Literal.none, validator=instance_of(Expression))
    # max_value: Expression = field(factory=Literal.none, validator=instance_of(Expression))
    min_value: Optional[Expression] = field(
        default=None,
        validator=optional(instance_of(Expression)),
        converter=_maybe_convert_to_int,
    )
    max_value: Optional[Expression] = field(
        default=None,
        validator=optional(instance_of(Expression)),
        converter=_maybe_convert_to_int,
    )

    @property
    def type(self) -> DataStrategyType:
        return DataStrategyType.INT

    @classmethod
    def uint(cls) -> 'RandomInt':
        min_value = Literal(0)
        return cls(min_value=min_value)

    @classmethod
    def uint8(cls) -> 'RandomInt':
        min_value = Literal(0)
        max_value = Literal(255)
        return cls(min_value=min_value, max_value=max_value)

    @classmethod
    def uint16(cls) -> 'RandomInt':
        min_value = Literal(0)
        max_value = Literal(65535)
        return cls(min_value=min_value, max_value=max_value)

    @classmethod
    def uint32(cls) -> 'RandomInt':
        min_value = Literal(0)
        max_value = Literal(4294967295)
        return cls(min_value=min_value, max_value=max_value)

    @classmethod
    def uint64(cls) -> 'RandomInt':
        min_value = Literal(0)
        max_value = Literal(18446744073709551615)
        return cls(min_value=min_value, max_value=max_value)

    @classmethod
    def int8(cls) -> 'RandomInt':
        min_value = Literal(-128)
        max_value = Literal(127)
        return cls(min_value=min_value, max_value=max_value)

    @classmethod
    def int16(cls) -> 'RandomInt':
        min_value = Literal(-32768)
        max_value = Literal(32767)
        return cls(min_value=min_value, max_value=max_value)

    @classmethod
    def int32(cls) -> 'RandomInt':
        min_value = Literal(-2147483648)
        max_value = Literal(2147483647)
        return cls(min_value=min_value, max_value=max_value)

    @classmethod
    def int64(cls) -> 'RandomInt':
        min_value = Literal(-9223372036854775808)
        max_value = Literal(9223372036854775807)
        return cls(min_value=min_value, max_value=max_value)

    def dependencies(self) -> Set[str]:
        dependencies = set()
        if self.min_value is not None:
            dependencies.update(self.min_value.references())
        if self.max_value is not None:
            dependencies.update(self.max_value.references())
        return dependencies

    def is_value_impossible(self, expr: Expression) -> bool:
        if expr.is_iterator:
            return True
        if expr.is_literal:
            if expr.is_float:
                k: int = int(expr.value)
                if k != expr.value:
                    return True
            elif expr.is_int:
                k = expr.value
            else:
                return True
            if self.min_value is not None and self.min_value.is_literal:
                return k < self.min_value.value
            if self.max_value is not None and self.max_value.is_literal:
                return k > self.max_value.value
        elif expr.is_function_call:
            if expr.function == 'len' or expr.function == 'abs':
                if self.max_value is not None and self.max_value.is_literal:
                    return self.max_value.value < 0
        return False

    def __str__(self) -> str:
        args = []
        if self.min_value is not None:
            args.append(f'min_value={self.min_value}')
        if self.max_value is not None:
            args.append(f'max_value={self.max_value}')
        args = ', '.join(args)
        return f'integers({args})'


@frozen
class RandomFloat(DataStrategy):
    # min_value: Expression = field(factory=Literal.none, validator=instance_of(Expression))
    # max_value: Expression = field(factory=Literal.none, validator=instance_of(Expression))
    min_value: Optional[Expression] = field(
        default=None,
        validator=optional(instance_of(Expression)),
    )
    max_value: Optional[Expression] = field(
        default=None,
        validator=optional(instance_of(Expression)),
    )
    # allow_nan: bool = True
    # allow_infinity: bool = True
    # allow_subnormal: bool = True
    # width: int = 64
    # exclude_min: bool = False
    # exclude_max: bool = False

    @property
    def type(self) -> DataStrategyType:
        return DataStrategyType.FLOAT

    @classmethod
    def float32(cls) -> 'RandomFloat':
        min_value = Literal(-3.3999999521443642e38)
        max_value = Literal(3.3999999521443642e38)
        return cls(min_value=min_value, max_value=max_value)

    @classmethod
    def float64(cls) -> 'RandomFloat':
        min_value = Literal(-1.7e308)
        max_value = Literal(1.7e308)
        return cls(min_value=min_value, max_value=max_value)

    def dependencies(self) -> Set[str]:
        dependencies = set()
        if self.min_value is not None:
            dependencies.update(self.min_value.references())
        if self.max_value is not None:
            dependencies.update(self.max_value.references())
        return dependencies

    def is_value_impossible(self, expr: Expression) -> bool:
        if expr.is_iterator:
            return True
        if expr.is_literal:
            if not expr.is_float and not expr.is_int:
                return True
            k = expr.value
            if self.min_value is not None and self.min_value.is_literal:
                return k < self.min_value.value
            if self.max_value is not None and self.max_value.is_literal:
                return k > self.max_value.value
        elif expr.is_function_call:
            if expr.function == 'len' or expr.function == 'abs':
                if self.max_value is not None and self.max_value.is_literal:
                    return self.max_value.value < 0
        return False

    def __str__(self) -> str:
        # min_value=None
        # max_value=None
        # allow_nan=None
        # allow_infinity=None
        # allow_subnormal=None
        # width=64
        # exclude_min=False
        # exclude_max=False
        args = []
        if self.min_value is not None:
            args.append(f'min_value={self.min_value}')
        if self.max_value is not None:
            args.append(f'max_value={self.max_value}')
        args = ', '.join(args)
        return f'floats({args})'


@frozen
class RandomString(DataStrategy):
    # min_size: Expression = field(factory=Literal.zero, validator=instance_of(Expression))
    # max_size: Expression = field(factory=Literal.none, validator=instance_of(Expression))
    min_size: Optional[Expression] = field(
        default=None,
        validator=optional(instance_of(Expression)),
    )
    max_size: Optional[Expression] = field(
        default=None,
        validator=optional(instance_of(Expression)),
    )
    # alphabet: Optional[Expression] = None

    @property
    def type(self) -> DataStrategyType:
        return DataStrategyType.STRING

    def dependencies(self) -> Set[str]:
        dependencies = set()
        if self.min_size is not None:
            dependencies.update(self.min_size.references())
        if self.max_size is not None:
            dependencies.update(self.max_size.references())
        return dependencies

    def is_value_impossible(self, expr: Expression) -> bool:
        if expr.is_iterator:
            return True
        if expr.is_literal:
            return not expr.is_string
        elif expr.is_function_call:
            return expr.function in (
                'len',
                'abs',
                'bool',
                'int',
                'float',
                'min',
                'max',
                'sin',
                'cos',
                'tan',
                'asin',
                'acos',
                'atan',
                'atan2',
            )
        return False

    def __str__(self) -> str:
        # alphabet=characters(codec='utf-8')
        # min_size=0
        # max_size=None
        args = []
        if self.min_size is not None:
            args.append(f'min_size={self.min_size}')
        if self.max_size is not None:
            args.append(f'max_size={self.max_size}')
        args = ', '.join(args)
        return f'text({args})'


@frozen
class RandomArray(DataStrategy):
    elements: DataStrategy = field(validator=instance_of(DataStrategy))
    # min_size: Expression = field(factory=Literal.zero, validator=instance_of(Expression))
    # max_size: Expression = field(factory=Literal.none, validator=instance_of(Expression))
    # unique: bool = False
    min_size: Optional[Expression] = field(
        default=None,
        validator=optional(instance_of(Expression)),
    )
    max_size: Optional[Expression] = field(
        default=None,
        validator=optional(instance_of(Expression)),
    )

    @property
    def type(self) -> DataStrategyType:
        return DataStrategyType.ARRAY

    def dependencies(self) -> Set[str]:
        dependencies = set()
        if self.min_size is not None:
            dependencies.update(self.min_size.references())
        if self.max_size is not None:
            dependencies.update(self.max_size.references())
        return dependencies

    def is_value_impossible(self, expr: Expression) -> bool:
        if expr.is_iterator:
            return True
        if expr.is_literal:
            return not expr.is_array
        return False

    def __str__(self) -> str:
        # min_size=0
        # max_size=None
        # unique_by=None
        # unique=False
        args = [str(self.elements)]
        if self.min_size is not None:
            args.append(f'min_size={self.min_size}')
        if self.max_size is not None:
            args.append(f'max_size={self.max_size}')
        args = ', '.join(args)
        return f'lists({args})'


@frozen
class RandomSample(DataStrategy):
    elements: Iterable[Expression] = field(
        factory=tuple,
        converter=tuple,
        validator=deep_iterable(instance_of(Expression)),
    )

    @property
    def type(self) -> DataStrategyType:
        return DataStrategyType.SAMPLE

    def dependencies(self) -> Set[str]:
        names = set()
        for expresion in self.elements:
            names.update(expresion.references())
        return names

    def add(self, element: Expression) -> 'RandomSample':
        for el in self.elements:
            if el.eq(element):
                return self
        return RandomSample(self.elements + (element,))

    def remove(self, element: Expression) -> 'RandomSample':
        elements = [el for el in self.elements if not el.eq(element)]
        return self if len(elements) == len(self.elements) else RandomSample(elements)

    def is_value_impossible(self, expr: Expression) -> bool:
        if expr.is_iterator:
            return True
        if not all(e.is_literal for e in self.elements):
            return False
        if expr.is_literal:
            return expr.value not in self.elements
        return False

    def __str__(self) -> str:
        return f'sampled_from({self.elements})'


@frozen
class RandomSpecial(DataStrategy):
    name: str

    @property
    def type(self) -> DataStrategyType:
        return DataStrategyType.SPECIAL

    def dependencies(self) -> Set[str]:
        names = set()
        # for expresion in self.arguments:
        #     names.update(expresion.references())
        return names

    def is_value_impossible(self, _expr: Expression) -> bool:
        return False

    def __str__(self) -> str:
        return f'gen_{self.name}()'


################################################################################
# Strategy AST - Statements
################################################################################


class StatementType(Enum):
    ASSIGN = auto()
    ASSUME = auto()
    LOOP = auto()
    BLOCK = auto()


@frozen
class Statement:
    @property
    def type(self) -> StatementType:
        raise NotImplementedError()

    @property
    def is_assignment(self) -> bool:
        return self.type == StatementType.ASSIGN

    @property
    def is_assumption(self) -> bool:
        return self.type == StatementType.ASSUME

    @property
    def is_loop(self) -> bool:
        return self.type == StatementType.LOOP

    @property
    def is_block(self) -> bool:
        return self.type == StatementType.BLOCK

    def dependencies(self) -> Set[str]:
        return set()

    def assignments(self) -> Set[str]:
        return set()

    def merge(self, other: 'Statement') -> 'Statement':
        c1 = type(self).__name__
        c2 = type(other).__name__
        raise TypeError(f'unable to merge {c1} with {c2}')


################################################################################
# Strategy AST - Assignment
################################################################################


@frozen
class Assignment(Statement):
    variable: str = field(validator=instance_of(str))
    expression: Expression = field(validator=instance_of(Expression))

    @property
    def type(self) -> StatementType:
        return StatementType.ASSIGN

    @classmethod
    def draw(cls, variable: str, strategy: DataStrategy) -> 'Assignment':
        strategy = check_type(strategy, DataStrategy)
        expression = ValueDraw(strategy)
        return cls(variable, expression)

    def dependencies(self) -> Set[str]:
        return self.expression.references()

    def assignments(self) -> Set[str]:
        return {self.variable}

    def __str__(self) -> str:
        return f'{self.variable} = {self.expression}'


################################################################################
# Strategy AST - Assumption
################################################################################


@frozen
class Assumption(Statement):
    expression: HplExpression

    @property
    def type(self) -> StatementType:
        return StatementType.ASSUME

    def dependencies(self) -> Set[str]:
        return self.expression.external_references()

    def __str__(self) -> str:
        r = TemplateRenderer.from_pkg_data()
        data = {'expression': self.expression, 'message': 'msg'}
        expression = r.render_template('py/expression.py.jinja', data, strip=True)
        expression = expression.replace('v_', '')
        return f'assume{expression}' if expression.startswith('(') else f'assume({expression})'
