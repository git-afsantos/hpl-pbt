# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

###############################################################################
# Imports
###############################################################################

from typing import Any, Iterable, Mapping, Optional, Set, Tuple

from enum import auto, Enum

from attrs import field, frozen
from attrs.validators import deep_iterable, instance_of, optional
from hpl.ast import HplExpression
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
    def is_value_draw(self) -> bool:
        return self.type == ExpressionType.DRAW

    def references(self) -> Set[str]:
        raise NotImplementedError()


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

    def references(self) -> Set[str]:
        return self.operand.references()

    def __str__(self) -> str:
        ws: str = ' ' if self.token.isalpha() else ''
        return f'({self.token}{ws}{self.operand})'


@frozen
class BinaryOperator(Expression):
    token: str
    operand1: Expression
    operand2: Expression

    @property
    def type(self) -> ExpressionType:
        return ExpressionType.BINARY_OPERATOR

    def references(self) -> Set[str]:
        return self.operand1.references() | self.operand2.references()

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

    def references(self) -> Set[str]:
        names = set()
        for arg in self.arguments:
            names.update(arg.references())
        for _name, arg in self.keyword_arguments:
            names.update(arg.references())
        return names

    def __str__(self) -> str:
        args = list(map(str, self.arguments))
        args.extend(f'{key}={arg}' for key, arg in self.keyword_arguments)
        args = ', '.join(args)
        return f'{self.function}({args})'


@frozen
class ValueDraw(Expression):
    strategy: 'DataStrategy'

    @property
    def type(self) -> ExpressionType:
        return ExpressionType.DRAW

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
            return Reference(expr.token)
        if expr.is_this_msg:
            return Reference('msg')
    elif expr.is_operator:
        pass
    elif expr.is_function_call:
        pass
    elif expr.is_quantifier:
        pass
    elif expr.is_accessor:
        pass
    raise ValueError(f'unable to handle HplExpression: {expr!r}')


@typechecked
def convert_to_int(expr: Expression) -> Expression:
    if expr.is_literal and expr.is_int:
        return expr
    if expr.is_function_call:
        if expr.function == 'len':
            return expr
    return FunctionCall('int', (expr,))


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
        if expr.is_literal:
            return not expr.is_bool
        return False

    def __str__(self) -> str:
        return 'booleans()'


@frozen
class RandomInt(DataStrategy):
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
        return self if element in self.elements else RandomSample(self.elements + (element,))

    def remove(self, element: Expression) -> 'RandomSample':
        elements = [el for el in self.elements if el != element]
        return self if len(elements) == len(self.elements) else RandomSample(elements)

    def is_value_impossible(self, expr: Expression) -> bool:
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
        return f"assume('{self.expression}')"
