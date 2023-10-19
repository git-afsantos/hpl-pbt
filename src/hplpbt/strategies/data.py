# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

###############################################################################
# Imports
###############################################################################

from typing import Any, Callable, Final, Iterable, List, Optional, Union

from enum import Enum, auto

from attrs import define, field
from hpl.ast import HplExpression
from hpl.types import TypeToken
from hplpbt.errors import ContradictionError

from hplpbt.strategies.ast import (
    Assumption,
    BinaryOperator,
    ConstantValue,
    DataStrategy,
    Expression,
    FunctionCall,
    Literal,
    RandomFloat,
    RandomInt,
    RandomSample,
    Reference,
    UnaryOperator,
    ValueDraw,
)

################################################################################
# Internal Structures: Basic (Data) Field Generator
################################################################################


@define
class NumberFieldGenerator:
    strategy: Union[RandomInt, RandomFloat, RandomSample, ConstantValue]
    assumptions: List[HplExpression] = field(factory=list)

    @classmethod
    def uint(cls) -> 'NumberFieldGenerator':
        return cls(RandomInt.uint())

    @classmethod
    def uint8(cls) -> 'NumberFieldGenerator':
        return cls(RandomInt.uint8())

    @classmethod
    def uint16(cls) -> 'NumberFieldGenerator':
        return cls(RandomInt.uint16())

    @classmethod
    def uint32(cls) -> 'NumberFieldGenerator':
        return cls(RandomInt.uint32())

    @classmethod
    def uint64(cls) -> 'NumberFieldGenerator':
        return cls(RandomInt.uint64())

    @classmethod
    def any_int(cls) -> 'NumberFieldGenerator':
        return cls(RandomInt())

    @classmethod
    def int8(cls) -> 'NumberFieldGenerator':
        return cls(RandomInt.int8())

    @classmethod
    def int16(cls) -> 'NumberFieldGenerator':
        return cls(RandomInt.int16())

    @classmethod
    def int32(cls) -> 'NumberFieldGenerator':
        return cls(RandomInt.int32())

    @classmethod
    def int64(cls) -> 'NumberFieldGenerator':
        return cls(RandomInt.int64())

    @classmethod
    def any_float(cls) -> 'NumberFieldGenerator':
        return cls(RandomFloat())

    @classmethod
    def float32(cls) -> 'NumberFieldGenerator':
        return cls(RandomFloat.float32())

    @classmethod
    def float64(cls) -> 'NumberFieldGenerator':
        return cls(RandomFloat.float64())

    def assume(self, condition: HplExpression):
        self.assumptions.append(condition)

    def eq(self, value: Expression):
        if self.strategy.is_value_impossible(value):
            raise ContradictionError(f'{self.strategy} == {value}')
        self.strategy = ConstantValue(value)

    def neq(self, value: Expression):
        if self.strategy.is_constant:
            assert isinstance(self.strategy, ConstantValue)
            if self.strategy.expression == value:
                raise ContradictionError(f'{self.strategy.expression} != {value}')
        elif self.strategy.is_sample:
            assert isinstance(self.strategy, RandomSample)
            self.strategy = self.strategy.remove(value)
            if len(self.strategy.elements) <= 0:
                raise ContradictionError(f'{self.strategy} != {value}')
        else:
            assert isinstance(self.strategy, (RandomInt, RandomFloat))
        # self.assumptions.append(_Assumption(self._init_ref, value, "!="))

    def lt(self, value: Expression):
        if self.strategy.is_constant:
            assert isinstance(self.strategy, ConstantValue)
            _check_lt(self.strategy.expression, value)
        elif self.strategy.is_sample:
            assert isinstance(self.strategy, RandomSample)
            test = lambda x: _can_be_lt(x, value)
            error = f'{self.strategy} < {value}'
            self.strategy = _filter_sample_strategy(self.strategy, test, error=error)
        elif self.strategy.is_int:
            assert isinstance(self.strategy, RandomInt)
            x = self.strategy.min_value
            if x is not None:
                _check_lt(x, value)
            y = self.strategy.max_value
            if y is None or _can_be_lt(value, y):
                self.strategy = RandomInt(min_value=x, max_value=value)
        elif self.strategy.is_float:
            assert isinstance(self.strategy, RandomFloat)
            x = self.strategy.min_value
            if x is not None:
                _check_lt(x, value)
            y = self.strategy.max_value
            if y is None or _can_be_lt(value, y):
                self.strategy = RandomFloat(min_value=x, max_value=value)

    def lte(self, value: Expression):
        if self.strategy.is_constant:
            assert isinstance(self.strategy, ConstantValue)
            _check_lt(self.strategy.expression, value)
        elif self.strategy.is_sample:
            assert isinstance(self.strategy, RandomSample)
            test = lambda x: _can_be_lt(x, value)
            error = f'{self.strategy} < {value}'
            self.strategy = _filter_sample_strategy(self.strategy, test, error=error)
        elif self.strategy.is_int:
            assert isinstance(self.strategy, RandomInt)
            x = self.strategy.min_value
            if x is not None:
                if x.eq(value):
                    self.strategy = ConstantValue(value)
                    return
                _check_lt(x, value)
            y = self.strategy.max_value
            if y is None or _can_be_lt(value, y):
                self.strategy = RandomInt(min_value=x, max_value=value)
        elif self.strategy.is_float:
            assert isinstance(self.strategy, RandomFloat)
            x = self.strategy.min_value
            if x is not None:
                if x.eq(value):
                    self.strategy = ConstantValue(value)
                    return
                _check_lt(x, value)
            y = self.strategy.max_value
            if y is None or _can_be_lt(value, y):
                self.strategy = RandomFloat(min_value=x, max_value=value)


################################################################################
# Helper Functions
################################################################################


def _filter_sample_strategy(
    sample: RandomSample,
    test: Callable[[Expression, Expression], bool],
    error: str = '',
) -> DataStrategy:
    new_elements = [element for element in sample.elements if test(element)]
    if not new_elements:
        raise ContradictionError(error)
    if len(new_elements) != len(sample.elements):
        if len(new_elements) == 1:
            return ConstantValue(new_elements[0])
        else:
            return RandomSample(new_elements)
    return sample


def _check_lt(x: Expression, y: Expression):
    if not _can_be_lt(x, y):
        raise ContradictionError(f'{x} < {y}')


def _can_be_lt(x: Expression, y: Expression) -> bool:
    # ensures that `x < y` is possible
    if not x.can_be_number or y.can_be_number:
        raise TypeError(f'{x} < {y}')

    if x.eq(y):
        return False

    if x.is_literal:
        assert isinstance(x, Literal)
        if y.is_function_call:
            assert isinstance(y, FunctionCall)
            if y.function == 'len':
                if not x.is_int:
                    raise TypeError(f'{x} < {y}')
            elif y.function == 'abs':
                if not x.is_int and not x.is_float:
                    raise TypeError(f'{x} < {y}')
    elif x.is_value_draw:
        assert isinstance(x, ValueDraw)
        if x.strategy.is_bool or x.strategy.is_string or x.strategy.is_array:
            raise TypeError(f'{x} < {y}')
        if x.strategy.is_constant:
            assert isinstance(x.strategy, ConstantValue)
            value: Expression = x.strategy.expression
            if value.

    elif y.is_literal:
        assert isinstance(y, Literal)
        if x.is_function_call:
            assert isinstance(x, FunctionCall)
            if x.function == 'len':
                if not y.is_int:
                    raise TypeError(f'{x} < {y}')
                return y.value >= 0
            elif x.function == 'abs':
                if not y.is_int and not y.is_float:
                    raise TypeError(f'{x} < {y}')
                return y.value >= 0
    return True


def _can_be_eq(x: Expression, y: Expression) -> bool:
    # assume some level of simplification already comes from previous steps

    if x.is_reference or y.is_reference:
        # unknown variables can always be equal unless we have context
        assert isinstance(x, Reference)
        assert isinstance(y, Reference)
        return True

    if x.eq(y):
        return True

    if x.is_unary_operator:
        assert isinstance(x, UnaryOperator)
        if x.operand.eq(y):
            return False

    elif x.is_binary_operator:
        assert isinstance(x, BinaryOperator)
        if x.operand1.eq(y) and x.operand2.is_literal:
            assert isinstance(x.operand2, Literal)
            # discard neutral operations
            if x.token == '+' or x.token == '-':
                return x.operand2.value == 0
            if x.token == '/':
                return x.operand2.value == 1
        elif x.operand2.eq(y) and x.operand1.is_literal:
            assert isinstance(x.operand1, Literal)
            # discard neutral operations
            if x.token == '+' or x.token == '-':
                return x.operand1.value == 0

    # FIXME everything below

    if y.is_unary_operator:
        assert isinstance(y, UnaryOperator)
        if x.eq(y.operand):
            return False
    elif y.is_binary_operator:
        assert isinstance(y, BinaryOperator)

    if x.is_literal:
        assert isinstance(x, Literal)
        if y.is_literal:
            assert isinstance(y, Literal)
            return x.value == y.value
        if y.is_binary_operator:
            assert isinstance(y, BinaryOperator)
        if y.is_value_draw:
            assert isinstance(y, ValueDraw)
        if y.is_function_call:
            assert isinstance(y, FunctionCall)
            if y.function == 'len':
                if not x.is_int:
                    raise TypeError(f'{x} < {y}')
            elif y.function == 'abs':
                if not x.is_int and not x.is_float:
                    raise TypeError(f'{x} < {y}')
    elif y.is_literal:
        assert isinstance(y, Literal)
        if x.is_function_call:
            assert isinstance(x, FunctionCall)
            if x.function == 'len':
                if not y.is_int:
                    raise TypeError(f'{x} < {y}')
                return y.value >= 0
            elif x.function == 'abs':
                if not y.is_int and not y.is_float:
                    raise TypeError(f'{x} < {y}')
                return y.value >= 0
    return True


################################################################################
# Internal Structures: Basic (Data) Field Generator
################################################################################


class DataGeneratorState(Enum):
    PENDING = auto()  # waiting for the parent to be initialized
    READY = auto()  # the parent is available, this can be initialized
    INITIALIZED = auto()  # the field has a value, but there is more to do
    FINALIZED = auto()  # the field and all subfields are fully processed


@define
class DataGenerator:
    """
    A DataGenerator is composed of multiple statements (initialization,
    assumptions, ...) and each statement has its own dependencies (references
    to other local/external fields) so that they can be sorted individually.
    This maximizes flexibility, and results in code that is closer to what a
    human would write, knowing in advance what each statement needs.
    Internally, the DataGenerator can be seen as a sort of state machine.
    When generating code, it changes its state as the requirements of each
    statement are satisfied and the statements are processed.
    """

    expression: Expression
    type_token: TypeToken
    parent: Optional['DataGenerator']
    strategy: DataStrategy
    assumptions: List[Assumption] = field(factory=list)
    is_ranged: bool = False
    reference_count: int = field(default=0, init=False, eq=False)
    _ready_ref: Any = None
    _init_ref: Any = None
    _loop_context: Optional[Any] = None

    def eq(self, value: Expression):
        if self.strategy.is_value_impossible(value):
            raise ContradictionError('{self.expression} == {value}')
        if self.strategy.is_constant:
            self.strategy = ConstantValue(value)
        else:
            self.strategy = ConstantValue(value)

    def neq(self, value: Expression):
        if self.strategy.is_sample:
            assert isinstance(self.strategy, RandomSample)
            self.strategy = self.strategy.remove(value)
            if len(self.strategy.elements) <= 0:
                raise ContradictionError('{self.expression} != {value}')
        self.assumptions.append(_Assumption(self._init_ref, value, "!="))
