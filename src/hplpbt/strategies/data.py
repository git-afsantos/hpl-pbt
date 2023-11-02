# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

###############################################################################
# Imports
###############################################################################

from typing import Callable, List

from attrs import define, field
from hpl.ast import (
    HplBinaryOperator,
    HplExpression,
    HplFunctionCall,
    HplQuantifier,
    HplUnaryOperator,
    HplVarReference,
)
from hplpbt.errors import ContradictionError

from hplpbt.strategies.ast import (
    BinaryOperator,
    ConstantValue,
    DataStrategy,
    Expression,
    FunctionCall,
    Literal,
    RandomBool,
    RandomFloat,
    RandomInt,
    RandomSample,
    RandomString,
    Reference,
    UnaryOperator,
    ValueDraw,
)

################################################################################
# Internal Structures: Basic (Data) Field Generator
################################################################################


@define
class BasicDataFieldGenerator:
    strategy: DataStrategy
    assumptions: List[HplExpression] = field(factory=list)

    def assume(self, phi: HplExpression):
        if not phi.can_be_bool:
            raise TypeError(f'not a boolean condition: {phi}')
        # self.assumptions.append(phi)

        if phi.is_value:
            assert not phi.is_set
            assert not phi.is_range
            assert not phi.is_this_msg
            # if phi.is_literal: pass
            # if phi.is_reference: pass
            if phi.is_variable:
                assert isinstance(phi, HplVarReference)
                self.eq(Literal.true())

        elif phi.is_accessor:
            # FIXME support this
            # if phi.is_field: assert isinstance(phi, HplFieldAccess)
            #if phi.is_indexed:
            #    assert isinstance(phi, HplArrayAccess)
            # ref = phi.base_object()
            pass

        elif phi.is_operator:
            if isinstance(phi, HplUnaryOperator):
                assert phi.operator.is_not
                self.reject(phi.operand)

            elif isinstance(phi, HplBinaryOperator):
                assert not phi.operator.is_arithmetic
                if phi.operator.is_inclusion:
                    pass
                elif phi.operator.is_equality:
                    pass
                elif phi.operator.is_inequality:
                    pass
                elif phi.operator.is_less_than:
                    pass
                elif phi.operator.is_less_than_eq:
                    pass
                elif phi.operator.is_greater_than:
                    pass
                elif phi.operator.is_greater_than_eq:
                    pass
                elif phi.operator.is_and:
                    pass
                elif phi.operator.is_or:
                    pass
                elif phi.operator.is_implies:
                    pass
                elif phi.operator.is_iff:
                    pass

        elif phi.is_function_call:
            # FIXME support this
            assert isinstance(phi, HplFunctionCall)

        elif phi.is_quantifier:
            # FIXME support this
            assert isinstance(phi, HplQuantifier)

    def reject(self, phi: HplExpression):
        if not phi.can_be_bool:
            raise TypeError(f'not a boolean condition: {phi}')

        if phi.is_value:
            assert not phi.is_set
            assert not phi.is_range
            assert not phi.is_this_msg
            # if phi.is_literal: pass
            # if phi.is_reference: pass
            if phi.is_variable:
                assert isinstance(phi, HplVarReference)
                self.eq(Literal.false())

        elif phi.is_accessor:
            # FIXME support this
            # if phi.is_field: assert isinstance(phi, HplFieldAccess)
            #if phi.is_indexed:
            #    assert isinstance(phi, HplArrayAccess)
            # ref = phi.base_object()
            pass

        elif phi.is_operator:
            if isinstance(phi, HplUnaryOperator):
                assert phi.operator.is_not
                self.assume(phi.operand)

            elif isinstance(phi, HplBinaryOperator):
                assert not phi.operator.is_arithmetic
                if phi.operator.is_inclusion:
                    pass
                elif phi.operator.is_equality:
                    pass
                elif phi.operator.is_inequality:
                    pass
                elif phi.operator.is_less_than:
                    pass
                elif phi.operator.is_less_than_eq:
                    pass
                elif phi.operator.is_greater_than:
                    pass
                elif phi.operator.is_greater_than_eq:
                    pass
                elif phi.operator.is_and:
                    pass
                elif phi.operator.is_or:
                    pass
                elif phi.operator.is_implies:
                    pass
                elif phi.operator.is_iff:
                    pass

        elif phi.is_function_call:
            # FIXME support this
            assert isinstance(phi, HplFunctionCall)

        elif phi.is_quantifier:
            # FIXME support this
            assert isinstance(phi, HplQuantifier)

    def eq(self, value: Expression):
        if self.strategy.is_value_impossible(value):
            raise ContradictionError(f'{self.strategy} == {value}')
        self.strategy = ConstantValue(value)

    def one_of(self, value: Expression):
        if self.strategy.is_value_impossible(value):
            raise ContradictionError(f'{self.strategy} in {value}')
        if self.strategy.is_constant:
            assert isinstance(self.strategy, ConstantValue)
        elif self.strategy.is_sample:
            assert isinstance(self.strategy, RandomSample)

    def neq(self, value: Expression):
        if self.strategy.is_constant:
            assert isinstance(self.strategy, ConstantValue)
            if self.strategy.expression.eq(value):
                raise ContradictionError(f'{self.strategy.expression} != {value}')
        elif self.strategy.is_sample:
            assert isinstance(self.strategy, RandomSample)
            self.strategy = self.strategy.remove(value)
            if len(self.strategy.elements) <= 0:
                raise ContradictionError(f'{self.strategy} != {value}')
        else:
            self._type_specific_neq(value)

    def _type_specific_neq(self, value: Expression):
        # self.assumptions.append(_Assumption(self._init_ref, value, "!="))
        pass

    def lt(self, value: Expression):
        raise TypeError(f'{self.strategy} < {value}')

    def lte(self, value: Expression):
        raise TypeError(f'{self.strategy} <= {value}')

    def gt(self, value: Expression):
        raise TypeError(f'{self.strategy} > {value}')

    def gte(self, value: Expression):
        raise TypeError(f'{self.strategy} >= {value}')


@define
class BooleanFieldGenerator(BasicDataFieldGenerator):
    @classmethod
    def any_bool(cls) -> 'BooleanFieldGenerator':
        return cls(RandomBool())

    def _type_specific_neq(self, value: Expression):
        assert isinstance(self.strategy, RandomBool)


@define
class NumberFieldGenerator(BasicDataFieldGenerator):
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

    def _type_specific_neq(self, value: Expression):
        assert isinstance(self.strategy, (RandomInt, RandomFloat))

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
        else:
            raise TypeError(f'{self.strategy} < {value}')

    def lte(self, value: Expression):
        if self.strategy.is_constant:
            assert isinstance(self.strategy, ConstantValue)
            if not _can_be_eq(self.strategy.expression, value):
                _check_lt(self.strategy.expression, value)
        elif self.strategy.is_sample:
            assert isinstance(self.strategy, RandomSample)
            test = lambda x: _can_be_eq(x, value) or _can_be_lt(x, value)
            error = f'{self.strategy} <= {value}'
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
            if y is None or _can_be_eq(value, y) or _can_be_lt(value, y):
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
            if y is None or _can_be_eq(value, y) or _can_be_lt(value, y):
                self.strategy = RandomFloat(min_value=x, max_value=value)
        else:
            raise TypeError(f'{self.strategy} <= {value}')

    def gt(self, value: Expression):
        if self.strategy.is_constant:
            assert isinstance(self.strategy, ConstantValue)
            _check_lt(value, self.strategy.expression)
        elif self.strategy.is_sample:
            assert isinstance(self.strategy, RandomSample)
            test = lambda x: _can_be_lt(value, x)
            error = f'{self.strategy} > {value}'
            self.strategy = _filter_sample_strategy(self.strategy, test, error=error)
        elif self.strategy.is_int:
            assert isinstance(self.strategy, RandomInt)
            y = self.strategy.max_value
            if y is not None:
                _check_lt(value, y)
            x = self.strategy.min_value
            if x is None or _can_be_lt(x, value):
                self.strategy = RandomInt(min_value=value, max_value=y)
        elif self.strategy.is_float:
            assert isinstance(self.strategy, RandomFloat)
            y = self.strategy.max_value
            if y is not None:
                _check_lt(value, y)
            x = self.strategy.min_value
            if x is None or _can_be_lt(x, value):
                self.strategy = RandomFloat(min_value=value, max_value=y)
        else:
            raise TypeError(f'{self.strategy} > {value}')

    def gte(self, value: Expression):
        if self.strategy.is_constant:
            assert isinstance(self.strategy, ConstantValue)
            if not _can_be_eq(self.strategy.expression, value):
                _check_lt(value, self.strategy.expression)
        elif self.strategy.is_sample:
            assert isinstance(self.strategy, RandomSample)
            test = lambda x: _can_be_eq(x, value) or _can_be_lt(value, x)
            error = f'{self.strategy} >= {value}'
            self.strategy = _filter_sample_strategy(self.strategy, test, error=error)
        elif self.strategy.is_int:
            assert isinstance(self.strategy, RandomInt)
            y = self.strategy.max_value
            if y is not None:
                if y.eq(value):
                    self.strategy = ConstantValue(value)
                    return
                _check_lt(value, y)
            x = self.strategy.min_value
            if x is None or _can_be_eq(x, value) or _can_be_lt(x, value):
                self.strategy = RandomInt(min_value=value, max_value=y)
        elif self.strategy.is_float:
            assert isinstance(self.strategy, RandomFloat)
            y = self.strategy.max_value
            if y is not None:
                if y.eq(value):
                    self.strategy = ConstantValue(value)
                    return
                _check_lt(value, y)
            x = self.strategy.min_value
            if x is None or _can_be_eq(x, value) or _can_be_lt(x, value):
                self.strategy = RandomFloat(min_value=value, max_value=y)
        else:
            raise TypeError(f'{self.strategy} >= {value}')


@define
class StringFieldGenerator(BasicDataFieldGenerator):
    @classmethod
    def any_string(cls) -> 'StringFieldGenerator':
        return cls(RandomString())

    def _type_specific_neq(self, value: Expression):
        assert isinstance(self.strategy, RandomString)


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
            return _can_be_lt(x.strategy.expression, y)
        if x.strategy.is_sample:
            assert isinstance(x.strategy, RandomSample)
            return any(_can_be_lt(el, y) for el in x.strategy.elements)

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
            if x.token == 'not':
                return False
            if x.operand.is_literal:
                assert isinstance(x.operand, Literal)
                if x.token == '+' or x.token == '-' or x.token == '~':
                    return x.operand.value == 0

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

    elif y.is_unary_operator:
        assert isinstance(y, UnaryOperator)
        if y.operand.eq(x):
            if y.token == 'not':
                return False
            if y.operand.is_literal:
                assert isinstance(y.operand, Literal)
                if y.token == '+' or y.token == '-' or y.token == '~':
                    return y.operand.value == 0

    elif y.is_binary_operator:
        assert isinstance(y, BinaryOperator)
        if y.operand1.eq(x) and y.operand2.is_literal:
            assert isinstance(y.operand2, Literal)
            # discard neutral operations
            if y.token == '+' or y.token == '-':
                return y.operand2.value == 0
            if y.token == '/':
                return y.operand2.value == 1
        elif y.operand2.eq(x) and y.operand1.is_literal:
            assert isinstance(y.operand1, Literal)
            # discard neutral operations
            if y.token == '+' or y.token == '-':
                return y.operand1.value == 0

    return True
