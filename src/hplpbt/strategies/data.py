# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

###############################################################################
# Imports
###############################################################################

from typing import Any, List, Optional

from enum import Enum, auto

from attrs import define, field
from hpl.types import TypeToken

from hplpbt.strategies.ast import Assumption, ConstantValue, DataStrategy, Expression, RandomSample

################################################################################
# Internal Structures: Basic (Data) Field Generator
################################################################################


class ContradictionError(Exception):
    pass


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
