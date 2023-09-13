# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

###############################################################################
# Imports
###############################################################################

from typing import Any, Iterable, List, Mapping, Optional, Union

from enum import Enum, auto

from attrs import define, field, frozen
from typeguard import typechecked

from hpl.ast import HplExpression, HplProperty, HplSpecification
from hpl.types import TypeToken

################################################################################
# Internal Structures: Basic (Data) Field Generator
################################################################################


class FieldGeneratorState(Enum):
    PENDING = auto()  # waiting for the parent to be initialized
    READY = auto()  # the parent is available, this can be initialized
    INITIALIZED = auto()  # the field has a value, but there is more to do
    FINALIZED = auto()  # the field and all subfields are fully processed


@define
class FieldGenerator:
    """
    A FieldGenerator is composed of multiple statements (initialization,
    assumptions, ...) and each statement has its own dependencies (references
    to other local/external fields) so that they can be sorted individually.
    This maximizes flexibility, and results in code that is closer to what a
    human would write, knowing in advance what each statement needs.
    Internally, the FieldGenerator can be seen as a sort of state machine.
    When generating code, it changes its state as the requirements of each
    statement are satisfied and the statements are processed.
    """

    expression: HplExpression
    type_token: TypeToken
    parent: Optional['FieldGenerator']
    strategy: Any
    assumptions: List[Any] = field(factory=list)
    is_ranged: bool = False
    reference_count: int = field(default=0, init=False, eq=False)
    _ready_ref: Any = None
    _init_ref: Any = None
    _loop_context: Optional[Any] = None


@frozen
class MessageStrategy:
    name: str


###############################################################################
# Interface
###############################################################################


@typechecked
def message_strategies_for_spec(
    spec: Union[HplSpecification, Iterable[HplProperty]],
    msg_types: Mapping[str, Mapping[str, Any]],
) -> List[MessageStrategy]:
    if isinstance(spec, HplSpecification):
        spec = spec.properties
    return [
        strat for property in spec
        for strat in message_strategies_for_property(property, msg_types)
    ]


@typechecked
def message_strategies_for_property(
    property: HplProperty,
    msg_types: Mapping[str, Mapping[str, Any]],
) -> List[MessageStrategy]:
    return [MessageStrategy('Hello')]


###############################################################################
# Helper Functions
###############################################################################
