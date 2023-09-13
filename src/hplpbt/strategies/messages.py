# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

###############################################################################
# Imports
###############################################################################

from typing import Any, Iterable, List, Mapping, Optional, Set, Union

from enum import Enum, auto

from attrs import define, field, frozen
from hpl.ast import HplEvent, HplExpression, HplProperty, HplSimpleEvent, HplSpecification
from hpl.types import TypeToken
from typeguard import typechecked

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
    input_channels: Mapping[str, str],
    type_defs: Mapping[str, Mapping[str, Any]],
) -> Set[MessageStrategy]:
    if isinstance(spec, HplSpecification):
        spec = spec.properties
    strategies = set()
    for hpl_property in spec:
        strategies.update(message_strategies_for_property(hpl_property, input_channels, type_defs))
    return strategies


@typechecked
def message_strategies_for_property(
    hpl_property: HplProperty,
    input_channels: Mapping[str, str],
    type_defs: Mapping[str, Mapping[str, Any]],
) -> Set[MessageStrategy]:
    strategies = set()

    event = hpl_property.scope.activator
    if event is not None:
        strategies.update(message_strategies_for_event(event, input_channels, type_defs))
    event = hpl_property.scope.terminator
    if event is not None:
        strategies.update(message_strategies_for_event(event, input_channels, type_defs))

    if hpl_property.pattern.is_absence:
        pass

    elif hpl_property.pattern.is_existence:
        pass

    elif hpl_property.pattern.is_requirement:
        event = hpl_property.pattern.trigger
        assert event is not None
        strategies.update(message_strategies_for_event(event, input_channels, type_defs))

    elif hpl_property.pattern.is_response:
        event = hpl_property.pattern.trigger
        assert event is not None
        strategies.update(message_strategies_for_event(event, input_channels, type_defs))

    elif hpl_property.pattern.is_prevention:
        event = hpl_property.pattern.trigger
        assert event is not None
        strategies.update(message_strategies_for_event(event, input_channels, type_defs))

    else:
        raise TypeError(f'unknown HPL property type: {hpl_property!r}')

    return strategies


@typechecked
def message_strategies_for_event(
    event: HplEvent,
    input_channels: Mapping[str, str],
    type_defs: Mapping[str, Mapping[str, Any]],
) -> Set[MessageStrategy]:
    return set(
        strat
        for msg in event.simple_events()
        for strat in _build_strategies(msg, input_channels, type_defs)
    )


###############################################################################
# Helper Functions
###############################################################################


def _build_strategies(
    event: HplSimpleEvent,
    input_channels: Mapping[str, str],
    type_defs: Mapping[str, Mapping[str, Any]],
) -> Set[MessageStrategy]:
    strategies = set()
    if event.name not in input_channels:
        return strategies
    strategies.add(MessageStrategy(event.name))
    return strategies
