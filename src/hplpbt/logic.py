# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

###############################################################################
# Imports
###############################################################################

from typing import Container, Iterable, List, Tuple

from attrs import field, frozen

from hpl.ast import (
    HplEvent,
    HplEventDisjunction,
    HplPattern,
    HplPredicate,
    HplProperty,
    HplSimpleEvent,
)
from typeguard import typechecked

###############################################################################
# Interface
###############################################################################


@frozen
class SystemAction:
    name: str
    channel: str
    guard: HplPredicate


@frozen
class StateMachine:
    inputs: Iterable[SystemAction] = field(factory=tuple, converter=tuple)
    outputs: Iterable[SystemAction] = field(factory=tuple, converter=tuple)


@typechecked
def split_assumptions(
    hpl_properties: Iterable[HplProperty],
    input_channels: Container[str],
) -> Tuple[List[HplProperty], List[HplProperty]]:
    """
    Given a list of properties and a collection of input channels,
    returns the list of assumptions and the list of behaviour specifications,
    calculated from the original list of properties.
    """
    assumptions = []
    behaviour = []
    for hpl_property in hpl_properties:
        assert isinstance(hpl_property, HplProperty)
        if hpl_property.pattern.is_absence:
            a, b = _split_safety(hpl_property, input_channels)
        elif hpl_property.pattern.is_existence:
            a, b = _split_liveness(hpl_property, input_channels)
        elif hpl_property.pattern.is_requirement:
            a, b = _split_safety(hpl_property, input_channels)
        elif hpl_property.pattern.is_response:
            a, b = _split_liveness(hpl_property, input_channels)
        elif hpl_property.pattern.is_prevention:
            a, b = _split_safety(hpl_property, input_channels)
        else:
            raise TypeError(f'unknown HPL property type: {hpl_property!r}')
        assumptions.extend(a)
        behaviour.extend(b)
    return assumptions, behaviour


###############################################################################
# Helper Functions
###############################################################################


def _split_safety(
    hpl_property: HplProperty,
    input_channels: Container[str],
) -> Tuple[List[HplProperty], List[HplProperty]]:
    # optimization: avoid creating new objects if the event is simple
    assert hpl_property.pattern.behaviour.is_simple_event
    if hpl_property.pattern.behaviour.name in input_channels:
        return [hpl_property], []
    else:
        return [], [hpl_property]


def _split_liveness(
    hpl_property: HplProperty,
    input_channels: Container[str],
) -> Tuple[List[HplProperty], List[HplProperty]]:
    inputs: List[HplSimpleEvent] = []
    outputs: List[HplSimpleEvent] = []
    for b in hpl_property.pattern.behaviour.simple_events():
        assert isinstance(b, HplSimpleEvent)
        if b.name in input_channels:
            inputs.append(b)
        else:
            outputs.append(b)
    if not outputs:
        # all behaviour events are inputs
        return [hpl_property], []
    if not inputs:
        # all behaviour events are outputs
        return [], [hpl_property]
    # mixed input and output events
    assert inputs and outputs
    # for the purposes of testing, avoid inputs and force the outputs
    # recreate the original property, but expecting only outputs
    new_pattern = hpl_property.pattern.but(behaviour=_recreate_events(outputs))
    hpl_property = hpl_property.but(pattern=new_pattern)
    # recreate an opposite of the original property, affecting only inputs
    if hpl_property.pattern.is_existence:
        new_pattern = HplPattern.absence(
            _recreate_events(inputs),
            min_time=hpl_property.pattern.min_time,
            max_time=hpl_property.pattern.max_time,
        )
    else:
        assert hpl_property.pattern.is_response
        new_pattern = HplPattern.prevention(
            hpl_property.pattern.trigger,
            _recreate_events(inputs),
            min_time=hpl_property.pattern.min_time,
            max_time=hpl_property.pattern.max_time,
        )
    assumption = hpl_property.but(pattern=new_pattern)
    return [assumption], [hpl_property]


def _recreate_events(events: List[HplSimpleEvent]) -> HplEvent:
    assert events
    result = events[-1]
    for i in range(len(events) - 1):
        result = HplEventDisjunction(events[i], result)
    return result
