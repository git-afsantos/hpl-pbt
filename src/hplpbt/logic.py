# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

###############################################################################
# Imports
###############################################################################

from typing import Container, Iterable, List, Tuple

from hpl.ast import HplProperty

###############################################################################
# Interface
###############################################################################


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
            a, b = _split_absence(hpl_property, input_channels)
        elif hpl_property.pattern.is_existence:
            a, b = _split_existence(hpl_property, input_channels)
        elif hpl_property.pattern.is_requirement:
            a, b = _split_requirement(hpl_property, input_channels)
        elif hpl_property.pattern.is_response:
            a, b = _split_response(hpl_property, input_channels)
        elif hpl_property.pattern.is_prevention:
            a, b = _split_prevention(hpl_property, input_channels)
        else:
            raise TypeError(f'unknown HPL property type: {hpl_property!r}')
        assumptions.extend(a)
        behaviour.extend(b)
    return assumptions, behaviour


###############################################################################
# Helper Functions
###############################################################################


def _split_absence(
    hpl_property: HplProperty,
    input_channels: Container[str],
) -> Tuple[List[HplProperty], List[HplProperty]]:
    # optimization: avoid creating new objects if the event is simple
    if hpl_property.pattern.behaviour.is_simple_event:
        if hpl_property.pattern.behaviour.name in input_channels:
            return [hpl_property], []
        else:
            return [], [hpl_property]
    assumptions = []
    behaviour = []
    for b in hpl_property.pattern.behaviour.simple_events():
        new_pattern = hpl_property.pattern.but(behaviour=b)
        new_property = hpl_property.but(pattern=new_pattern)
        if b.name in input_channels:
            assumptions.append(new_property)
        else:
            behaviour.append(new_property)
    return assumptions, behaviour


def _split_existence(
    hpl_property: HplProperty,
    input_channels: Container[str],
) -> Tuple[List[HplProperty], List[HplProperty]]:
    # optimization: avoid creating new objects if the event is simple
    return [], [hpl_property]


def _split_requirement(
    hpl_property: HplProperty,
    input_channels: Container[str],
) -> Tuple[List[HplProperty], List[HplProperty]]:
    # optimization: avoid creating new objects if the event is simple
    return [], [hpl_property]


def _split_response(
    hpl_property: HplProperty,
    input_channels: Container[str],
) -> Tuple[List[HplProperty], List[HplProperty]]:
    # optimization: avoid creating new objects if the event is simple
    return [], [hpl_property]


def _split_prevention(
    hpl_property: HplProperty,
    input_channels: Container[str],
) -> Tuple[List[HplProperty], List[HplProperty]]:
    # optimization: avoid creating new objects if the event is simple
    return [], [hpl_property]