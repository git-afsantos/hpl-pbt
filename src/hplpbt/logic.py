# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

###############################################################################
# Imports
###############################################################################

from typing import Container, Iterable, List, Tuple

from hpl.ast import HplProperty
from typeguard import typechecked

###############################################################################
# Interface
###############################################################################


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
    # FIXME: this logic is not quite right
    for b in hpl_property.pattern.behaviour.simple_events():
        if b.name in input_channels:
            return [hpl_property], []
    return [], [hpl_property]
