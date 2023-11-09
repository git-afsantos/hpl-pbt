# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

###############################################################################
# Imports
###############################################################################

from typing import Final, Iterable, List, Mapping, Optional, Set, Union

from attrs import field, frozen
from attrs.validators import deep_iterable, deep_mapping, instance_of
from hpl.ast import HplEvent, HplProperty, HplSimpleEvent, HplSpecification
from hplpbt.strategies.messages import MessageStrategy, MessageStrategyBuilder
from typeguard import typechecked

from hplpbt.types import MessageType

################################################################################
# Constants
################################################################################

INF: Final[float] = float('inf')

################################################################################
# Internal Structures: Traces and Segments
################################################################################


@frozen
class TraceSegment:
    delay: float
    timeout: float
    mandatory: Iterable[MessageStrategy] = field(factory=tuple, converter=tuple)
    spam: Iterable[MessageStrategy] = field(factory=tuple, converter=tuple)
    helpers: Iterable[MessageStrategy] = field(factory=tuple, converter=tuple)


@frozen
class TraceStrategy:
    segments: Iterable[TraceSegment] = field(factory=tuple, converter=tuple)


################################################################################
# Interface
################################################################################


@typechecked
def strategies_from_spec(
    spec: Union[HplSpecification, Iterable[HplProperty]],
    input_channels: Mapping[str, str],
    type_defs: Mapping[str, MessageType],
    assumptions: Optional[Iterable[HplProperty]] = None,
) -> Set[TraceStrategy]:
    assumptions = assumptions if assumptions is not None else []
    builder = TraceStrategyBuilder(input_channels, type_defs, assumptions=assumptions)
    return builder.build_from_spec(spec)


@typechecked
def strategy_from_property(
    hpl_property: HplProperty,
    input_channels: Mapping[str, str],
    type_defs: Mapping[str, MessageType],
    assumptions: Optional[Iterable[HplProperty]] = None,
) -> TraceStrategy:
    assumptions = assumptions if assumptions is not None else []
    builder = TraceStrategyBuilder(input_channels, type_defs, assumptions=assumptions)
    return builder.build_from_property(hpl_property)


################################################################################
# Trace Builder
################################################################################


@frozen
class TraceStrategyBuilder:
    input_channels: Mapping[str, str] = field(
        validator=deep_mapping(
            instance_of(str),
            instance_of(str),
            mapping_validator=instance_of(Mapping),
        )
    )
    type_defs: Mapping[str, MessageType] = field(
        validator=deep_mapping(
            instance_of(str),
            instance_of(MessageType),
            mapping_validator=instance_of(Mapping),
        )
    )
    assumptions: Iterable[HplProperty] = field(
        factory=list,
        validator=deep_iterable(instance_of(HplProperty), iterable_validator=instance_of(Iterable))
    )

    @input_channels.validator
    def _check_all_channels_defined(self, _attribute, channels: Mapping[str, str]) -> None:
        for type_name in channels.values():
            if type_name not in self.type_defs:
                raise ValueError(f'message type {type_name!r} is not defined')

    def build_from_spec(
        self,
        spec: Union[HplSpecification, Iterable[HplProperty]],
    ) -> Set[TraceStrategy]:
        if isinstance(spec, HplSpecification):
            spec = spec.properties
        strategies = set()
        for hpl_property in spec:
            strategies.update(self.build_from_property(hpl_property))
        return strategies

    def build_from_property(self, hpl_property: HplProperty) -> TraceStrategy:
        segments: List[TraceSegment] = []

        # first segment: activator
        event = hpl_property.scope.activator
        if event is not None:
            seg = self.publish_segment(event)
            segments.append(seg)

        # main segments: pattern-based events
        if hpl_property.pattern.is_absence:
            pass  # spam segment
        elif hpl_property.pattern.is_existence:
            pass  # spam segment
        elif hpl_property.pattern.is_requirement:
            event = hpl_property.pattern.trigger
            assert event is not None
            # spam avoid trigger segment
        elif hpl_property.pattern.is_response:
            event = hpl_property.pattern.trigger
            assert event is not None
            # trigger segment
            # spam segment
        elif hpl_property.pattern.is_prevention:
            event = hpl_property.pattern.trigger
            assert event is not None
            # trigger segment
            # spam segment
        else:
            raise TypeError(f'unknown HPL property type: {hpl_property!r}')

        # last segment: terminator
        event = hpl_property.scope.terminator
        if event is not None:
            seg = self.publish_segment(event)
            segments.append(seg)

        return TraceStrategy(segments=segments)

    def publish_segment(self, event: HplEvent) -> TraceSegment:
        builder = MessageStrategyBuilder(
            self.input_channels,
            self.type_defs,
            assumptions=self.assumptions
        )
        mandatory: Set[MessageStrategy] = set()
        helpers: Set[MessageStrategy] = set()
        for ev in event.simple_events():
            assert isinstance(ev, HplSimpleEvent)
            strategy, dependencies = builder.build_pack_from_simple_event(ev)
            mandatory.add(strategy)
            helpers.update(dependencies)
        return TraceSegment(0, INF, mandatory=mandatory, helpers=helpers)
