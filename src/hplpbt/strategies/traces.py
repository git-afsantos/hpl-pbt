# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

###############################################################################
# Imports
###############################################################################

from typing import Final, Iterable, List, Mapping, Optional, Set, Union

from attrs import evolve, field, frozen
from attrs.validators import deep_iterable, deep_mapping, instance_of
from hpl.ast import (
    HplEvent,
    HplExpression,
    HplPattern,
    HplPredicate,
    HplProperty,
    HplScope,
    HplSimpleEvent,
    HplSpecification,
)
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
    delay: float = 0.0
    timeout: float = INF
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
        validator=deep_iterable(
            instance_of(HplProperty),
            iterable_validator=instance_of(Iterable),
        ),
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
            segments.append(self.publish_segment(event))

        # main segments: pattern-based events
        if hpl_property.pattern.is_absence:
            segments.append(self.spam_segment())
        elif hpl_property.pattern.is_existence:
            segments.append(self.spam_segment())
        elif hpl_property.pattern.is_requirement:
            event = hpl_property.pattern.trigger
            assert event is not None
            timeout = hpl_property.pattern.max_time
            if hpl_property.pattern.has_max_time:
                # 1. check for behaviour without trigger for the duration
                segments.append(self.spam_segment_avoiding(event, timeout=timeout))
                # 2. emit a trigger after the duration
                segments.append(self.publish_segment(event))
                # 3. avoid further triggers for the safe duration
                # plus an unsafe interval to check for late behaviour
                timeout = 3 * timeout if timeout < 1.0 else 2 * timeout
                segments.append(self.spam_segment_avoiding(event, timeout=timeout))
            else:
                # untimed; check for behaviour without trigger
                segments.append(self.spam_segment_avoiding(event, timeout=timeout))
        elif hpl_property.pattern.is_response:
            event = hpl_property.pattern.trigger
            assert event is not None
            timeout = hpl_property.pattern.max_time
            segments.append(self.publish_segment(event))
            segments.append(self.spam_segment(timeout=timeout))
        elif hpl_property.pattern.is_prevention:
            event = hpl_property.pattern.trigger
            assert event is not None
            timeout = hpl_property.pattern.max_time
            segments.append(self.publish_segment(event))
            segments.append(self.spam_segment(timeout=timeout))
        else:
            raise TypeError(f'unknown HPL property type: {hpl_property!r}')

        # last segment: terminator
        event = hpl_property.scope.terminator
        if event is not None:
            segments.append(self.publish_segment(event))

        return TraceStrategy(segments=segments)

    def publish_segment(
        self,
        event: HplEvent,
        delay: float = 0.0,
        timeout: float = INF,
    ) -> TraceSegment:
        segment: TraceSegment = self.spam_segment_avoiding(event, delay=delay, timeout=timeout)
        # FIXME apply general trace assumptions
        builder = MessageStrategyBuilder(self.input_channels, self.type_defs)
        mandatory: Set[MessageStrategy] = set()
        helpers: Set[MessageStrategy] = set(segment.helpers)
        for ev in event.simple_events():
            assert isinstance(ev, HplSimpleEvent)
            strategy, dependencies = builder.build_pack_from_simple_event(ev)
            mandatory.add(strategy)
            helpers.update(dependencies)
        return evolve(segment, mandatory=mandatory, helpers=helpers)

    def spam_segment_avoiding(
        self,
        event: HplEvent,
        delay: float = 0.0,
        timeout: float = INF,
    ) -> TraceSegment:
        # build assumptions to avoid all possible triggers
        anti_triggers: Mapping[str, HplPredicate] = {}
        for ev in event.simple_events():
            assert isinstance(ev, HplSimpleEvent)
            # discard unknown input channels
            type_name: Optional[str] = self.input_channels.get(ev.name)
            if type_name is None:
                continue
            # store predicates per message type
            phi: Optional[HplPredicate] = anti_triggers.get(type_name)
            if phi is None:
                phi = ev.predicate.negate()
            else:
                phi = phi.join(ev.predicate.negate())
            anti_triggers[type_name] = phi
        # FIXME must do something with general trace assumptions
        # assumptions = list(self.assumptions)
        # for name, phi in anti_triggers.items():
        #     scope = HplScope.globally()
        #     behaviour = HplSimpleEvent.publish(name, phi)
        #     pattern = HplPattern.absence(behaviour)
        #     assumptions.append(HplProperty(scope, pattern))
        # ------------------------------------------------------
        return self.spam_segment(conditions=anti_triggers, delay=delay, timeout=timeout)

    def spam_segment(
        self,
        conditions: Optional[Mapping[str, HplExpression]] = None,
        delay: float = 0.0,
        timeout: float = INF,
    ) -> TraceSegment:
        # build spam messages that avoid trigger conditions
        new_type_defs = _apply_extra_type_conditions(self.type_defs, conditions)
        builder = MessageStrategyBuilder(self.input_channels, new_type_defs)
        spam: Set[MessageStrategy] = set()
        helpers: Set[MessageStrategy] = set()
        for type_name in self.input_channels.values():
            strategy, dependencies = builder.build_pack_for_type_name(type_name)
            spam.add(strategy)
            helpers.update(dependencies)
        return TraceSegment(delay=delay, timeout=timeout, spam=spam, helpers=helpers)


################################################################################
# Helper Functions
################################################################################


def _apply_extra_type_conditions(
    type_defs: Mapping[str, MessageType],
    conditions: Optional[Mapping[str, HplPredicate]],
) -> Mapping[str, MessageType]:
    if not conditions:
        return type_defs
    i = 1
    new_type_defs = {}
    for name, type_def in type_defs.items():
        phi = conditions.get(name)
        if phi is None:
            new_type_defs[name] = type_def
        else:
            new_name = f'{name}_v{i}'
            i += 1
            phi = type_def.precondition.join(phi)
            new_type_defs[name] = evolve(type_def, name=new_name, precondition=phi)
    return new_type_defs
