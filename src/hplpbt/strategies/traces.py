# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

###############################################################################
# Imports
###############################################################################

from typing import Final, Iterable, List, Mapping, Optional, Set, Union

from attrs import define, evolve, field, frozen
from attrs.validators import deep_iterable, deep_mapping, instance_of
from hpl.ast import (
    HplEvent,
    HplExpression,
    HplPredicate,
    HplProperty,
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
class TraceEvent:
    channel: str
    strategy: MessageStrategy


@frozen
class TraceSegmentStrategy:
    delay: float = 0.0
    timeout: float = INF
    mandatory: Iterable[TraceEvent] = field(factory=tuple, converter=tuple)
    spam: Iterable[TraceEvent] = field(factory=tuple, converter=tuple)
    helpers: Iterable[MessageStrategy] = field(factory=tuple, converter=tuple)

    @property
    def has_timeout(self) -> bool:
        return self.timeout < INF

    @property
    def has_spam(self) -> bool:
        return len(self.spam) > 0

    @property
    def has_mandatory(self) -> bool:
        return len(self.mandatory) > 0

    def spam_strategies(self) -> List[MessageStrategy]:
        return [event.strategy for event in self.spam]

    def mandatory_strategies(self) -> List[MessageStrategy]:
        return [event.strategy for event in self.mandatory]



@frozen
class TraceStrategy:
    trace_name: str
    hpl_property: HplProperty
    segments: Iterable[TraceSegmentStrategy] = field(factory=tuple, converter=tuple)

    @property
    def name(self) -> str:
        return f'gen_{self.trace_name}'

    def all_msg_strategies(self) -> Set[MessageStrategy]:
        strategies: Set[MessageStrategy] = set()
        for segment in self.segments:
            strategies.update(segment.helpers)
            strategies.update(segment.spam_strategies())
            strategies.update(segment.mandatory_strategies())
        return strategies

    def get_return_type(self) -> str:
        return_types: Set[str] = set()
        for segment in self.segments:
            return_types.update(strat.return_type for strat in segment.spam_strategies())
            return_types.update(strat.return_type for strat in segment.mandatory_strategies())
        if not return_types:
            return 'Any'
        if len(return_types) == 1:
            for return_type in return_types:
                return return_type
        return f'Union[{", ".join(return_types)}]'


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


@define
class _TraceNameGenerator:
    name: str = 'trace'
    index: int = 0
    triggers: int = 0

    @property
    def trigger_name(self) -> str:
        return f'{self.name}_trigger{self.triggers}'

    def generate(self) -> str:
        self.index += 1
        self.triggers = 0
        self.name = f'trace{self.index}'
        return self.name

    def generate_trigger(self) -> str:
        self.triggers += 1
        return self.trigger_name


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
    _name_generator: _TraceNameGenerator = field(factory=_TraceNameGenerator)

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
            strategies.add(self.build_from_property(hpl_property))
        return strategies

    def build_from_property(self, hpl_property: HplProperty) -> TraceStrategy:
        trace_name = self._name_generator.generate()
        segments: List[TraceSegmentStrategy] = []

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

        return TraceStrategy(trace_name, hpl_property, segments=segments)

    def publish_segment(
        self,
        event: HplEvent,
        delay: float = 0.0,
        timeout: float = INF,
    ) -> TraceSegmentStrategy:
        segment: TraceSegmentStrategy = self.spam_segment_avoiding(event, delay=delay, timeout=timeout)
        # FIXME apply general trace assumptions
        # 1. alter type definitions to generate triggers satisfying conditions
        triggers: Mapping[str, HplPredicate] = {}
        for ev in event.simple_events():
            assert isinstance(ev, HplSimpleEvent)
            # discard unknown input channels
            type_name: Optional[str] = self.input_channels.get(ev.name)
            if type_name is None:
                continue
            # store predicates per message type
            # phi = phi.disjoin(ev.predicate.negate())
            triggers[type_name] = ev.predicate
        modifier = self._name_generator.generate_trigger()
        new_type_defs = _apply_extra_type_conditions(self.type_defs, triggers, modifier=modifier)
        builder = MessageStrategyBuilder(self.input_channels, new_type_defs)
        mandatory: Set[TraceEvent] = set()
        helpers: Set[MessageStrategy] = set(segment.helpers)
        for ev in event.simple_events():
            assert isinstance(ev, HplSimpleEvent)
            strategy, dependencies = builder.build_pack_from_simple_event(ev)
            mandatory.add(TraceEvent(ev.name, strategy))
            helpers.update(dependencies)
        return evolve(segment, mandatory=mandatory, helpers=helpers)

    def spam_segment_avoiding(
        self,
        event: HplEvent,
        delay: float = 0.0,
        timeout: float = INF,
    ) -> TraceSegmentStrategy:
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
    ) -> TraceSegmentStrategy:
        # build spam messages that avoid trigger conditions
        new_type_defs = _apply_extra_type_conditions(
            self.type_defs,
            conditions,
            modifier=self._name_generator.name,
        )
        builder = MessageStrategyBuilder(self.input_channels, new_type_defs)
        spam: Set[TraceEvent] = set()
        helpers: Set[MessageStrategy] = set()
        for channel_name, type_name in self.input_channels.items():
            strategy, dependencies = builder.build_pack_for_type_name(type_name)
            spam.add(TraceEvent(channel_name, strategy))
            helpers.update(dependencies)
        return TraceSegmentStrategy(delay=delay, timeout=timeout, spam=spam, helpers=helpers)


################################################################################
# Helper Functions
################################################################################


def _apply_extra_type_conditions(
    type_defs: Mapping[str, MessageType],
    conditions: Optional[Mapping[str, HplPredicate]],
    modifier: str = '',
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
            new_name = f'{name}_{modifier}v{i}'
            i += 1
            phi = type_def.precondition.join(phi)
            new_type_defs[name] = evolve(type_def, name=new_name, precondition=phi)
    return new_type_defs
