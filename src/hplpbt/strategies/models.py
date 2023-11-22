# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

###############################################################################
# Imports
###############################################################################

from typing import Final, Iterable, List, Mapping, Optional, Set, Union

from collections import defaultdict

from attrs import define, evolve, field, frozen
from attrs.validators import deep_iterable, deep_mapping, instance_of
from hpl.ast import (
    HplEvent,
    HplExpression,
    HplPredicate,
    HplProperty,
    HplSimpleEvent,
    HplSpecification,
    HplVacuousTruth,
)
from hplpbt.strategies.messages import MessageStrategy, MessageStrategyBuilder
from typeguard import typechecked

from hplpbt.types import MessageType

################################################################################
# Constants
################################################################################

INF: Final[float] = float('inf')

################################################################################
# Internal Structures: Rules for State Machines
################################################################################


"""
from hypothesis.stateful import RuleBasedStateMachine, precondition, rule, initialize, invariant

class DynamicStateMachine(RuleBasedStateMachine):
    def __init__(self, sut):
        super().__init__()
        self.sut = sut
        self.inbox1: Iterable[int] = deque()  # one per output channel
        self.monitor1: int = object()
        self.monitor2: int = object()

    @property
    def has_pending_output(self) -> bool:
        if self.inbox1:  # one of these blocks
            return True  # per output channel
        return False

    @initialize()
    def setup_sut(self):
        self.sut.setup()
        t = time.time()
        self.monitor1.on_launch(t)
        self.monitor2.on_launch(t)

    def teardown(self):
        self.sut.teardown()
        t = time.time()
        self.monitor1.on_shutdown(t)
        self.monitor2.on_shutdown(t)

    @precondition(lambda self: not self.has_pending_output)
    @rule(ms=sampled_from((5, 10, 20, 50, 100, 200, 500, 1000)))
    def sleep(self, ms):
        time.sleep(ms / 1000.0)

    @precondition(lambda self: not self.has_pending_output)
    @rule(draw=data())
    def send_a(self, data):
        if self.monitor1.is_active and self.monitor2.is_active:
            msg = data.draw(a_strategy1())
        elif self.monitor1.is_active:
            msg = data.draw(a_strategy2())
        else:
            msg = data.draw(a_default_strategy())
        self.sut.send('a', msg)

    @precondition(lambda self: self.has_pending_output)
    @rule()
    def receive_b(self):
        msg = self.inbox1.popleft()
        t = time.time()
        self.monitor1.on_msg_b(msg, t)

    @invariant()
    def check_monitors(self):
        assert self.monitor1.verdict is not False
        assert self.monitor2.verdict is not False

    @invariant()
    def check_inboxes(self):
        self.inbox1.extend(self.sut.receive('b'))
"""


@frozen
class TraceEvent:
    channel: str
    strategy: MessageStrategy


@frozen
class Rule:
    channel: str
    monitors: Iterable[str] = field(factory=tuple, converter=tuple)
    is_output: bool = False


################################################################################
# Interface
################################################################################


@typechecked
def strategies_from_spec(
    spec: Union[HplSpecification, Iterable[HplProperty]],
    input_channels: Mapping[str, str],
    type_defs: Mapping[str, MessageType],
    assumptions: Optional[Iterable[HplProperty]] = None,
) -> Set[Rule]:
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
# State Machine Builder
################################################################################


@frozen
class InputRule:
    channel: str
    predicate: HplPredicate = field(factory=HplVacuousTruth)
    inactive_monitors: Iterable[int] = field(factory=tuple, converter=tuple)
    safe_monitors: Iterable[int] = field(factory=tuple, converter=tuple)
    active_monitors: Iterable[int] = field(factory=tuple, converter=tuple)

    def fork(
        self,
        phi: HplPredicate,
        inactive_monitors: Optional[Iterable[int]] = None,
        safe_monitors: Optional[Iterable[int]] = None,
        active_monitors: Optional[Iterable[int]] = None,
    ) -> 'InputRule':
        inactive_monitors = (inactive_monitors or ()) + self.inactive_monitors
        safe_monitors = (safe_monitors or ()) + self.safe_monitors
        active_monitors = (active_monitors or ()) + self.active_monitors
        phi = phi.join(self.predicate)
        return evolve(
            self,
            predicate=phi,
            inactive_monitors=inactive_monitors,
            safe_monitors=safe_monitors,
            active_monitors=active_monitors,
        )


def f(properties: Iterable[HplProperty], input_channels: Iterable[str]):
    defaults = _get_default_predicates(properties)
    rulemap = {c: [InputRule(c, predicate=defaults[c])] for c in input_channels}
    i = 0
    for prop in properties:
        i += 1
        if prop.scope.has_activator:
            for event in prop.scope.activator.simple_events():
                assert isinstance(event, HplSimpleEvent)
                rules: List[InputRule] = rulemap.get(event.name, [])
                forks = [rule.fork(event.predicate, inactive_monitors=(i,)) for rule in rules]
                rules.extend(forks)
        if prop.scope.has_terminator:
            for event in prop.scope.terminator.simple_events():
                assert isinstance(event, HplSimpleEvent)
                rules: List[InputRule] = rulemap.get(event.name, [])
                forks = [rule.fork(event.predicate, safe_monitors=(i,)) for rule in rules]
                rules.extend(forks)
        if prop.pattern.is_absence:
            if prop.scope.is_global:
                continue  # already handled
            for event in prop.pattern.behaviour.simple_events():
                assert isinstance(event, HplSimpleEvent)
                rules: List[InputRule] = rulemap.get(event.name, [])
                phi = event.predicate.negate()
                forks = [rule.fork(phi, active_monitors=(i,)) for rule in rules]
                rules.extend(forks)


def _get_default_predicates(properties: Iterable[HplProperty]) -> Mapping[str, HplPredicate]:
    predicates: Mapping[str, HplPredicate] = defaultdict(HplVacuousTruth)
    for prop in properties:
        if not prop.scope.is_global:
            continue
        if not prop.pattern.is_absence:
            continue
        for event in prop.pattern.behaviour.simple_events():
            assert isinstance(event, HplSimpleEvent)
            phi = event.predicate.negate()
            predicates[event.name] = predicates[event.name].join(phi)
    return predicates


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
