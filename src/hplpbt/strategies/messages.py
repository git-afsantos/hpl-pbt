# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

###############################################################################
# Imports
###############################################################################

from typing import Any, Iterable, Mapping, Optional, Set, Union

from attrs import field, frozen
from attrs.validators import deep_iterable, deep_mapping, instance_of
from hpl.ast import HplEvent, HplProperty, HplSimpleEvent, HplSpecification
# from hpl.types import TypeToken
from typeguard import typechecked

from hplpbt.types import MessageType, message_from_data

################################################################################
# Message Strategy
################################################################################


@frozen
class MessageStrategy:
    name: str


###############################################################################
# Interface
###############################################################################


@typechecked
def strategies_from_spec(
    spec: Union[HplSpecification, Iterable[HplProperty]],
    input_channels: Mapping[str, str],
    type_defs: Mapping[str, Mapping[str, Any]],
    assumptions: Optional[Iterable[HplProperty]] = None,
) -> Set[MessageStrategy]:
    assumptions = assumptions if assumptions is not None else []
    msg_types = {name: message_from_data(name, data) for name, data in type_defs.items()}
    builder = MessageStrategyBuilder(input_channels, msg_types, assumptions=assumptions)
    return builder.build_from_spec(spec)


@typechecked
def strategies_from_property(
    hpl_property: HplProperty,
    input_channels: Mapping[str, str],
    type_defs: Mapping[str, Mapping[str, Any]],
    assumptions: Optional[Iterable[HplProperty]] = None,
) -> Set[MessageStrategy]:
    assumptions = assumptions if assumptions is not None else []
    msg_types = {name: message_from_data(name, data) for name, data in type_defs.items()}
    builder = MessageStrategyBuilder(input_channels, msg_types, assumptions=assumptions)
    return builder.build_from_property(property)


@typechecked
def strategies_from_event(
    event: HplEvent,
    input_channels: Mapping[str, str],
    type_defs: Mapping[str, Mapping[str, Any]],
    assumptions: Optional[Iterable[HplProperty]] = None,
) -> Set[MessageStrategy]:
    assumptions = assumptions if assumptions is not None else []
    msg_types = {name: message_from_data(name, data) for name, data in type_defs.items()}
    builder = MessageStrategyBuilder(input_channels, msg_types, assumptions=assumptions)
    return builder.build_from_event(event)


###############################################################################
# Message Strategy Builder
###############################################################################


@frozen
class MessageStrategyBuilder:
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
    ) -> Set[MessageStrategy]:
        if isinstance(spec, HplSpecification):
            spec = spec.properties
        strategies = set()
        for hpl_property in spec:
            strategies.update(self.build_from_property(hpl_property))
        return strategies

    def build_from_property(self, hpl_property: HplProperty) -> Set[MessageStrategy]:
        strategies = set()

        event = hpl_property.scope.activator
        if event is not None:
            strategies.update(self.build_from_event(event))
        event = hpl_property.scope.terminator
        if event is not None:
            strategies.update(self.build_from_event(event))

        if hpl_property.pattern.is_absence:
            pass

        elif hpl_property.pattern.is_existence:
            pass

        elif hpl_property.pattern.is_requirement:
            event = hpl_property.pattern.trigger
            assert event is not None
            strategies.update(self.build_from_event(event))

        elif hpl_property.pattern.is_response:
            event = hpl_property.pattern.trigger
            assert event is not None
            strategies.update(self.build_from_event(event))

        elif hpl_property.pattern.is_prevention:
            event = hpl_property.pattern.trigger
            assert event is not None
            strategies.update(self.build_from_event(event))

        else:
            raise TypeError(f'unknown HPL property type: {hpl_property!r}')

        return strategies

    def build_from_event(self, event: HplEvent) -> Set[MessageStrategy]:
        if event.is_simple_event:
            return self._build_strategies(event)
        return set(strat for msg in event.simple_events() for strat in self._build_strategies(msg))

    def _build_strategies(self, event: HplSimpleEvent) -> Set[MessageStrategy]:
        strategies = set()
        if event.name not in self.input_channels:
            return strategies
        type_name: str = self.input_channels[event.name]
        type_def: MessageType = self.type_defs[type_name]
        strategies.add(MessageStrategy(repr(type_def)))
        return strategies
