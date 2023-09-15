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

from hplpbt.types import MessageType

################################################################################
# Message Strategy
################################################################################


@frozen
class StrategyArgument:
    name: str
    strategy: str


@frozen
class MessageStrategy:
    name: str
    return_type: str
    arguments: Iterable[StrategyArgument] = field(factory=tuple, converter=tuple)


###############################################################################
# Interface
###############################################################################


@typechecked
def strategies_from_spec(
    spec: Union[HplSpecification, Iterable[HplProperty]],
    input_channels: Mapping[str, str],
    type_defs: Mapping[str, MessageType],
    assumptions: Optional[Iterable[HplProperty]] = None,
) -> Set[MessageStrategy]:
    assumptions = assumptions if assumptions is not None else []
    builder = MessageStrategyBuilder(input_channels, type_defs, assumptions=assumptions)
    return builder.build_from_spec(spec)


@typechecked
def strategies_from_property(
    hpl_property: HplProperty,
    input_channels: Mapping[str, str],
    type_defs: Mapping[str, MessageType],
    assumptions: Optional[Iterable[HplProperty]] = None,
) -> Set[MessageStrategy]:
    assumptions = assumptions if assumptions is not None else []
    builder = MessageStrategyBuilder(input_channels, type_defs, assumptions=assumptions)
    return builder.build_from_property(hpl_property)


@typechecked
def strategies_from_event(
    event: HplEvent,
    input_channels: Mapping[str, str],
    type_defs: Mapping[str, MessageType],
    assumptions: Optional[Iterable[HplProperty]] = None,
) -> Set[MessageStrategy]:
    assumptions = assumptions if assumptions is not None else []
    builder = MessageStrategyBuilder(input_channels, type_defs, assumptions=assumptions)
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
    _cache: Mapping[str, MessageStrategy] = field(factory=dict, init=False, eq=False, repr=False)

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
        strategies.add(self._strategy_for_type(type_def))
        return strategies

    def _strategy_for_type(self, type_def: MessageType) -> MessageStrategy:
        strategy = self._cache.get(type_def.name)
        if strategy is not None:
            return strategy

        arguments = []
        for param in type_def.positional_parameters:
            if param.is_array:
                arg = StrategyArgument('arg', 'list')
                # FIXME strategy sub arguments
            else:
                arg = StrategyArgument('arg', param.base_type)
            arguments.append(arg)

        for name, param in type_def.keyword_parameters.items():
            if param.is_array:
                arg = StrategyArgument(name, 'list')
                # FIXME strategy sub arguments
            else:
                arg = StrategyArgument(name, param.base_type)
            arguments.append(arg)

        strategy = MessageStrategy(type_def.name, type_def.qualified_name, arguments=arguments)
        self._cache[type_def.name] = strategy
        return strategy
