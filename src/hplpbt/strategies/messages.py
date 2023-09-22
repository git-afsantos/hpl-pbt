# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

###############################################################################
# Imports
###############################################################################

from typing import Iterable, List, Mapping, Optional, Set, Tuple, Union

from attrs import field, frozen
from attrs.validators import deep_iterable, deep_mapping, instance_of
from hpl.ast import (
    HplEvent,
    HplExpression,
    HplPredicate,
    HplProperty,
    HplSimpleEvent,
    HplSpecification,
    HplVarReference,
)
from hpl.rewrite import refactor_reference
# from hpl.types import TypeToken
from typeguard import check_type, typechecked

from hplpbt.strategies.ast import (
    Assignment,
    Assumption,
    FunctionCall,
    RandomArray,
    RandomBool,
    RandomFloat,
    RandomInt,
    RandomSpecial,
    RandomString,
    Statement,
)
from hplpbt.strategies.logic import split_and
from hplpbt.types import MessageType, ParameterDefinition

################################################################################
# Message Strategy
################################################################################


@frozen
class StrategyArgument:
    name: str
    strategy: str


@frozen
class MessageStrategy:
    """
    This is likely to generate a function body that is divided in three stages:
    1. initialize necessary arguments and other independent variables
    2. initialize the message object, using values from stage 1
    3. initialize dependent fields of the message itself and run assumptions
    """
    name: str
    return_type: str
    return_variable: str
    arguments: Iterable[StrategyArgument] = field(factory=tuple, converter=tuple)
    body: Iterable[Statement] = field(factory=tuple, converter=tuple)

    @body.validator
    def _check_body(self, _attribute, value: Iterable[Statement]):
        body: Iterable[Statement] = check_type(value, Iterable[Statement])
        for statement in body:
            if statement.is_assignment:
                assignment: Assignment = check_type(statement, Assignment)
                if assignment.variable == self.return_variable:
                    return
        raise ValueError(f'variable {self.return_variable} is undefined in {body!r}')

    def __str__(self) -> str:
        parts = ['@composite', f'def {self.name}(draw) -> {self.return_type}:']
        for statement in self.body:
            parts.append(f'    {statement}')
        # check the last statement for optimization
        if statement.is_assignment and statement.variable == self.return_variable:
            parts[-1] = f'    return {statement.expression}'
        else:
            parts.append(f'    return {self.return_variable}')
        return '\n'.join(parts)


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
        if event.name not in self.input_channels:
            return set()
        type_name: str = self.input_channels[event.name]
        type_def: MessageType = self.type_defs[type_name]
        return self._strategies_for_type(type_def)

    def _strategies_for_type(self, type_def: MessageType) -> Set[MessageStrategy]:
        strategies = set()
        for type_name in type_def.dependencies():
            other: MessageType = self.type_defs[type_name]
            strategies.update(self._strategies_for_type(other))
        strategies.add(self._strategy_for_type(type_def))
        return strategies

    def _strategy_for_type(self, type_def: MessageType) -> MessageStrategy:
        strategy = self._cache.get(type_def.name)
        if strategy is not None:
            return strategy

        body = self._generate_body_from_type_params(type_def)
        assert len(body) > 0
        assert isinstance(body[-1], Assignment)
        ret_var = body[-1].variable

        ret_type = type_def.qualified_name
        strategy = MessageStrategy(f'gen_{type_def.name}', ret_type, ret_var, body=body)
        self._cache[type_def.name] = strategy
        return strategy

    def _generate_body_from_type_params(self, type_def: MessageType) -> List[Statement]:
        body = []
        refmap = {}
        args = []
        for i, param in enumerate(type_def.positional_parameters):
            variable = f'arg{i}'
            body.extend(self._generate_param(variable, param))
            args.append(variable)
            refmap[f'_{i}'] = variable
        kwargs = []
        i += 1
        for name, param in type_def.keyword_parameters.items():
            variable = f'arg{i}'
            body.extend(self._generate_param(variable, param))
            kwargs.append((name, variable))
            refmap[name] = variable
            i += 1

        # FIXME not the right place
        conditions = self._preprocess_precondition(type_def.precondition)
        for phi in conditions:
            for alias in phi.external_references():
                var = refmap[alias]
                phi = phi.replace_var_reference(alias, HplVarReference(f'@{var}'))
            body.append(Assumption(phi))

        name = type_def.qualified_name
        constructor = FunctionCall(name, arguments=args, keyword_arguments=kwargs)
        body.append(Assignment('msg', constructor))
        return body

    def _generate_param(self, name: str, param: ParameterDefinition) -> List[Statement]:
        statements = []
        if param.base_type == 'bool':
            s = RandomBool()
        elif param.base_type == 'int':
            s = RandomInt()
        elif param.base_type == 'float':
            s = RandomFloat()
        elif param.base_type == 'string':
            s = RandomString()
        else:
            s = RandomSpecial(param.base_type)
        if param.is_array:
            s = RandomArray(s)
        statements.append(Assignment.draw(name, s))
        return statements

    def _preprocess_precondition(self, pre: HplPredicate) -> List[HplExpression]:
        if pre.is_vacuous:
            assert pre.is_true
            return []
        # break the first level of conjunctions
        return split_and(pre.condition)
