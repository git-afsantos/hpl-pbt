# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

###############################################################################
# Imports
###############################################################################

from typing import Dict, Final, Iterable, List, Mapping, Optional, Set, Tuple, Union

from attrs import define, field, frozen
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
from hpl.rewrite import simplify, split_and
# from hpl.types import TypeToken
from typeguard import check_type, typechecked

from hplpbt.strategies.ast import (
    Assignment,
    Assumption,
    DataStrategy,
    FunctionCall,
    RandomArray,
    RandomBool,
    RandomFloat,
    RandomInt,
    RandomSpecial,
    RandomString,
    Reference,
    Statement,
)
from hplpbt.types import MessageType, ParameterDefinition

################################################################################
# Constants
################################################################################


STRATEGY_FACTORIES: Final[Mapping[str, DataStrategy]] = {
    'bool': RandomBool,
    'int': RandomInt,
    'uint': RandomInt.uint,
    'uint8': RandomInt.uint8,
    'uint16': RandomInt.uint16,
    'uint32': RandomInt.uint32,
    'uint64': RandomInt.uint64,
    'int8': RandomInt.int8,
    'int16': RandomInt.int16,
    'int32': RandomInt.int32,
    'int64': RandomInt.int64,
    'float': RandomFloat,
    'float32': RandomFloat.float32,
    'float64': RandomFloat.float64,
    'string': RandomString,
}


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

    def _strategies_for_type(
        self,
        type_def: MessageType,
        visited: Optional[Set[str]] = None,
    ) -> Set[MessageStrategy]:
        visited = set() if visited is None else visited
        strategies = set()
        # mark self as visited to avoid cyclic dependencies
        visited.add(type_def.name)
        for type_name in type_def.dependencies():
            other: MessageType = self.type_defs[type_name]
            # avoid cyclic dependencies
            if other.name not in visited:
                strategies.update(self._strategies_for_type(other, visited=visited))
        strategies.add(self._strategy_for_type(type_def))
        return strategies

    def _strategy_for_type(self, type_def: MessageType) -> MessageStrategy:
        strategy = self._cache.get(type_def.name)
        if strategy is not None:
            return strategy
        builder = SingleMessageStrategyBuilder(type_def)
        strategy = builder.build()
        self._cache[type_def.name] = strategy
        return strategy


@frozen
class SingleMessageStrategyBuilder:
    # the message type to build instances of
    message_type: MessageType
    # arguments to provide to the message type constructor
    positional_arguments: List[Tuple[str, ParameterDefinition]] = field(factory=list)
    keyword_arguments: List[Tuple[str, ParameterDefinition]] = field(factory=list)
    # assumptions about the arguments to the message type constructor
    preconditions: List[HplExpression] = field(factory=list)

    def build(self) -> MessageStrategy:
        # 1. Iterate over message type parameters and create local variables
        #    to hold generated data to pass into the constructor.
        #    Store the name mapping from user input variables to local variables.
        refmap = self._generate_argument_names()

        # 2. Simplify the message type's precondition predicate and break it
        #    down into a list of simpler conditions.
        #    Replace references to user-input variable names with references
        #    to the generated local variable names.
        self._preprocess_preconditions(refmap)

        body = self._generate_body_from_type_params()
        assert len(body) > 0
        assert isinstance(body[-1], Assignment)
        ret_var = body[-1].variable
        ret_type = self.message_type.qualified_name
        return MessageStrategy(f'gen_{self.message_type.name}', ret_type, ret_var, body=body)

    def _generate_argument_names(self) -> Dict[str, str]:
        # returns a reference map, mapping user-input names to generated names
        # resets/rebuilds positional and keyword argument lists
        # self.positional_arguments = []
        # self.keyword_arguments = []
        refmap = {}
        for i, param in enumerate(self.message_type.positional_parameters):
            variable = f'arg{i}'
            self.positional_arguments.append((variable, param))
            refmap[f'_{i}'] = variable
        i = len(refmap)
        for name, param in self.message_type.keyword_parameters.items():
            variable = f'arg{i}'
            self.keyword_arguments.append((variable, param))
            assert name not in refmap
            refmap[name] = variable
            i += 1
        return refmap

    def _preprocess_preconditions(self, refmap: Mapping[str, str]):
        pre = self.message_type.precondition
        # self.preconditions = []
        if pre.is_vacuous:
            assert pre.is_true
            return
        # break the first level of conjunctions
        conditions = split_and(simplify(pre.condition))
        # replace references to user-input variables
        for phi in conditions:
            for alias in phi.external_references():
                var = refmap[alias]
                phi = phi.replace_var_reference(alias, HplVarReference(f'@{var}'))
            # store only the final form of the expression
            self.preconditions.append(phi)

    def _generate_body_from_type_params(self) -> List[Statement]:
        body = []
        args = []
        for name, param in self.positional_arguments:
            body.extend(self._generate_param(name, param))
            args.append(Reference(name))
        kwargs = []
        for name, param in self.keyword_arguments:
            body.extend(self._generate_param(name, param))
            kwargs.append((name, Reference(name)))
        for phi in self.preconditions:
            body.append(Assumption(phi))

        name = self.message_type.qualified_name
        constructor = FunctionCall(name, arguments=args, keyword_arguments=kwargs)
        body.append(Assignment('msg', constructor))
        return body

    def _generate_param(self, name: str, param: ParameterDefinition) -> List[Statement]:
        statements = []
        factory = STRATEGY_FACTORIES.get(param.base_type)
        s: DataStrategy = RandomSpecial(param.base_type) if factory is None else factory()
        if param.is_array:
            s = RandomArray(s)
        statements.append(Assignment.draw(name, s))
        return statements
