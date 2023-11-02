# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

###############################################################################
# Imports
###############################################################################

from typing import Callable, Dict, Final, Iterable, List, Mapping, Optional, Set, Tuple, Union

from attrs import field, frozen
from attrs.validators import deep_iterable, deep_mapping, instance_of
from hpl.ast import (
    # HplArrayAccess,
    HplEvent,
    HplExpression,
    # HplFieldAccess,
    HplProperty,
    HplSimpleEvent,
    HplSpecification,
    HplVarReference,
)
from hpl.rewrite import simplify, split_and
from hplpbt.strategies.data import BasicDataFieldGenerator, BooleanFieldGenerator, NumberFieldGenerator, StringFieldGenerator
# from hpl.types import TypeToken
from typeguard import check_type, typechecked

from hplpbt.strategies._codegen import sort_statements
from hplpbt.strategies.ast import (
    Assignment,
    Assumption,
    DataStrategy,
    FunctionCall,
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


DATA_GENERATORS: Final[Mapping[str, Callable[[], BasicDataFieldGenerator]]] = {
    'bool': BooleanFieldGenerator.any_bool,
    'int': NumberFieldGenerator.any_int,
    'uint': NumberFieldGenerator.uint,
    'uint8': NumberFieldGenerator.uint8,
    'uint16': NumberFieldGenerator.uint16,
    'uint32': NumberFieldGenerator.uint32,
    'uint64': NumberFieldGenerator.uint64,
    'int8': NumberFieldGenerator.int8,
    'int16': NumberFieldGenerator.int16,
    'int32': NumberFieldGenerator.int32,
    'int64': NumberFieldGenerator.int64,
    'float': NumberFieldGenerator.any_float,
    'float32': NumberFieldGenerator.float32,
    'float64': NumberFieldGenerator.float64,
    'string': StringFieldGenerator.any_string,
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


@frozen
class MessageStrategyArgument:
    name: str
    original_name: str
    generator: BasicDataFieldGenerator

    def get_reference(self) -> Reference:
        return Reference(self.name)


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
    positional_arguments: List[Tuple[str, MessageStrategyArgument]] = field(factory=list)
    keyword_arguments: List[Tuple[str, MessageStrategyArgument]] = field(factory=list)
    # assumptions about the arguments to the message type constructor
    preconditions: List[HplExpression] = field(factory=list)
    # name of the variable containing the constructed message
    message_variable: str = 'msg'

    def build(self) -> MessageStrategy:
        # 1. Iterate over message type parameters and create local variables
        #    to hold generated data to pass into the constructor.
        #    Store the name mapping from user input variables to local variables.
        # 2. Create a data generator for each argument variable.
        refmap = self._create_strategy_arguments()

        # 3. Simplify the message type's precondition predicate and break it
        #    down into a list of simpler conditions.
        #    Replace references to user-input variable names with references
        #    to the generated local variable names.
        self._preprocess_preconditions(refmap)

        # 4. Iterate over argument assumptions and try to modify the strategies
        #    associated with each data generator; supply them with appropriate
        #    arguments, etc., to produce optimized code that does not rely
        #    solely on assumptions to discard invalid data.
        self._refine_data_generators()

        # 5. Produce a list of statements for each data generator and sort them
        #    according to their references and dependencies.
        #    Detect and report cyclic dependencies.
        #    Break down into suboptimal code if necessary to avoid cycles.
        body = self._generate_message_from_type_params()

        # 6. Construct the message object using the previously generated arguments.
        #    (Bundled into the previous step to avoid state variables)
        assert len(body) > 0
        assert isinstance(body[-1], Assignment)
        assert body[-1].variable == self.message_variable

        # 7. Create data constructors for each message field referenced in the
        #    post-conditions (assumptions about the generated message).

        # 8. Iterate over the post-conditions and try to build values
        #    that are correct by construction.

        # 9. Produce a list of statements for each data generator and sort them
        #    according to their references and dependencies.

        # 10. Return the fully constructed message strategy.
        ret_type: str = self.message_type.qualified_name
        function_name: str = f'gen_{self.message_type.name}'
        return MessageStrategy(function_name, ret_type, self.message_variable, body=body)

    def _create_strategy_arguments(self) -> Dict[str, str]:
        # returns a reference map, mapping user-input names to generated names
        # resets/rebuilds positional and keyword argument lists
        self.positional_arguments.clear()
        self.keyword_arguments.clear()
        refmap = {}
        for i, param in enumerate(self.message_type.positional_parameters):
            variable = f'arg{i}'
            original_name = f'_{i}'
            gen = self._param_data_generator(param)
            arg = MessageStrategyArgument(variable, original_name, gen)
            self.positional_arguments.append((variable, arg))
            refmap[original_name] = variable
        i = len(refmap)
        for name, param in self.message_type.keyword_parameters.items():
            variable = f'arg{i}'
            gen = self._param_data_generator(param)
            arg = MessageStrategyArgument(variable, original_name, gen)
            self.keyword_arguments.append((variable, arg))
            assert name not in refmap
            refmap[name] = variable
            i += 1
        return refmap

    def _param_data_generator(self, param: ParameterDefinition) -> BasicDataFieldGenerator:
        factory = DATA_GENERATORS.get(param.base_type)
        if factory is None:
            gen: BasicDataFieldGenerator = BasicDataFieldGenerator(RandomSpecial(param.base_type))
        else:
            gen = factory()
        #if param.is_array:
        #    s = RandomArray(s)
        return gen

    def _preprocess_preconditions(self, refmap: Mapping[str, str]):
        pre = self.message_type.precondition
        self.preconditions.clear()
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

    def _refine_data_generators(self):
        # create a mapping of each argument for easier access
        argmap: Dict[str, MessageStrategyArgument] = {}
        for name, arg in self.positional_arguments:
            argmap[name] = arg
        for name, arg in self.keyword_arguments:
            argmap[name] = arg

        # iterate over all preconditions
        for phi in self.preconditions:
            refs = phi.external_references()
            # ignore preconditions that do not have references to arguments
            # or preconditions that have references to unknown variables
            if not refs or any(not name in argmap for name in refs):
                continue
            # process precondition with data generators
            # use just one, other variables will be related to it
            for name in refs:
                arg = argmap[name].generator.assume(phi)
                break

    def _generate_message_from_type_params(self) -> List[Statement]:
        body = []
        args = []
        for name, arg in self.positional_arguments:
            body.extend(self._generate_argument(name, arg))
            args.append(Reference(name))
        kwargs = []
        for name, arg in self.keyword_arguments:
            body.extend(self._generate_argument(name, arg))
            kwargs.append((name, Reference(name)))
        for phi in self.preconditions:
            body.append(Assumption(phi))

        body = sort_statements(body)

        name = self.message_type.qualified_name
        constructor = FunctionCall(name, arguments=args, keyword_arguments=kwargs)
        body.append(Assignment(self.message_variable, constructor))
        return body

    def _generate_argument(self, name: str, arg: MessageStrategyArgument) -> List[Statement]:
        statements = []
        statements.append(Assignment.draw(name, arg.generator.strategy))
        return statements
