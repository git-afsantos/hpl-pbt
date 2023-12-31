# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

###############################################################################
# Imports
###############################################################################

from typing import Any, Container, Final, Iterable, List, Mapping, Optional, Set, Tuple, Union

from pathlib import Path

from hpl.ast import HplEvent, HplProperty, HplSimpleEvent, HplSpecification
from hpl.parser import property_parser, specification_parser
from hpl.rewrite import canonical_form
from hplrv.gen import TemplateRenderer

from hplpbt.logic import check_atomic_conditions_in_canonical_form, split_assumptions
from hplpbt.strategies.traces import strategies_from_spec, TraceStrategy
from hplpbt.types import BuiltinParameterType, type_map_from_data

###############################################################################
# Constants
###############################################################################

MSG_TYPES_KEY_CHANNELS: Final[str] = 'messages'
MSG_TYPES_KEY_TYPEDEFS: Final[str] = 'data'

###############################################################################
# Interface
###############################################################################


def generate_tests_from_files(
    paths: Iterable[Union[Path, str]],
    msg_types: Mapping[str, Mapping[str, Any]],
) -> str:
    """
    Produces test code snippets,
    given a list of paths to HPL files with specifications.
    """
    parser = specification_parser()
    properties: List[HplProperty] = []
    for input_path in paths:
        path: Path = Path(input_path).resolve(strict=True)
        text: str = path.read_text(encoding='utf-8').strip()
        spec: HplSpecification = parser.parse(text)
        properties.extend(spec.properties)
    return generate_tests(properties, msg_types)


def generate_tests_from_spec(
    spec: Union[HplSpecification, str],
    msg_types: Mapping[str, Mapping[str, Any]],
) -> str:
    """
    Produces test code snippets, given an HPL specification.
    """
    if not isinstance(spec, HplSpecification):
        parser = specification_parser()
        spec = parser.parse(spec)
    return generate_tests(spec.properties, msg_types)


def generate_tests(
    input_properties: Iterable[Union[str, HplProperty]],
    msg_types: Mapping[str, Mapping[str, Any]],
) -> str:
    _validate_msg_types(msg_types)
    input_properties = _parsed_properties(input_properties)
    input_properties = _discard_useless_properties(input_properties)
    input_channels = msg_types[MSG_TYPES_KEY_CHANNELS]
    output_channels = _get_output_channels(input_properties, input_channels)
    canonical_properties = [p for ps in map(canonical_form, input_properties) for p in ps]
    # assumptions: properties about system input channels
    # behaviour: properties about system output channels
    assumptions, behaviour = split_assumptions(canonical_properties, input_channels)
    _check_atomic_conditions_assumptions(assumptions, input_channels)
    _check_atomic_conditions_testable(behaviour, input_channels)
    type_defs = type_map_from_data(msg_types[MSG_TYPES_KEY_TYPEDEFS])
    trace_strategies = strategies_from_spec(
        behaviour,
        input_channels,
        type_defs,
        assumptions=assumptions,
    )
    r = TemplateRenderer.from_pkg_data(pkg='hplpbt', template_dir='templates')
    msg_strategies = set()
    for trace_strategy in trace_strategies:
        msg_strategies.update(trace_strategy.all_msg_strategies())
    data = {
        'trace_strategies': trace_strategies,
        'msg_strategies': msg_strategies,
        'imports': _import_list(trace_strategies),
        'assumptions': assumptions,
        'behaviour': behaviour,
    }
    return r.render_template('test-script.py.jinja', data)


###############################################################################
# Helper Functions
###############################################################################


def _parsed_properties(properties: Iterable[Union[str, HplProperty]]) -> List[HplProperty]:
    parser = property_parser()
    return [p if isinstance(p, HplProperty) else parser.parse(p) for p in properties]


def _get_output_channels(
    properties: Iterable[HplProperty],
    input_channels: Container[str],
) -> Set[str]:
    output_channels = set()
    for prop in properties:
        if prop.scope.has_activator:
            for event in prop.scope.activator.simple_events():
                assert isinstance(event, HplSimpleEvent)
                if event.name not in input_channels:
                    output_channels.add(event.name)
        if prop.scope.has_terminator:
            for event in prop.scope.terminator.simple_events():
                assert isinstance(event, HplSimpleEvent)
                if event.name not in input_channels:
                    output_channels.add(event.name)
        if prop.pattern.trigger is not None:
            for event in prop.pattern.trigger.simple_events():
                assert isinstance(event, HplSimpleEvent)
                if event.name not in input_channels:
                    output_channels.add(event.name)
        for event in prop.pattern.behaviour.simple_events():
            assert isinstance(event, HplSimpleEvent)
            if event.name not in input_channels:
                output_channels.add(event.name)
    return output_channels


def _import_list(trace_strategies: Iterable[TraceStrategy]) -> List[Tuple[str, List[str]]]:
    imports: Mapping[str, List[str]] = {}
    for trace_strategy in trace_strategies:
        for msg_strategy in trace_strategy.all_msg_strategies():
            classes = imports.get(msg_strategy.package)
            if classes is None:
                classes = [msg_strategy.class_name]
                imports[msg_strategy.package] = classes
            else:
                classes.append(msg_strategy.class_name)
    import_list: List[Tuple[str, List[str]]] = []
    for package in sorted(imports.keys()):
        import_list.append((package, sorted(imports[package])))
    return import_list


def _discard_useless_properties(properties: Iterable[HplProperty]) -> List[HplProperty]:
    # drop unbounded liveness properties from the specification
    # these do not impose any obligations on tests
    new_properties = []
    for prop in properties:
        if not prop.scope.has_terminator:
            if prop.pattern.is_existence and not prop.pattern.has_max_time:
                continue  # discard
            if prop.pattern.is_response and not prop.pattern.has_max_time:
                continue  # discard
        new_properties.append(prop)
    return new_properties


def _check_atomic_conditions_assumptions(
    assumptions: Iterable[HplProperty],
    input_channels: Container[str],
):
    # assumptions will avoid generating what they can
    # if we can rely on behaviour events, we do not need to generate specific data
    for prop in assumptions:
        _check_can_generate_event(prop.pattern.behaviour, input_channels)


def _check_atomic_conditions_testable(
    properties: Iterable[HplProperty],
    input_channels: Container[str],
):
    for prop in properties:
        _check_can_generate_event(prop.scope.activator, input_channels)
        _check_can_generate_event(prop.scope.terminator, input_channels)
        _check_can_generate_event(prop.pattern.trigger, input_channels)


def _check_can_generate_event(any_event: Optional[HplEvent], input_channels: Container[str]):
    if any_event is None:
        return
    for event in any_event.simple_events():
        assert isinstance(event, HplSimpleEvent)
        if event.name in input_channels:
            check_atomic_conditions_in_canonical_form(event.predicate)


###############################################################################
# Input Validation
###############################################################################


def _validate_msg_types(msg_types: Mapping[str, Mapping[str, Any]]):
    if not isinstance(msg_types, Mapping):
        raise TypeError(f'expected Mapping[str, Mapping[str, Any]], got {msg_types!r}')
    typedefs = set()
    data: Mapping[str, Mapping[str, Any]] = msg_types[MSG_TYPES_KEY_TYPEDEFS]
    for key, value in data.items():
        if not isinstance(key, str):
            raise TypeError(f"expected str keys in '{MSG_TYPES_KEY_TYPEDEFS}', found {key!r}")
        if not isinstance(value, Mapping):
            raise TypeError(
                f"expected Mapping[str, Any] values for each '{MSG_TYPES_KEY_TYPEDEFS}' key,"
                f' found {value!r}'
            )
        _validate_type_def(key, value, data)
        typedefs.add(key)
    messages: Mapping[str, str] = msg_types[MSG_TYPES_KEY_CHANNELS]
    for key, value in messages.items():
        if not isinstance(key, str):
            raise TypeError(f"expected str keys in '{MSG_TYPES_KEY_CHANNELS}', found {key!r}")
        if not isinstance(value, str):
            raise TypeError(
                f"expected str values for each '{MSG_TYPES_KEY_CHANNELS}' key,"
                f" found {value!r}"
            )
        if value not in typedefs:
            raise ValueError(f"unknown message type {value!r}")


def _validate_type_def(name: str, type_def: Mapping[str, Any], custom_types: Container[str]):
    module: Optional[str] = type_def.get('import')
    if module is not None and not isinstance(module, str):
        raise TypeError(f"expected str 'import' value for {name}, found {module!r}")
    try:
        _validate_type_pos_args(type_def.get('args', ()), custom_types)
        _validate_type_kwargs(type_def.get('kwargs', {}), custom_types)
    except TypeError as err:
        raise TypeError(f"in type definition '{name}': {err}") from err
    except ValueError as err:
        raise ValueError(f"in type definition '{name}': {err}") from err


def _validate_type_pos_args(args: Iterable[str], custom_types: Container[str]):
    if not isinstance(args, Iterable):
        raise TypeError(f"expected Iterable 'args' value, found {args!r}")
    for param_type in args:
        try:
            _validate_param_type(param_type, custom_types)
        except TypeError as err:
            raise TypeError(f"in 'args' entry: {err}") from err
        except ValueError as err:
            raise ValueError(f"in 'args' entry: {err}") from err


def _validate_type_kwargs(kwargs: Mapping[str, str], custom_types: Container[str]):
    if not isinstance(kwargs, Mapping):
        raise TypeError(f"expected Mapping 'kwargs' value, found {kwargs!r}")
    for name, param_type in kwargs.items():
        if not isinstance(name, str):
            raise TypeError(f"expected str 'kwargs' key, found {name!r}")
        try:
            _validate_param_type(param_type, custom_types)
        except TypeError as err:
            raise TypeError(f"in 'kwargs' value for '{name}': {err}") from err
        except ValueError as err:
            raise ValueError(f"in 'kwargs' value for '{name}': {err}") from err


def _validate_param_type(param_type: str, custom_types: Container[str]):
    if not isinstance(param_type, str):
        raise TypeError(f"expected str parameter type, found {param_type!r}")
    param_type = param_type.split('[', maxsplit=1)[0]
    try:
        BuiltinParameterType(param_type)
    except ValueError:
        if param_type in custom_types:
            return
        raise ValueError(f'undefined parameter type: {param_type!r}')
