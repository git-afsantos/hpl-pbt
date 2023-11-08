# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

###############################################################################
# Imports
###############################################################################

from typing import Any, Container, Final, Iterable, List, Mapping, Optional, Tuple, Union

from pathlib import Path

from attrs import frozen
from hpl.ast import HplProperty, HplSpecification
from hpl.parser import property_parser, specification_parser
from hpl.rewrite import canonical_form
from jinja2 import Environment, PackageLoader

from hplpbt.logic import split_assumptions
from hplpbt.strategies.messages import MessageStrategy, strategies_from_spec
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
    input_channels = msg_types[MSG_TYPES_KEY_CHANNELS]
    canonical_properties = [p for ps in map(canonical_form, input_properties) for p in ps]
    assumptions, behaviour = split_assumptions(canonical_properties, input_channels)
    type_defs = type_map_from_data(msg_types[MSG_TYPES_KEY_TYPEDEFS])
    msg_strategies = strategies_from_spec(behaviour, input_channels, type_defs)
    # parts = ['# assumptions']
    # parts.extend(map(repr, map(str, assumptions)))
    # parts.append('# behaviour')
    # parts.extend(map(repr, map(str, behaviour)))
    # parts.append('# strategies')
    # parts.extend(map(str, msg_strategies))
    # return '\n'.join(parts)
    r = TemplateRenderer.from_pkg_data()
    data = {'strategies': msg_strategies, 'imports': _import_list(msg_strategies)}
    return r.render_template('test-script.py.jinja', data)


###############################################################################
# Helper Functions
###############################################################################


@frozen
class TemplateRenderer:
    jinja_env: Environment

    @classmethod
    def from_pkg_data(
        cls,
        pkg: str = 'hplpbt',
        template_dir: str = 'templates'
    ) -> 'TemplateRenderer':
        return cls(Environment(
            loader=PackageLoader(pkg, template_dir),
            line_statement_prefix=None,
            line_comment_prefix=None,
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=False,
        ))

    def render_template(
        self,
        template_file: str,
        data: Mapping[str, Any],
        strip: bool = True,
        encoding: Optional[str] = None
    ) -> str:
        template = self.jinja_env.get_template(template_file)
        text = template.render(**data)
        if strip:
            text = text.strip()
        if encoding is None:
            return text
        return text.encode(encoding)


def _parsed_properties(properties: Iterable[Union[str, HplProperty]]) -> List[HplProperty]:
    parser = property_parser()
    return [p if isinstance(p, HplProperty) else parser.parse(p) for p in properties]


def _import_list(strategies: Iterable[MessageStrategy]) -> List[Tuple[str, List[str]]]:
    imports: Mapping[str, List[str]] = {}
    for strategy in strategies:
        classes = imports.get(strategy.package)
        if classes is None:
            classes = [strategy.class_name]
            imports[strategy.package] = classes
        else:
            classes.append(strategy.class_name)
    import_list: List[Tuple[str, List[str]]] = []
    for package in sorted(imports.keys()):
        import_list.append((package, sorted(imports[package])))
    return import_list


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
