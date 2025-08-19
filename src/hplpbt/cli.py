# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

"""
Module that contains the command line program.

Why does this file exist, and why not put this in __main__?

  In some cases, it is possible to import `__main__.py` twice.
  This approach avoids that. Also see:
  https://click.palletsprojects.com/en/5.x/setuptools/#setuptools-integration

Some of the structure of this file came from this StackExchange question:
  https://softwareengineering.stackexchange.com/q/418600
"""

###############################################################################
# Imports
###############################################################################

from typing import Any, Final

import argparse
from pathlib import Path
import sys
from traceback import print_exc

from ruamel.yaml import YAML

from hplpbt import __version__ as current_version
from hplpbt.gen import generate_tests, generate_tests_from_files

###############################################################################
# Constants
###############################################################################

PROG: Final[str] = 'hplpbt'

###############################################################################
# Argument Parsing
###############################################################################


def parse_arguments(argv: list[str] | None) -> dict[str, Any]:
    description = 'Property-based test generator for HPL properties.'
    parser = argparse.ArgumentParser(prog=PROG, description=description)

    parser.add_argument(
        '--version',
        action='version',
        version=f'{PROG} {current_version}',
        help='prints the program version',
    )

    parser.add_argument('-o', '--output', help='output file to place generated code')

    parser.add_argument(
        '-f',
        '--files',
        action='store_true',
        help='process args as HPL files (default: HPL properties)',
    )

    parser.add_argument(
        'msg_types',
        type=Path,
        help='path to a YAML/JSON file with types for each message channel',
    )

    parser.add_argument('specs', nargs='+', help='input properties')

    args = parser.parse_args(args=argv)
    return vars(args)


###############################################################################
# Setup
###############################################################################


def load_configs(args: dict[str, Any]) -> dict[str, Any]:
    try:
        config: dict[str, Any] = {}
        # with open(args['config_path'], 'r') as file_pointer:
        # yaml.safe_load(file_pointer)

        # arrange and check configs here

        return config
    except Exception as err:
        # log or raise errors
        print(err, file=sys.stderr)
        if str(err) == 'Really Bad':
            raise err

        # Optional: return some sane fallback defaults.
        sane_defaults: dict[str, Any] = {}
        return sane_defaults


###############################################################################
# Commands
###############################################################################


def handle_test_generation(args: dict[str, Any], _configs: dict[str, Any]) -> int:
    yaml = YAML(typ='safe')
    msg_types: dict[str, str] = yaml.load(args['msg_types'])

    if args.get('files'):
        output: str = generate_tests_from_files(args['specs'], msg_types)
    else:
        output = generate_tests(args['specs'], msg_types)

    output_path: str | None = args.get('output')
    if output_path:
        path: Path = Path(output_path).resolve(strict=False)
        path.write_text(output, encoding='utf-8')
    else:
        print(output)

    return 0  # success


###############################################################################
# Entry Point
###############################################################################


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)

    try:
        config = load_configs(args)
        return handle_test_generation(args, config)

    except KeyboardInterrupt:
        print('Aborted manually.', file=sys.stderr)
        return 1

    except Exception as err:
        print('An unhandled exception crashed the application!')
        print(err)
        print_exc()
        return 1
