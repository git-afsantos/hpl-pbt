# HPL PBT

This project provides tools to generate [Hypothesis](https://github.com/HypothesisWorks/hypothesis) Property-based Tests based on [HPL properties](https://github.com/git-afsantos/hpl-specs).

- [Installation](#installation)
- [Usage](#usage)
- [GitHub Features](#github-features)
- [Tooling](#tooling)

## Installation

Install this package with

```bash
pip install hpl-pbt
```

## Usage

### Test Generation

This package provides a library and a command line interface from which you can generate property-based test templates with a simple command.

#### Requirements

First, you will need a mapping of input message channels to data types.
This can be specified in YAML or JSON formats, when provided as a file. For example:

```yaml
%YAML 1.2
# file: inputs.yaml
---
messages:
    int_msg: IntMessage
    string_msg: UInt8Message
    object_msg: ObjectMessage
data:
    IntMessage:
        params:
            - int
    UInt8Message:
        params:
            -
                type: int
                min_value: 0
                max_value: 255
    ObjectMessage:
        import: third_party_pkg
        params:
            - bool
            - int[]
            -
                name: keyword_arg
                type: Subtype[]
    Subtype:
        params:
            - string
```

Then, you will also need to provide a HPL specification.
This can be either a list of properties, or a `.hpl` file, depending on how you want to use this package.

#### As a Standalone Tool

This package provides the `hpl-pbt` CLI script.

**Required Arguments**

1. a path to the data type mapping file
2. either a list of properties or a list of `.hpl` files, depending on flags

**Optional Arguments**

- flag `-f` or `--files`: treat positional arguments as a list of paths to `.hpl` files instead of a list of properties
- argument `-o` or `--output`: pass a path to an output file for the generated code (default: print to screen)

**Example**

```bash
# generating tests from a specification file
hpl-pbt -f -o tests.py inputs.yaml properties.hpl
```

#### As a Library

This repository provides the `hplpbt` Python package.

**Example**

```python
from typing import Any, Dict, List
from pathlib import Path
from hpl.ast import HplProperty
from hpl.parser import property_parser
from hplpbt.rclpy import generate_tests

parser = property_parser()

msg_types: Dict[str, Any] = { 'messages': {}, 'data': {} }
properties: List[HplProperty] = [parser.parse('globally: no /a {data > 0}')]

test_code: str = generate_tests(properties, msg_types)

path: Path = Path('test_script.py')
path.write_text(test_code, encoding='utf-8')
```

## GitHub Features

The `.github` directory comes with a number of files to configure certain GitHub features.

- Various Issue templates can be found under `ISSUE_TEMPLATE`.
- A Pull Request template can be found at `PULL_REQUEST_TEMPLATE.md`.
- Automatically mark issues as stale after a period of inactivity. The configuration file can be found at `.stale.yml`.
- Keep package dependencies up to date with Dependabot. The configuration file can be found at `dependabot.yml`.
- Keep Release Drafts automatically up to date with Pull Requests, using the [Release Drafter GitHub Action](https://github.com/marketplace/actions/release-drafter). The configuration file can be found at `release-drafter.yml` and the workflow at `workflows/release-drafter.yml`.
- Automatic package building and publishing when pushing a new version tag to `main`. The workflow can be found at `workflows/publish-package.yml`.

## Tooling

This package sets up various `tox` environments for static checks, testing, building and publishing.
It is also configured with `pre-commit` hooks to perform static checks and automatic formatting.

If you do not use `tox`, you can build the package with `build` and install a development version with `pip`.

Assume `cd` into the repository's root.

To install the `pre-commit` hooks:

```bash
pre-commit install
```

To run type checking:

```bash
tox -e typecheck
```

To run linting tools:

```bash
tox -e lint
```

To run automatic formatting:

```bash
tox -e format
```

To run tests:

```bash
tox
```

To build the package:

```bash
tox -e build
```

To build the package (with `build`):

```bash
python -m build
```

To clean the previous build files:

```bash
tox -e clean
```

To test package publication (publish to *Test PyPI*):

```bash
tox -e publish
```

To publish the package to PyPI:

```bash
tox -e publish -- --repository pypi
```

To install an editable version:

```bash
pip install -e .
```
