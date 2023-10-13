# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

# This file provides a series of utilities related to generating code
# for message strategies. Things such as sorting statements based on
# their dependencies.

###############################################################################
# Imports
###############################################################################

from typing import Iterable, List, Set

from attrs import frozen

from hplpbt.errors import CyclicDependencyError
from hplpbt.strategies.ast import Statement

################################################################################
# Sortable Statements
################################################################################


@frozen
class SortableStatement:
    ast: Statement  # the statement itself
    pending: Set[str]  # dependencies that are still pending

    @classmethod
    def of(cls, statement: Statement) -> 'SortableStatement':
        return cls(statement, statement.dependencies())


@frozen
class StatementGroup:
    """Logical grouping of all required statements to fully build a data field."""
    statements: List[SortableStatement]
    data_field: str


def sort_statements(statements: Iterable[Statement]) -> List[Statement]:
    sorted_statements: List[Statement] = []
    queue: List[SortableStatement] = [SortableStatement.of(s) for s in reversed(statements)]
    while queue:
        progress: bool = False
        new_queue: List[SortableStatement] = []
        while queue:
            statement = queue.pop()
            if statement.pending:
                new_queue.append(statement)
            else:
                progress = True
                sorted_statements.append(statement.ast)
                for variable in statement.ast.assignments():
                    for other in queue:
                        if variable in other.pending:
                            other.pending.remove(variable)
                    for other in new_queue:
                        if variable in other.pending:
                            other.pending.remove(variable)
        queue = new_queue
        if not progress:
            report_cyclic_dependency(queue)
    return sorted_statements


def report_cyclic_dependency(statements: Iterable[SortableStatement]):
    v = set()
    n = len(statements)
    for statement in statements:
        v.update(statement.pending)
    raise CyclicDependencyError(f'unable to satisfy dependecies on {v} for {n} statements')
