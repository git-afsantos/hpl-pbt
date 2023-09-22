# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

###############################################################################
# Imports
###############################################################################

from typing import List

from hpl.ast import HplBinaryOperator, HplExpression, HplQuantifier, HplUnaryOperator
from hpl.rewrite import (
    And,
    Forall,
    Not,
    Or,
    empty_test,
    is_and,
    is_false,
    is_implies,
    is_not,
    is_or,
    is_true,
)
from typeguard import typechecked

################################################################################
# Logic Convenience Functions
################################################################################


@typechecked
def split_and(phi: HplExpression) -> List[HplExpression]:
    conditions: List[HplExpression] = []
    stack: List[HplExpression] = [phi]
    while stack:
        expr: HplExpression = stack.pop()
        # preprocessing
        if is_true(expr):
            continue
        if is_false(expr):
            raise ValueError(f'unsatisfiable: {phi}')
        expr = _and_presplit_transform(expr)
        # expr should be either an And or something undivisible
        # splits
        if is_and(expr):
            assert isinstance(expr, HplBinaryOperator)
            stack.append(expr.a)
            stack.append(expr.b)
        else:
            conditions.append(expr)
    return conditions


@typechecked
def _and_presplit_transform(phi: HplExpression) -> HplExpression:
    # This should not need a loop any longer
    # previous = None
    # while phi is not previous:
    #     previous = phi
    #     if is_not(phi):
    #         phi = _split_and_not(phi)
    #     elif phi.is_quantifier:
    #         phi = _split_and_quantifier(phi)

    if is_not(phi):
        return _split_and_not(phi)
    if phi.is_quantifier:
        return _split_and_quantifier(phi)
    return phi  # atomic


@typechecked
def _split_and_not(neg: HplUnaryOperator) -> HplExpression:
    """
    Transform a Negation into either an And or something undivisible
    """
    phi: HplExpression = neg.operand
    if is_not(phi):
        # ~~p  ==  p
        assert isinstance(phi, HplUnaryOperator)
        return _and_presplit_transform(phi.operand)
    if is_or(phi):
        # ~(a | b)  ==  ~a & ~b
        assert isinstance(phi, HplBinaryOperator)
        return And(Not(phi.a), Not(phi.b))
    if is_implies(phi):
        # ~(a -> b)  ==  ~(~a | b)  ==  a & ~b
        return And(phi.a, Not(phi.b))
    if phi.is_quantifier:
        assert isinstance(phi, HplQuantifier)
        if phi.is_existential:
            # (~E x: p)  ==  (A x: ~p)
            p = Not(phi.condition)
            assert p.contains_reference(phi.variable)
            phi = Forall(phi.variable, phi.domain, p)
            return _split_and_quantifier(phi)
    return neg


@typechecked
def _split_and_quantifier(quant: HplQuantifier) -> HplExpression:
    """
    Transform a Quantifier into either an And or something undivisible
    """
    var: str = quant.variable
    phi: HplExpression = quant.condition
    if quant.is_universal:
        # (A x: p & q)  ==  ((A x: p) & (A x: q))
        phi = _and_presplit_transform(phi)
        if is_and(phi):
            assert isinstance(phi, HplBinaryOperator)
            if phi.a.contains_reference(var):
                qa = Forall(var, quant.domain, phi.a)
            else:
                qa = Or(empty_test(quant.domain), phi.a)
            if phi.b.contains_reference(var):
                qb = Forall(var, quant.domain, phi.b)
            else:
                qb = Or(empty_test(quant.domain), phi.b)
            return And(qa, qb)
    elif quant.is_existential:
        # (E x: p -> q)  ==  (E x: ~p | q)
        # (E x: p | q)  ==  ((E x: p) | (E x: q))
        pass  # not worth splitting disjunctions
    return quant  # nothing to do
