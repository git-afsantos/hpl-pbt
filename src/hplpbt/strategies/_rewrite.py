# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

################################################################################
# Imports
################################################################################

from hpl.ast import (
    And,
    Exists,
    HplBinaryOperator,
    HplDataAccess,
    HplExpression,
    HplQuantifier,
    HplUnaryOperator,
    HplValue,
    Not,
    Or,
)
from hpl.ast.expressions import BinaryOperatorDefinition
from hpl.rewrite import (
    _simplify,
    inverse_operator,
    is_and,
    is_false,
    is_iff,
    is_implies,
    is_not,
    is_or,
    is_true,
)
from typeguard import typechecked

################################################################################
# HPL Manipulation
################################################################################


@typechecked
def split_or(phi: HplExpression) -> list[HplExpression]:
    conditions: list[HplExpression] = []
    stack: list[HplExpression] = [phi]
    while stack:
        expr: HplExpression = stack.pop()
        # preprocessing
        if is_true(expr):
            # raise ValueError(f'tautology: {phi}')
            return [expr]
        if is_false(expr):
            continue
        expr = _or_presplit_transform(expr)
        # expr should be either an Or or something undivisible
        # splits
        if is_or(expr):
            assert isinstance(expr, HplBinaryOperator)
            stack.append(expr.a)
            stack.append(expr.b)
        else:
            conditions.append(expr)
    return conditions


@typechecked
def _or_presplit_transform(phi: HplExpression) -> HplExpression:
    if is_implies(phi):
        # a -> b  ==  ~a | b
        return Or(phi.a, Not(phi.b))
    if is_not(phi):
        return _split_or_not(phi)
    if phi.is_quantifier:
        return _split_or_quantifier(phi)
    return phi  # atomic


@typechecked
def _split_or_not(neg: HplUnaryOperator) -> HplExpression:
    """
    Transform a Negation into either an Or or something undivisible
    """
    phi: HplExpression = neg.operand
    if is_not(phi):
        # ~~p  ==  p
        assert isinstance(phi, HplUnaryOperator)
        return _or_presplit_transform(phi.operand)
    if is_and(phi):
        # ~(a & b)  ==  ~a | ~b
        assert isinstance(phi, HplBinaryOperator)
        return Or(Not(phi.a), Not(phi.b))
    if is_iff(phi):
        # ~(a == b)  ==  ~(a -> b) | ~(b -> a)  ==  (a & ~b) | (b & ~a)
        assert isinstance(phi, HplBinaryOperator)
        a = And(phi.a, Not(phi.b))
        b = And(phi.b, Not(phi.a))
        return Or(a, b)
    if phi.is_quantifier:
        assert isinstance(phi, HplQuantifier)
        if phi.is_universal:
            # (~A x: p)  ==  (E x: ~p)
            p = Not(phi.condition)
            assert p.contains_reference(phi.variable)
            phi = Exists(phi.variable, phi.domain, p)
            return _split_or_quantifier(phi)
    return neg


@typechecked
def _split_or_quantifier(quant: HplQuantifier) -> HplExpression:
    """
    Transform a Quantifier into either an Or or something undivisible
    """
    var: str = quant.variable
    phi: HplExpression = quant.condition
    if quant.is_existential:
        # (E x: p | q)  ==  ((E x: p) | (E x: q))
        phi = _or_presplit_transform(phi)
        if is_or(phi):
            assert isinstance(phi, HplBinaryOperator)
            if phi.a.contains_reference(var):
                qa = Exists(var, quant.domain, phi.a)
            else:
                qa = phi.a
            if phi.b.contains_reference(var):
                qb = Exists(var, quant.domain, phi.b)
            else:
                qb = phi.b
            return Or(qa, qb)
    elif quant.is_universal:
        # (A x: p <-> q)  ==  (A x: (p -> q) & (q -> p))
        # (A x: p & q)  ==  ((A x: p) & (A x: q))
        pass  # not worth splitting conjunctions
    return quant  # nothing to do


def canonical_form(expr: HplBinaryOperator) -> HplBinaryOperator:
    op: BinaryOperatorDefinition = expr.operator
    a: HplExpression = _simplify(expr.operand1)
    b: HplExpression = _simplify(expr.operand2)

    if a is expr.operand1 and b is expr.operand2:
        return _canonical_lhs(expr)
    else:
        return _canonical_lhs(HplBinaryOperator(op, a, b))


def _canonical_lhs(expr: HplBinaryOperator) -> HplBinaryOperator:
    op: BinaryOperatorDefinition = expr.operator
    a: HplExpression = expr.operand1
    b: HplExpression = expr.operand2

    if not op.is_comparison:
        return expr

    aIsRef: bool = _is_any_reference_type(a)
    bIsRef: bool = _is_any_reference_type(b)

    if aIsRef and not bIsRef:
        return expr
    if bIsRef and not aIsRef:
        return HplBinaryOperator(inverse_operator(op), b, a)

    # 1. self-references
    if a.contains_self_reference():
        return _push_transforms_to_rhs(expr)
    if b.contains_self_reference():
        expr = HplBinaryOperator(inverse_operator(op), b, a)
        return _push_transforms_to_rhs(expr)
    # 2. external variables
    if a.external_references():
        return _push_transforms_to_rhs(expr)
    if b.external_references():
        expr = HplBinaryOperator(inverse_operator(op), b, a)
        return _push_transforms_to_rhs(expr)
    return expr


def _push_transforms_to_rhs(expr: HplBinaryOperator) -> HplBinaryOperator:
    op: BinaryOperatorDefinition = expr.operator
    a: HplExpression = expr.operand1
    b: HplExpression = expr.operand2
    changed = False
    while a.is_operator:
        if isinstance(a, HplUnaryOperator):
            b = HplUnaryOperator(a.operator, b)
            a = a.operand
            op = inverse_operator(op)
            changed = True
        else:
            break
    return HplBinaryOperator(op, a, b) if changed else expr


def _is_any_reference_type(expr: HplExpression) -> bool:
    if expr.is_value:
        assert isinstance(expr, HplValue)
        return expr.is_reference
    if expr.is_accessor:
        assert isinstance(expr, HplDataAccess)
        return True
    return False
