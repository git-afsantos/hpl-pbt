# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

################################################################################
# Imports
################################################################################

from hpl.ast import (
    And,
    Exists,
    HplBinaryOperator,
    HplExpression,
    HplQuantifier,
    HplUnaryOperator,
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
    is_self_or_field,
    is_true,
)
from typeguard import typechecked

################################################################################
# HPL Manipulation
################################################################################


@typechecked
def split_or(phi: HplExpression) -> List[HplExpression]:
    conditions: List[HplExpression] = []
    stack: List[HplExpression] = [phi]
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
        _canonical_lhs(expr)
    else:
        _canonical_lhs(HplBinaryOperator(op, a, b))



    # always push literals to the RHS
    if isinstance(b, HplLiteral):
        return expr if noop else HplBinaryOperator(op, a, b)

    # always push self references to the LHS
    if is_self_or_field(a, deep=True):
        return expr if noop else HplBinaryOperator(op, a, b)

    # (1 < x) == (x > 1)
    flip: bool = isinstance(a, HplLiteral)
    # (@y < x) == (x > @y)
    flip = flip or is_self_or_field(b, deep=True)
    if flip:
        if op.commutative:
            return expr.but(operand1=b, operand2=a)
        inv: Optional[BinaryOperatorDefinition] = INVERSE_OPERATORS.get(op)
        if inv is None:
            return expr if noop else HplBinaryOperator(op, a, b)
        return HplBinaryOperator(inv, b, a)

    if op.associative:
        left: bool = isinstance(a, HplBinaryOperator) and a.operator == op
        right: bool = isinstance(b, HplBinaryOperator) and b.operator == op
        if left and right:
            a1: HplExpression = a.operand1
            a2: HplExpression = a.operand2
            b1: HplExpression = b.operand1
            b2: HplExpression = b.operand2
            if isinstance(a2, HplLiteral):
                noop = False
                x = b1
                b1 = b2
                b2 = a2
                a2 = x
                # it is now possible to have redundancy on the LHS
                a = _simplify_binary_operator(HplBinaryOperator(op, a1, a2))
                # it is now possible to have two literals on the RHS
                b = _simplify_binary_operator(HplBinaryOperator(op, b1, b2))
            if is_self_or_field(b1, deep=True):
                noop = False
                x = b1
                b1 = a2
                a2 = a1
                a1 = x
                # it is now possible to have redundancy on the LHS
                a = _simplify_binary_operator(HplBinaryOperator(op, a1, a2))
                # it is now possible to have two literals on the RHS
                b = _simplify_binary_operator(HplBinaryOperator(op, b1, b2))
        elif left:
            if isinstance(a.operand2, HplLiteral):
                noop = False
                x = b
                b = a.operand2
                # it is now possible to have redundancy on the LHS
                a = _simplify_binary_operator(HplBinaryOperator(op, a.operand1, x))
        elif right:
            if is_self_or_field(b.operand1, deep=True):
                noop = False
                x = a
                a = b.operand1
                # it is now possible to have redundancy on the RHS
                b = _simplify_binary_operator(HplBinaryOperator(op, x, b.operand2))

    return expr if noop else HplBinaryOperator(op, a, b)


def _canonical_lhs(expr: HplBinaryOperator) -> HplBinaryOperator:
    op: BinaryOperatorDefinition = expr.operator
    a: HplExpression = expr.operand1
    b: HplExpression = expr.operand2

    # 1. self-references
    if is_self_or_field(a):
        return expr
    if is_self_or_field(b):
        return HplBinaryOperator(inverse_operator(op), b, a)
    if a.contains_self_reference():
        if op.is_arithmetic or op.is_comparison:
            changed = False
            while isinstance(a, HplUnaryOperator):
                b = HplUnaryOperator(a.operator, b)
                a = a.operand
                op = inverse_operator(op)
                changed = True
            if changed:
                return _canonical_lhs(HplBinaryOperator(op, a, b))
    return expr
