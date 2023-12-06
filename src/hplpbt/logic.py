# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

###############################################################################
# Imports
###############################################################################

from typing import Container, Iterable, List, Tuple, TypeVar

from attrs import field, frozen

from hpl.ast import (
    And,
    Exists,
    HplBinaryOperator,
    HplEvent,
    HplEventDisjunction,
    HplExpression,
    HplPattern,
    HplPredicate,
    HplProperty,
    HplQuantifier,
    HplSimpleEvent,
    HplUnaryOperator,
    Not,
    Or,
)
from hpl.rewrite import empty_test, is_and, is_false, is_iff, is_not, is_or, is_true, simplify
from typeguard import typechecked

###############################################################################
# Interface
###############################################################################


@frozen
class SystemAction:
    name: str
    channel: str
    guard: HplPredicate


@frozen
class StateMachine:
    inputs: Iterable[SystemAction] = field(factory=tuple, converter=tuple)
    outputs: Iterable[SystemAction] = field(factory=tuple, converter=tuple)


@typechecked
def split_assumptions(
    hpl_properties: Iterable[HplProperty],
    input_channels: Container[str],
) -> Tuple[List[HplProperty], List[HplProperty]]:
    """
    Given a list of properties and a collection of input channels,
    returns the list of assumptions and the list of behaviour specifications,
    calculated from the original list of properties.
    """
    assumptions = []
    behaviour = []
    for hpl_property in hpl_properties:
        assert isinstance(hpl_property, HplProperty)
        if hpl_property.pattern.is_absence:
            a, b = _split_safety(hpl_property, input_channels)
        elif hpl_property.pattern.is_existence:
            a, b = _split_liveness(hpl_property, input_channels)
        elif hpl_property.pattern.is_requirement:
            a, b = _split_safety(hpl_property, input_channels)
        elif hpl_property.pattern.is_response:
            a, b = _split_liveness(hpl_property, input_channels)
        elif hpl_property.pattern.is_prevention:
            a, b = _split_safety(hpl_property, input_channels)
        else:
            raise TypeError(f'unknown HPL property type: {hpl_property!r}')
        assumptions.extend(a)
        behaviour.extend(b)
    return assumptions, behaviour


@typechecked
def check_atomic_conditions_in_canonical_form(predicate: HplPredicate):
    for phi in _atomic_conditions(predicate):
        if isinstance(phi, HplBinaryOperator):
            if not phi.operator.is_comparison:
                continue
            x: HplExpression = phi.operand1
            if x.is_operator or x.is_quantifier:
                raise ValueError(f'unable to handle non-canonical expression: {phi}')


###############################################################################
# Helper Functions
###############################################################################


def _split_safety(
    hpl_property: HplProperty,
    input_channels: Container[str],
) -> Tuple[List[HplProperty], List[HplProperty]]:
    # optimization: avoid creating new objects if the event is simple
    assert hpl_property.pattern.behaviour.is_simple_event
    if hpl_property.pattern.behaviour.name in input_channels:
        return [hpl_property], []
    else:
        return [], [hpl_property]


def _split_liveness(
    hpl_property: HplProperty,
    input_channels: Container[str],
) -> Tuple[List[HplProperty], List[HplProperty]]:
    inputs: List[HplSimpleEvent] = []
    outputs: List[HplSimpleEvent] = []
    for b in hpl_property.pattern.behaviour.simple_events():
        assert isinstance(b, HplSimpleEvent)
        if b.name in input_channels:
            inputs.append(b)
        else:
            outputs.append(b)
    if not outputs:
        # all behaviour events are inputs
        return [hpl_property], []
    if not inputs:
        # all behaviour events are outputs
        return [], [hpl_property]
    # mixed input and output events
    assert inputs and outputs
    # for the purposes of testing, avoid inputs and force the outputs
    # recreate the original property, but expecting only outputs
    new_pattern = hpl_property.pattern.but(behaviour=_recreate_events(outputs))
    hpl_property = hpl_property.but(pattern=new_pattern)
    # recreate an opposite of the original property, affecting only inputs
    if hpl_property.pattern.is_existence:
        new_pattern = HplPattern.absence(
            _recreate_events(inputs),
            min_time=hpl_property.pattern.min_time,
            max_time=hpl_property.pattern.max_time,
        )
    else:
        assert hpl_property.pattern.is_response
        new_pattern = HplPattern.prevention(
            hpl_property.pattern.trigger,
            _recreate_events(inputs),
            min_time=hpl_property.pattern.min_time,
            max_time=hpl_property.pattern.max_time,
        )
    assumption = hpl_property.but(pattern=new_pattern)
    return [assumption], [hpl_property]


def _recreate_events(events: List[HplSimpleEvent]) -> HplEvent:
    assert events
    result = events[-1]
    for i in range(len(events) - 1):
        result = HplEventDisjunction(events[i], result)
    return result


def _atomic_conditions(predicate: HplPredicate) -> List[HplExpression]:
    conditions = []
    stack = [simplify(predicate.condition)]
    while stack:
        phi: HplExpression = stack.pop()
        if isinstance(phi, HplUnaryOperator):
            if phi.operator.is_not:
                stack.append(phi.operand)
            else:
                conditions.append(phi)
        elif isinstance(phi, HplBinaryOperator):
            op = phi.operator
            if phi.operator.is_comparison or phi.operator.is_inclusion:
                conditions.append(phi)
            else:
                assert not phi.operator.is_arithmetic
                stack.append(phi.operand1)
                stack.append(phi.operand2)
        else:
            conditions.append(phi)
    return conditions


###############################################################################
# HPL Extension
###############################################################################


P = TypeVar('P', HplPredicate, HplExpression)


@typechecked
def split_or(predicate_or_expression: P) -> List[HplExpression]:
    if predicate_or_expression.is_predicate:
        assert isinstance(predicate_or_expression, HplPredicate)
        return _split_or_expr(predicate_or_expression.condition)
    assert isinstance(predicate_or_expression, HplExpression)
    return _split_or_expr(predicate_or_expression)


@typechecked
def _split_or_expr(phi: HplExpression) -> List[HplExpression]:
    conditions: List[HplExpression] = []
    stack: List[HplExpression] = [phi]
    while stack:
        expr: HplExpression = stack.pop()
        # preprocessing
        if is_false(expr):
            continue
        if is_true(expr):
            return [expr]  # tautology
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
        # ~(a <-> b)
        # ==  ~((~a | b) & (~b | a))
        # ==  ~(~a | b) | ~(~b | a)
        # == (a & ~b) | (b & ~a)
        p = And(phi.a, Not(phi.b))
        q = And(phi.b, Not(phi.a))
        return Or(p, q)
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
                qa = And(Not(empty_test(quant.domain)), phi.a)
            if phi.b.contains_reference(var):
                qb = Exists(var, quant.domain, phi.b)
            else:
                qb = And(Not(empty_test(quant.domain)), phi.b)
            return Or(qa, qb)
    elif quant.is_universal:
        pass  # not worth splitting
    return quant  # nothing to do
