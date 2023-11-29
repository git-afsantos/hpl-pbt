# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

###############################################################################
# Imports
###############################################################################

from math import isnan

from hplpbt.strategies._calc import (
    add,
    INFINITY,
    inverse,
    MINUS_INFINITY,
    NumberLiteral,
    Product,
    Sum,
    Symbol,
)

################################################################################
# Test Functions
################################################################################


def test_add_literals():
    a = NumberLiteral(0)
    b = NumberLiteral(1)
    c = add(a, b)
    assert c.is_literal
    assert isinstance(c, NumberLiteral)
    assert c.value == 1

    a = NumberLiteral(-1)
    b = NumberLiteral(1)
    c = add(a, b)
    assert c.is_literal
    assert isinstance(c, NumberLiteral)
    assert c.value == 0

    a = NumberLiteral(1)
    b = NumberLiteral(1)
    c = add(a, b)
    assert c.is_literal
    assert isinstance(c, NumberLiteral)
    assert c.value == 2

    a = NumberLiteral(-1)
    b = NumberLiteral(-1)
    c = add(a, b)
    assert c.is_literal
    assert isinstance(c, NumberLiteral)
    assert c.value == -2

    a = INFINITY
    b = NumberLiteral(-1)
    c = add(a, b)
    assert c.is_literal
    assert isinstance(c, NumberLiteral)
    assert c == INFINITY

    a = INFINITY
    b = MINUS_INFINITY
    c = add(a, b)
    assert c.is_literal
    assert isinstance(c, NumberLiteral)
    assert isnan(c.value)


def test_solve_symbol():
    x = Symbol('x')
    y = x.solve(x=Symbol('y', minus_sign=True))
    assert y.is_symbol
    assert isinstance(y, Symbol)
    assert y.name == 'y'
    assert y.minus_sign


def test_solve_sum():
    # identity element
    x = Symbol('x')
    c = NumberLiteral(0)
    result = Sum((x, c)).solve()
    assert result.is_symbol
    assert isinstance(result, Symbol)
    assert result is x

    # substitution of one part
    x = Symbol('x')
    y = Symbol('y')
    c = NumberLiteral(40)
    result = Sum((x, y, c)).solve(y=NumberLiteral(2))
    assert result.is_sum
    assert isinstance(result, Sum)
    assert len(result.parts) == 2
    assert result.parts[0] == x
    assert result.parts[1].is_literal
    assert isinstance(result.parts[1], NumberLiteral)
    assert result.parts[1].value == 42

    # substitution of all parts
    x = Symbol('x')
    y = Symbol('y')
    c = NumberLiteral(40)
    result = Sum((x, y, c)).solve(x=y, y=NumberLiteral(1))
    assert result.is_literal
    assert isinstance(result, NumberLiteral)
    assert result.value == 42

    # cancellation of literals
    x = Symbol('x')
    y = Symbol('y')
    c = NumberLiteral(42)
    result = Sum((x, y, c)).solve(y=NumberLiteral(-42))
    assert result.is_symbol
    assert isinstance(result, Symbol)
    assert result == x

    # cancellation of symbols
    x = Symbol('x')
    y = Symbol('y')
    c = NumberLiteral(42)
    result = Sum((x, y, c)).solve(x=Symbol('y', minus_sign=True))
    assert result.is_literal
    assert isinstance(result, NumberLiteral)
    assert result == c


def test_solve_product():
    # identity element
    x = Symbol('x')
    c = NumberLiteral(1)
    result = Product((x, c)).solve()
    assert result.is_symbol
    assert isinstance(result, Symbol)
    assert result is x

    # absorbing element
    x = Symbol('x')
    c = NumberLiteral(0)
    result = Product((x, c)).solve()
    assert result.is_literal
    assert isinstance(result, NumberLiteral)
    assert result is c

    # substitution of one part
    x = Symbol('x')
    y = Symbol('y')
    c = NumberLiteral(21)
    result = Product((x, y, c)).solve(y=NumberLiteral(2))
    assert result.is_product
    assert isinstance(result, Product)
    assert len(result.factors) == 2
    assert result.factors[0] == x
    assert result.factors[1].is_literal
    assert isinstance(result.factors[1], NumberLiteral)
    assert result.factors[1].value == 42

    # substitution of all parts
    x = Symbol('x')
    y = Symbol('y')
    c = NumberLiteral(42)
    result = Product((x, y, c)).solve(x=y, y=NumberLiteral(1))
    assert result.is_literal
    assert isinstance(result, NumberLiteral)
    assert result.value == 42

    # cancellation of literals
    x = Symbol('x')
    y = Symbol('y')
    c = NumberLiteral(2)
    result = Product((x, y, c)).solve(y=NumberLiteral(1/2))
    assert result.is_symbol
    assert isinstance(result, Symbol)
    assert result == x

    # cancellation of symbols
    x = Symbol('x')
    y = Symbol('y')
    c = NumberLiteral(42)
    result = Product((x, y, c)).solve(x=inverse(Symbol('y')))
    assert result.is_literal
    assert isinstance(result, NumberLiteral)
    assert result == c
