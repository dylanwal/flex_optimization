from typing import Callable

import numpy as np
import pytest

import flex_optimization.problems as fo_p

problems = [i.func for i in fo_p.ProblemClassification.population]


@pytest.mark.parametrize("problem", problems)
def test_list_input(problem: Callable):
    args = [[-5, -2, 0, 1, 3, 5], [-5, -2, 0, 1, 3, 5]]
    result = problem(args)
    assert len(result.shape) == 1
    assert result.shape[0] == 6


@pytest.mark.parametrize("problem", problems)
def test_nparray_input(problem: Callable):
    n = 10
    x = np.linspace(-5, 5, n)
    y = np.linspace(-5, 5, n)
    args = [x, y]
    result = problem(args)
    assert len(result.shape) == 1
    assert result.shape[0] == n


@pytest.mark.parametrize("problem", problems)
def test_nparray_input2(problem: Callable):
    n = 10
    x = np.linspace(-5, 5, n)
    y = np.linspace(-5, 5, n)
    args = np.column_stack((x, y))
    result = problem(args)
    assert len(result.shape) == 1
    assert result.shape[0] == n
