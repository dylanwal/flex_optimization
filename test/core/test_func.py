import pytest

import numpy as np

import flex_optimization as fo


def test_tuple_return():
    def func(args) -> tuple:
        mean = fo.problems.nd_gaussian(args, sigma=[2,2])
        std = fo.problems.nd_gaussian(args, sigma=[3,3], center=[2,2])
        return mean, std

    def metric(args) -> float:
        mean = args[0]
        std = args[1]
        return mean - std

    problem = fo.Problem(
        func=func,
        variables=[fo.ContinuousVariable(-5, 5, name="x"), fo.ContinuousVariable(-5, 5, name="y")],
        type_=fo.OptimizationType.MAX,
        metric=metric,
    )

    method = fo.methods.MethodFactorial(problem=problem, levels=10)

    method.run()


def test_kwargs_return():
    def func(x, y) -> np.ndarray:
        mean = fo.problems.nd_gaussian([x, y], sigma=[2,2])
        std = fo.problems.nd_gaussian([x, y], sigma=[3,3], center=[2,2])
        return np.array([mean, std])

    def metric(args) -> float:
        mean = args[0]
        std = args[1]
        return mean - std

    problem = fo.Problem(
        func=func,
        variables=[fo.ContinuousVariable(-5, 5, name="x"), fo.ContinuousVariable(-5, 5, name="y")],
        type_=fo.OptimizationType.MAX,
        metric=metric,
        pass_kwargs=True
    )

    method = fo.methods.MethodFactorial(problem=problem, levels=10)

    method.run()


def test_list_return():
    def func(args) -> list:
        mean = fo.problems.nd_gaussian(args, sigma=[2,2])
        std = fo.problems.nd_gaussian(args, sigma=[3,3], center=[2,2])
        return [mean, std]

    def metric(args) -> float:
        mean = args[0]
        std = args[1]
        return mean - std

    problem = fo.Problem(
        func=func,
        variables=[fo.ContinuousVariable(-5, 5, name="x"), fo.ContinuousVariable(-5, 5, name="y")],
        type_=fo.OptimizationType.MAX,
        metric=metric,
    )

    method = fo.methods.MethodFactorial(problem=problem, levels=10)

    method.run()


def test_nparray_return():
    def func(args) -> np.ndarray:
        mean = fo.problems.nd_gaussian(args, sigma=[2,2])
        std = fo.problems.nd_gaussian(args, sigma=[3,3], center=[2,2])
        return np.array([mean, std])

    def metric(args) -> float:
        mean = args[0]
        std = args[1]
        return mean - std

    problem = fo.Problem(
        func=func,
        variables=[fo.ContinuousVariable(-5, 5, name="x"), fo.ContinuousVariable(-5, 5, name="y")],
        type_=fo.OptimizationType.MAX,
        metric=metric,
    )

    method = fo.methods.MethodFactorial(problem=problem, levels=10)

    method.run()
