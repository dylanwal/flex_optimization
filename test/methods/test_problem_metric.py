import pytest

import numpy as np

import flex_optimization as fo


def test_args():
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


def test_kwargs():
    def func(args) -> tuple:
        mean = fo.problems.nd_gaussian(args, sigma=[2,2])
        std = fo.problems.nd_gaussian(args, sigma=[3,3], center=[2,2])
        return mean, std

    def metric(mean, std) -> float:
        return mean - std

    problem = fo.Problem(
        func=func,
        variables=[fo.ContinuousVariable(-5, 5, name="x"), fo.ContinuousVariable(-5, 5, name="y")],
        type_=fo.OptimizationType.MAX,
        metric=metric,
    )

    method = fo.methods.MethodFactorial(problem=problem, levels=10)

    method.run()


def test_return_list():
    def func(args) -> tuple:
        mean = fo.problems.nd_gaussian(args, sigma=[2,2])
        std = fo.problems.nd_gaussian(args, sigma=[3,3], center=[2,2])
        return mean, std

    def metric(mean, std) -> list:
        return [mean - std]

    problem = fo.Problem(
        func=func,
        variables=[fo.ContinuousVariable(-5, 5, name="x"), fo.ContinuousVariable(-5, 5, name="y")],
        type_=fo.OptimizationType.MAX,
        metric=metric,
    )

    method = fo.methods.MethodFactorial(problem=problem, levels=10)

    method.run()


def test_nparray_list():
    def func(args) -> tuple:
        mean = fo.problems.nd_gaussian(args, sigma=[2,2])
        std = fo.problems.nd_gaussian(args, sigma=[3,3], center=[2,2])
        return mean, std

    def metric(mean, std) -> np.ndarray:
        return np.array([mean - std])

    problem = fo.Problem(
        func=func,
        variables=[fo.ContinuousVariable(-5, 5, name="x"), fo.ContinuousVariable(-5, 5, name="y")],
        type_=fo.OptimizationType.MAX,
        metric=metric,
    )

    method = fo.methods.MethodFactorial(problem=problem, levels=10)

    method.run()