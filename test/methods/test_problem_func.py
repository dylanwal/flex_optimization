import pytest

import flex_optimization as fo


def test_factoral():
    problem = fo.Problem(
        func=fo.problems.nd_gaussian,
        variables=[fo.ContinuousVariable(-5, 5, name="x"), fo.ContinuousVariable(-5, 5, name="y")],
        kwargs=dict(center=[0.2342, 0.1234], sigma=[1, 3]),
        type_=fo.OptimizationType.MAX
    )
    method = fo.methods.MethodFactorial(problem, levels=3)
    method.run()


def test_covariance():
    problem = fo.Problem(
        func=fo.problems.nd_gaussian,
        variables=[fo.ContinuousVariable(-5, 5, name="x"), fo.ContinuousVariable(-5, 5, name="y")],
        kwargs=dict(center=[0.2342, 0.1234], sigma=[1, 3]),
        type_=fo.OptimizationType.MAX
    )
    method = fo.methods.MethodCovariance(problem, levels=3)
    method.run()


def test_multicovariance():
    problem = fo.Problem(
        func=fo.problems.nd_gaussian,
        variables=[fo.ContinuousVariable(-5, 5, name="x"), fo.ContinuousVariable(-5, 5, name="y")],
        kwargs=dict(center=[0.2342, 0.1234], sigma=[1, 3]),
        type_=fo.OptimizationType.MAX
    )
    method = fo.methods.MethodMultiCovariance(problem, levels=3)
    method.run()


def test_star():
    problem = fo.Problem(
        func=fo.problems.nd_gaussian,
        variables=[fo.ContinuousVariable(-5, 5, name="x"), fo.ContinuousVariable(-5, 5, name="y")],
        kwargs=dict(center=[0.2342, 0.1234], sigma=[1, 3]),
        type_=fo.OptimizationType.MAX
    )
    method = fo.methods.MethodStar(problem, levels=3)
    method.run()


def test_random():
    problem = fo.Problem(
        func=fo.problems.nd_gaussian,
        variables=[fo.ContinuousVariable(-5, 5, name="x"), fo.ContinuousVariable(-5, 5, name="y")],
        kwargs=dict(center=[0.2342, 0.1234], sigma=[1, 3]),
        type_=fo.OptimizationType.MAX
    )
    method = fo.methods.MethodRandom(problem, fo.stop_criteria.StopFunctionEvaluation(20))
    method.run()


def test_sobol():
    problem = fo.Problem(
        func=fo.problems.nd_gaussian,
        variables=[fo.ContinuousVariable(-5, 5, name="x"), fo.ContinuousVariable(-5, 5, name="y")],
        kwargs=dict(center=[0.2342, 0.1234], sigma=[1, 3]),
        type_=fo.OptimizationType.MAX
    )
    method = fo.methods.MethodSobol(problem, fo.stop_criteria.StopFunctionEvaluation(20))
    method.run()


def test_latinhypercube():
    problem = fo.Problem(
        func=fo.problems.nd_gaussian,
        variables=[fo.ContinuousVariable(-5, 5, name="x"), fo.ContinuousVariable(-5, 5, name="y")],
        kwargs=dict(center=[0.2342, 0.1234], sigma=[1, 3]),
        type_=fo.OptimizationType.MAX
    )
    method = fo.methods.MethodLatinHypercube(problem, fo.stop_criteria.StopFunctionEvaluation(20))
    method.run()


def test_halton():
    problem = fo.Problem(
        func=fo.problems.nd_gaussian,
        variables=[fo.ContinuousVariable(-5, 5, name="x"), fo.ContinuousVariable(-5, 5, name="y")],
        kwargs=dict(center=[0.2342, 0.1234], sigma=[1, 3]),
        type_=fo.OptimizationType.MAX
    )
    method = fo.methods.MethodHalton(problem, fo.stop_criteria.StopFunctionEvaluation(20))
    method.run()


def test_multinormal():
    problem = fo.Problem(
        func=fo.problems.nd_gaussian,
        variables=[fo.ContinuousVariable(-5, 5, name="x"), fo.ContinuousVariable(-5, 5, name="y")],
        kwargs=dict(center=[0.2342, 0.1234], sigma=[1, 3]),
        type_=fo.OptimizationType.MAX
    )
    method = fo.methods.MethodMultiNormal(problem, fo.stop_criteria.StopFunctionEvaluation(20))
    method.run()


def test_BODragon():
    problem = fo.Problem(
        func=fo.problems.nd_gaussian,
        variables=[fo.ContinuousVariable(-5, 5, name="x"), fo.ContinuousVariable(-5, 5, name="y")],
        kwargs=dict(center=[0.2342, 0.1234], sigma=[1, 3]),
        type_=fo.OptimizationType.MAX
    )
    method = fo.methods.MethodBODragon(problem, fo.stop_criteria.StopFunctionEvaluation(10))
    method.run()


def test_BFGS():
    problem = fo.Problem(
        func=fo.problems.nd_gaussian,
        variables=[fo.ContinuousVariable(-5, 5, name="x"), fo.ContinuousVariable(-5, 5, name="y")],
        kwargs=dict(center=[0.2342, 0.1234], sigma=[1, 3]),
        type_=fo.OptimizationType.MAX
    )
    method = fo.methods.MethodBFGS(problem, stop_criteria=fo.stop_criteria.StopFunctionEvaluation(20), x0=[1, 1])
    method.run()

