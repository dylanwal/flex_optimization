import numpy as np

import flex_optimization as fo


def func(args) -> tuple:
    mean = fo.problems.nd_gaussian(args, sigma=[2,2])
    std = fo.problems.nd_gaussian(args, sigma=[3,3], center=[2,2])
    return mean, std


def func2(args) -> list:
    mean = fo.problems.nd_gaussian(args, sigma=[2,2])
    std = fo.problems.nd_gaussian(args, sigma=[3,3], center=[2,2])
    return [mean, std]


def func3(args) -> np.ndarray:
    mean = fo.problems.nd_gaussian(args, sigma=[2,2])
    std = fo.problems.nd_gaussian(args, sigma=[3,3], center=[2,2])
    return np.array([mean, std])


def func4(x, y) -> np.ndarray:
    mean = fo.problems.nd_gaussian([x, y], sigma=[2,2])
    std = fo.problems.nd_gaussian([x, y], sigma=[3,3], center=[2,2])
    return np.array([mean, std])


def func5(args) -> tuple:
    mean = fo.problems.nd_gaussian(args, sigma=[2,2])
    std = fo.problems.nd_gaussian(args, sigma=[3,3], center=[2,2])
    return float(mean), float(std)


def metric(args) -> float:
    mean = args[0]
    std = args[1]
    return mean - std


def metric2(mean, std) -> float:
    return mean - std


def metric3(mean, std) -> list:
    return [mean - std]


def metric4(mean, std) -> np.ndarray:
    return np.array([mean - std])


def true_func(args):
    out = func(args)
    return metric(out)


def main():
    problem = fo.Problem(
        func=func,
        variables=[fo.ContinuousVariable(-5, 5, name="x"), fo.ContinuousVariable(-5, 5, name="y")],
        type_=fo.OptimizationType.MAX,
        metric=metric4,
        pass_kwargs=False
    )

    # METHODS
    # method = fo.methods.MethodFactorial(problem=problem, levels=10)
    method = fo.methods.MethodRandom(problem, fo.stop_criteria.StopFunctionEvaluation(100))

    method.run()

    vis = fo.VizOptimization(method.recorder, true_func=true_func)
    fig = vis.plot_3d_vis()
    fig.show()


if __name__ == "__main__":
    main()
