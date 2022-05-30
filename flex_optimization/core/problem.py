from abc import ABC
from typing import Callable

from flex_optimization import OptimizationType
from flex_optimization.core.data_point import DataPoint
from flex_optimization.core.variable import Variable, DiscreteVariable, ContinuousVariable


class Problem(ABC):

    def __init__(self,
                 func: Callable,
                 variables: list[Variable] | tuple[Variable],
                 kwargs: dict = None,
                 metric: Callable = None,
                 type_: OptimizationType = OptimizationType.MIN,
                 pass_kwargs: bool = False):
        """

        Parameters
        ----------
        func: Callable
            function to be evaluated
            must return int, float, list[int|float], tuple[int, float]
        variables: list[Variable] | tuple[Variable]
            variables
        kwargs: dict
            additional keyword arguments to be passed to function
        metric: Callable
            calculation of optimization metrics from function output
        type_: OptimizationType
            type of optimization

        """
        self._func = func
        self._metric = metric
        self.kwargs = kwargs if kwargs is not None else {}
        self.variables = variables
        self.type_ = type_
        self.pass_kwargs = pass_kwargs
        self._temp_data: list[DataPoint] = []

    def __repr__(self):
        return f"Find the {self.type_.name} of '{self.func.__name__}' with {self.num_variables} variables"

    @property
    def num_variables(self):
        return len(self.variables)

    @property
    def num_variables_discrete(self):
        return len([True for var in self.variables if isinstance(var, DiscreteVariable)])

    @property
    def num_variables_continuous(self):
        return len([True for var in self.variables if isinstance(var, ContinuousVariable)])

    @property
    def variable_names(self) -> list[str]:
        return [var.name for var in self.variables]

    def func(self, *args, **kwargs):
        return self._func(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        if self.pass_kwargs:
            kwargs = kwargs | self._args_to_kwargs(*args)
            args = ()
        kwargs_ = self.kwargs | kwargs
        return self.func(*args, **kwargs_)

    def _args_to_kwargs(self, args) -> dict:
        out = {}
        for arg, key in zip(args, self.variable_names):
            out[key] = arg
        return out

    # def evaluate_multi(self, points: list | list[list], **kwargs) -> list:
    #     result = []
    #     for point in points:
    #         result = self.func(point, **kwargs, **self.kwargs)
    #     return result

    def evaluate_capture(self, *args, **kwargs):
        # TODO: added because scipy doesn't allow intermittent values, fix scipy callback
        result = self.evaluate(*args, **kwargs, **self.kwargs)  # _func
        metric = self.metric(result)
        self._temp_data.append(DataPoint(*args, result, metric))
        return metric

    def evaluate_multiprocessing(self, points, processes: int):
        """ For mini-batches only. """
        import multiprocessing
        with multiprocessing.Pool(processes) as p:
            result = p.map(self.evaluate, points)
        return result

    def metric(self, result):
        if self._metric is not None:
            try:
                return self._metric(result)
            except TypeError:
                return self._metric(*result)

        return result
