from abc import ABC
from typing import Callable

from flex_optimization import OptimizationType
from flex_optimization.core.variable import Variable, DiscreteVariable, ContinuousVariable


class Problem(ABC):

    def __init__(self,
                 func: Callable,
                 variables: list[Variable] | tuple[Variable],
                 kwargs: dict = None,
                 metric: Callable = None,
                 type_: OptimizationType = OptimizationType.MIN):
        """

        Parameters
        ----------
        func: Callable
            function to be evaluated
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
        self.kwargs = kwargs
        self.variables = variables
        self.type_ = type_

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
        return self._func(*args, **kwargs, **self.kwargs)

    def metric(self, results):
        if self._metric is not None:
            return self._metric(results)

        return results
