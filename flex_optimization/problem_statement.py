from abc import ABC, abstractmethod
from typing import Callable, Union
import multiprocessing

import pandas as pd

from flex_optimization.method_logger import logger


class Variable(ABC):
    COUNTER = 0

    def __init__(self, name: str = None):
        Variable.COUNTER += 1
        if name is None:
            name = f"var_{self.COUNTER}"
        self.name = name

        logger.info(f"{type(self).__name__} | {repr(self)}")


class DiscreteVariable(Variable):

    def __init__(self, items: Union[list, tuple], name: str = None):
        self.items = items
        super().__init__(name)

    def __len__(self):
        return len(self.items)

    def __repr__(self):
        if len(self) < 5:
            items = self.items
        else:
            items = self.items[:3] + ["..."]
        return f"{self.name} = {len(self)}; {items}"


class ContinuousVariable(Variable):

    def __init__(self, min_: Union[int, float], max_: Union[int, float], type_: type = float, name: str = None):
        self.min_ = min_
        self.max_ = max_
        self.type_ = type_
        super().__init__(name)

    def __repr__(self):
        return f"{self.name} = [{self.min_}, {self.max_}]"


class Problem(ABC):

    def __init__(self,
                 func: Callable,
                 variables: Union[list[Variable], tuple[Variable]],
                 kwargs: dict = None,
                 metric: Callable = None,
                 optimization_type: bool = False):
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
        optimization_type: bool
            0: minimize
            1: maximize

        """
        self._func = func
        self._metric = metric
        self.kwargs = kwargs
        self.variables = variables
        self.optimization_type = optimization_type

        logger.info(f"{type(self).__name__} | {repr(self)}")

    @property
    def variable_names(self) -> list[str]:
        return [var.name for var in self.variables]

    def func(self, *args, **kwargs):
        return self._func(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        result = self._func(*args, **kwargs, **self.kwargs)
        metric = self.metric(result)
        logger.monitor(f"{args} --> {metric}")
        return metric

    def metric(self, results):
        if self._metric is not None:
            return self._metric(results)

        return results


class Method(ABC):
    def __init__(self, problem: Problem, multiprocess: Union[bool, int] = False):
        self.problem = problem
        self.multiprocess = multiprocess
        self._data = multiprocessing.Queue()
        self.data = pd.DataFrame(columns=problem.variable_names+["metric"])
        self.result = None

        logger.info(f"{type(self).__name__} | {repr(self)}")

    @abstractmethod
    def run(self):
        ...

    def _get_pool_size(self):
        if isinstance(self.multiprocess, int):
            return self.multiprocess

        return multiprocessing.cpu_count() - 1

    def get_best_result(self):
        if self.problem.optimization_type:
            best_result_index = self.data["metric"].idxmax()
        else:
            best_result_index = self.data["metric"].idxmin()

        self.result = self.data.iloc[best_result_index].to_dict()

        logger.info(f"\nBest Result: {self.result}")


class PassiveMethod(Method, ABC):
    def __init__(self, problem: Problem, multiprocess: Union[bool, int] = False):
        self.number_points = None
        super().__init__(problem, multiprocess)

    @abstractmethod
    def get_points(self):
        pass

    def run(self):
        points = self.get_points()
        self.number_points = len(points)
        if self.multiprocess > 1:
            with multiprocessing.Pool(self._get_pool_size()) as p:
                output = p.map(self.problem.evaluate, points)
        else:
            output = []
            for point in points:
                output.append(self.problem.evaluate(point))

        for i, name in enumerate(self.problem.variable_names):
            self.data[name] = [p[i] for p in points]
        self.data["metric"] = output

        self.get_best_result()


class StopCriteria:
    def evaluate(self, method: Method, *args, **kwargs) -> bool:
        """ True = Continue; False = Stop """
        pass


class ActiveMethod(Method, ABC):

    def __init__(self,
                 problem: Problem,
                 stop_criterion: Union[StopCriteria, list[StopCriteria], list[list[StopCriteria]]],
                 multiprocess: Union[bool, int] = False):
        if not isinstance(stop_criterion, list):
            stop_criterion = [stop_criterion]
        self.stop_criterion: list[StopCriteria] = stop_criterion
        self.stop_criteria = None
        self._flag_init = False
        super().__init__(problem, multiprocess)

    def inti_method(self):
        pass

    @abstractmethod
    def get_point(self):
        pass

    def run(self, guard_value: int = 1_000):
        for _ in range(guard_value):
            point = self.get_point()
            result = self.problem.evaluate(point)

            # save data
            self.data.loc[0 if pd.isnull(self.data.index.max()) else self.data.index.max() + 1] = point + [result]

            if not self._check_stop_criterion():
                break

        self.get_best_result()

    def _check_stop_criterion(self):
        """ True = Continue; False = Stop """
        for criteria in self.stop_criterion:
            if isinstance(criteria, list):
                if not self._multi_stop_criterion(criteria):
                    self.stop_criteria = criteria
                    logger.info(f"Stop Criteria met:{criteria}")
                    return False

            elif not criteria.evaluate(self):
                self.stop_criteria = criteria
                logger.info(f"Stop Criteria met:{criteria}")
                return False


        return True

    def _multi_stop_criterion(self, criterion):
        stopping_results = []
        for criteria in criterion:
            stopping_results.append(criteria.evaluate(self))

        return any(stopping_results)
