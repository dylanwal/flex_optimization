import random

import numpy as np

from flex_optimization.core.data_point import DataPoint
from flex_optimization.core.recorder import Recorder
from flex_optimization.core.variable import ContinuousVariable, DiscreteVariable
from flex_optimization.core.problem import Problem
from flex_optimization.core.method_subclass import ActiveMethod, StopCriteria
from flex_optimization.methods import MethodClassification, MethodType


class MethodRandom(ActiveMethod):

    def __init__(self,
                 problem: Problem,
                 stop_criterion: StopCriteria | list[StopCriteria] | list[list[StopCriteria]],
                 multiprocess: bool | int = False,
                 recorder: Recorder = None):

        super().__init__(problem, stop_criterion, multiprocess, recorder)

    @staticmethod
    def _select_point(var):
        if isinstance(var, ContinuousVariable):
            return np.random.uniform(var.min_, var.max_, 1)[0]
        elif isinstance(var, DiscreteVariable):
            return random.choice(var.items)
        else:
            raise NotImplementedError

    def get_point(self) -> list:
        return [self._select_point(var) for var in self.problem.variables]

    def get_points(self, num: int) -> list[list]:
        """ Get multiple points. """
        return [[self._select_point(var) for var in self.problem.variables] for _ in range(num)]

    def _multi_run_step(self, step: int):
        points = self.get_points(min(step, self._get_steps()))

        def callback(results):
            point_, result = results
            metric = self.problem.metric(result)
            self.recorder.record(self.recorder.EVALUATION, data_point=DataPoint(point_, result, metric))

        from functools import partial
        from flex_optimization.core.utils import PoolHandler
        from flex_optimization.core.method_subclass import temp_func
        pool = PoolHandler(
            func=partial(temp_func, func=self.problem.evaluate),
            pool_size=self._get_pool_size(),
            pool_points=points,
            callback=callback
        )
        pool.run()


method_class = MethodClassification(
    name="random",
    func=MethodRandom,
    type_=MethodType.ACTIVE_SAMPLING
)
