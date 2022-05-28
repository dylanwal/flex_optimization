
import numpy as np
from scipy.optimize import minimize

from flex_optimization.core.recorder import Recorder
from flex_optimization.core.variable import ContinuousVariable, DiscreteVariable
from flex_optimization.core.problem import Problem
from flex_optimization.core.method_subclass import ActiveMethod, StopCriteria


class MethodBFGS(ActiveMethod):

    def __init__(self,
                 problem: Problem,
                 x0,
                 stop_criteria: StopCriteria | list[StopCriteria] | list[list[StopCriteria]],
                 multiprocess: bool | int = False,
                 recorder: Recorder = None):

        self.x0 = x0

        super().__init__(problem, stop_criteria, multiprocess, recorder)

    def run(self, guard_value: int = 1_000):
        if self.problem.optimization_type:
            def maximize_func(*args, **kwargs):
                return -1 * self.problem.evaluate(*args, **kwargs)
            func = maximize_func
        else:
            func = self.problem.evaluate

        r = minimize(func, self.x0, method='BFGS', jac='2-point', callback=self.callback)
        print(r)

    def callback(self, *args, **kwargs):
        print(args)

    def get_point(self) -> list:
        pass


    def main_loop(self):
        pass