
import numpy as np
from scipy.optimize import minimize

from flex_optimization import OptimizationType, NotSupported
from flex_optimization.core.recorder import Recorder
from flex_optimization.core.variable import DiscreteVariable
from flex_optimization.core.problem import Problem
from flex_optimization.core.method_subclass import ActiveMethod, StopCriteria
from flex_optimization.methods import MethodType, MethodClassification


class MethodBFGS(ActiveMethod):

    def __init__(self,
                 problem: Problem,
                 x0,
                 stop_criteria: StopCriteria | list[StopCriteria] | list[list[StopCriteria]],
                 multiprocess: bool | int = False,
                 recorder: Recorder = None):

        self.x0 = x0

        super().__init__(problem, stop_criteria, multiprocess, recorder)

    def method_init(self):
        for var in self.problem.variables:
            if isinstance(var, DiscreteVariable):
                raise NotSupported("Discrete variables are not currently supported for this algorithm. ")
        self._flag_init = True

    def run_steps(self, algo_steps: int = 1):
        if not self._flag_init:
            self.method_init()

        try:
            if self.problem.type_ == OptimizationType.MAX:
                def maximize_func(*args, **kwargs):
                    return -1 * self.problem.evaluate_capture(*args, **kwargs)
                func = maximize_func
            else:
                func = self.problem.evaluate_capture

            result = minimize(func, self.x0, method='BFGS', jac='2-point', callback=self.callback)
            self._check_result(result)

        except KeyboardInterrupt as e:
            self.recorder._error_exit(e)

        except Exception as e:
            print(e)
            self.recorder._error_exit(e)

    def callback(self, *args, **kwargs):
        self.iteration_count += 1
        point = []
        result = []
        metric = []
        for i in range(len(self.problem._temp_data)):
            temp = self.problem._temp_data.pop(0)
            point.append(temp[0])
            result.append(temp[1])
            metric.append(temp[2])
        super()._tell(point, result, metric)

    def _check_result(self, result):
        if result.success:
            return

        self.recorder.record(self.recorder.WARNING, result.message)

    def get_point(self) -> list:
        raise NotImplementedError("This algorithm does not support this function. ")


method_class = MethodClassification(
    name="Broyden–Fletcher–Goldfarb–Shanno (BFGS)",
    func=MethodBFGS,
    type_=MethodType.ACTIVE_GRADIENT
)
