
from scipy.stats.qmc import MultivariateNormalQMC
from scipy.interpolate import interp1d

from flex_optimization.core.data_point import DataPoint
from flex_optimization.core.recorder import Recorder
from flex_optimization.core.variable import ContinuousVariable, DiscreteVariable
from flex_optimization.core.problem import Problem
from flex_optimization.core.method_subclass import ActiveMethod, StopCriteria
from flex_optimization.methods import MethodType, MethodClassification


def map_number(old_value, old_min, old_max, new_min, new_max) -> float:
    if old_value < old_min:
        return new_min
    if old_value > old_max:
        return new_max

    old_span = old_max - old_min
    new_span = new_max - new_min
    scaled_value = float(old_value - old_min) / float(old_span)
    return new_min + (scaled_value * new_span)


class MethodMultiNormal(ActiveMethod):

    def __init__(self,
                 problem: Problem,
                 stop_criterion: StopCriteria | list[StopCriteria] | list[list[StopCriteria]],
                 multiprocess: bool | int = False,
                 recorder: Recorder = None):

        self.sobol = MultivariateNormalQMC(mean=[0]*len(problem.variables))
        super().__init__(problem, stop_criterion, multiprocess, recorder)

    @staticmethod
    def _re_map(var, value):
        if isinstance(var, ContinuousVariable):
            return map_number(value, -3, 3, var.min_, var.max_)
        elif isinstance(var, DiscreteVariable):
            return var.item[int(map_number(value, -3, 3, 0, len(var)-1))]
        else:
            raise NotImplementedError

    def get_point(self) -> list:
        var = self.sobol.random()[0]
        out = []
        for i, v in enumerate(self.problem.variables):
            out.append(self._re_map(v, var[i]))

        return out

    def get_points(self, num: int) -> list[list]:
        """ Get multiple points. """
        return [self.get_point() for _ in range(num)]

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
    name="multi-normal",
    func=MethodMultiNormal,
    type_=MethodType.ACTIVE_SAMPLING
)
