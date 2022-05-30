
from scipy.stats.qmc import Halton
from scipy.interpolate import interp1d

from flex_optimization.core.recorder import Recorder
from flex_optimization.core.variable import ContinuousVariable, DiscreteVariable
from flex_optimization.core.problem import Problem
from flex_optimization.core.method_subclass import ActiveMethod, StopCriteria
from flex_optimization.methods import MethodType, MethodClassification


def map_number(old_value, old_min, old_max, new_min, new_max) -> float:
    return float(interp1d([old_min, old_max], [new_min, new_max])(old_value))


class MethodHalton(ActiveMethod):

    def __init__(self,
                 problem: Problem,
                 stop_criterion: StopCriteria | list[StopCriteria] | list[list[StopCriteria]],
                 multiprocess: bool | int = False,
                 recorder: Recorder = None):

        self.halton = Halton(d=len(problem.variables))
        super().__init__(problem, stop_criterion, multiprocess, recorder)

    def get_point(self) -> list:
        var = self.halton.random()[0]
        out = []
        for i, v in enumerate(self.problem.variables):
            if isinstance(v, ContinuousVariable):
                out.append(map_number(var[i], 0, 1, v.min_, v.max_))
            elif isinstance(v, DiscreteVariable):
                out.append(v.item[int(map_number(var[i], 0, 1, 0, len(v)-1))])
            else:
                raise NotImplementedError

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
    name="Halton",
    func=MethodHalton,
    type_=MethodType.ACTIVE_SAMPLING
)
