from typing import Union

from scipy.stats.qmc import LatinHypercube
from scipy.interpolate import interp1d


from flex_optimization.problem_statement import ActiveMethod, Problem, StopCriteria, \
    ContinuousVariable, DiscreteVariable


def map_number(old_value, old_min, old_max, new_min, new_max) -> float:
    return float(interp1d([old_min, old_max], [new_min, new_max])(old_value))


class MethodLatinHypercube(ActiveMethod):

    def __init__(self,
             problem: Problem,
             stop_criteria: Union[StopCriteria, list[StopCriteria]],
             multiprocess: Union[bool, int] = False, **kwargs):

        self.hypercube = LatinHypercube(d=len(problem.variables))
        super().__init__(problem, stop_criteria, multiprocess)

    def get_point(self) -> list:
        var = self.hypercube.random()[0]
        out = []
        for i, v in enumerate(self.problem.variables):
            if isinstance(v, ContinuousVariable):
                out.append(map_number(var[i], 0, 1, v.min_, v.max_))
            elif isinstance(v, DiscreteVariable):
                out.append(v.item[int(map_number(var[i], 0, 1, 0, len(v)-1))])
            else:
                raise NotImplementedError

        return out
