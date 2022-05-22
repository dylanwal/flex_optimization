from scipy.stats.qmc import Sobol

from flex_optimization.problem_statement import ActiveMethod, Problem


class MethodSobol(ActiveMethod):

    def __init__(self,
             problem: Problem,
             stop_criteria: Union[StopCriteria, list[StopCriteria]],
             multiprocess: Union[bool, int] = False):

        super().__init__(problem, stop_criteria, multiprocess)

    def get_point(self):
