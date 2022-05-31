from scipy import optimize

from flex_optimization import OptimizationType, NotSupported
from flex_optimization.core.recorder import Recorder
from flex_optimization.core.utils import save_if_error
from flex_optimization.core.variable import DiscreteVariable
from flex_optimization.core.problem import Problem
from flex_optimization.core.method_subclass import ActiveMethod, StopCriteria
from flex_optimization.stop_criteria import StopIterationEvaluation, StopAbsoluteChange, StopRelativeChange, \
    StopFunctionEvaluation


class MethodDualAnneal(ActiveMethod):
    """
    Method: Dual Annealing

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html?highlight=dual%20anneal#scipy.optimize.dual_annealing

    """

    def __init__(self,
                 problem: Problem,
                 stop_criterion: StopCriteria | list[StopCriteria] | list[list[StopCriteria]],
                 multiprocess: bool | int = False,
                 recorder: Recorder = None,
                 options: dict = None,
                 _method: str = None):

        self.options = options if options is not None else {}
        self._method = _method

        super().__init__(problem, stop_criterion, multiprocess, recorder)

    def method_init(self):
        self._flag_init = True
        for var in self.problem.variables:
            if isinstance(var, DiscreteVariable):
                raise NotSupported("Discrete variables are not currently supported for this algorithm. ")

        # stopping information
        for stop in self.stop_criterion:
            if stop is isinstance(stop, list):
                continue  # compound stop criteria evaluated in flex_optimization not scipy
            if isinstance(stop, StopFunctionEvaluation):
                self.options["maxiter"] = stop.num_eval
            elif isinstance(stop, StopAbsoluteChange):
                self.options["f_tol"] = stop.cut_off_value
            elif isinstance(stop, StopRelativeChange):
                self.options["f_rtol"] = stop.cut_off_value

        # add defaults
        bounds = []
        for var in self.problem.variables:
            bounds.append([var.min_, var.max_])
        self.options["bounds"] = bounds

    @save_if_error
    def run_steps(self, algo_steps: int = 1):
        if not self._flag_init:
            self.method_init()

        if self.problem.type_ == OptimizationType.MAX:
            def maximize_func(*args, **kwargs):
                return -1 * self.problem.evaluate_capture(*args, **kwargs)

            func = maximize_func
        else:
            func = self.problem.evaluate_capture

        result = optimize.dual_annealing(func, callback=self.callback, **self.options)
        self._check_result(result)

    def callback(self, *args, **kwargs):
        self.iteration_count += 1
        for i in range(len(self.problem._temp_data)):
            temp_datapoint = self.problem._temp_data.pop(0)
            temp_datapoint.iteration = self.iteration_count
            super()._tell(temp_datapoint)
        if not self._check_stop_criterion():
            return True

    def _check_result(self, result):
        if result.success:
            return

        self.recorder.record(self.recorder.WARNING, text=f"SciPy: {result.message}")

    def get_point(self) -> list:
        raise NotImplementedError("This algorithm does not support this function. ")
