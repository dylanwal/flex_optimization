from abc import ABC
import random

import numpy as np
from numpy.random import normal, laplace, logistic, gumbel
import scipy
from scipy.optimize import minimize

from flex_optimization import OptimizationType, NotSupported
from flex_optimization.core.recorder import Recorder
from flex_optimization.core.utils import save_if_error
from flex_optimization.core.variable import DiscreteVariable
from flex_optimization.core.problem import Problem
from flex_optimization.core.method_subclass import ActiveMethod, StopCriteria
from flex_optimization.stop_criteria import StopIterationEvaluation, StopAbsoluteChange

dist_dict = {
    "normal": normal,
    "laplace": laplace,
    "logistic": logistic,
    "gumbel": gumbel,
}


def random_restart(func):
    def wrapper(self, *args, **kwargs):
        if self.rand_rest_p > random.uniform(0, 1):
            return self.move_random()
        else:
            return func(self, *args, **kwargs)

    return wrapper


def move_random(ss_positions):
    position = []
    for search_space_pos in ss_positions:
        pos_ = random.choice(search_space_pos)
        position.append(pos_)

    return np.array(position)


def _move_climb(position, speed = 0.03, distribution: str = "normal"):
    sigma = max_positions * speed
    pos_normal = dist_dict[distribution](position, sigma, len(position))

    return conv2pos(pos_normal)


def conv2pos(pos):
    # position to int
    r_pos = np.rint(pos)

    n_zeros = [0] * len(max_positions)
    # clip into search space boundaries
    pos = np.clip(r_pos, n_zeros, max_positions).astype(int)

    dist = scipy.spatial.distance.cdist(r_pos.reshape(1, -1), pos.reshape(1, -1))
    threshold = self.conv.search_space_size / (100 ** n_dimensions)

    if dist > threshold:
        return move_random(search_space_positions)

    return pos


class MethodHillClimber(ActiveMethod, ABC):
    """
    Method: Hill Climber
    """

    def __init__(self,
                 problem: Problem,
                 stop_criterion: StopCriteria | list[StopCriteria] | list[list[StopCriteria]],
                 x0: list | tuple | np.ndarray,
                 multiprocess: bool | int = False,
                 recorder: Recorder = None,
                 options: dict = None,
                 _method: str = None):

        self.x0 = x0
        self.n_neighbours = n_neighbours # 3
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
            if isinstance(stop, StopIterationEvaluation):
                if 'options' in self.options:
                    self.options["options"] = self.options["options"] | {"maxiter": stop.num_eval}
                else:
                    self.options["options"] = {"maxiter": stop.num_eval}

            elif isinstance(stop, StopAbsoluteChange):
                self.options["tol"] = stop.cut_off_value

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

        result = minimize(func, self.x0, method=self._method, callback=self.callback, **self.options)
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

        self.recorder.record(self.recorder.WARNING, text="SciPy: " + result.message)

    def get_point(self) -> list:
        raise NotImplementedError("This algorithm does not support this function. ")
