import random

import numpy as np

from flex_optimization.core.variable import ContinuousVariable, DiscreteVariable
from flex_optimization.methods import MethodClassification, MethodType
from flex_optimization.methods.active_methods._sampler import MethodSampler
from flex_optimization.methods.active_methods.sampler.utils import map_number


class MethodRandom(MethodSampler):
    """
    Method: Random Number

    """

    def method_init(self):
        self.sampler = np.random.default_rng(seed=self.seed)
        self._flag_init = True

    def _select_point(self, var):
        if isinstance(var, ContinuousVariable):
            return map_number(self.sampler.random(), 0, 1, var.min_, var.max_)
        elif isinstance(var, DiscreteVariable):
            return random.choice(var.items)
        else:
            raise NotImplementedError

    def get_point(self) -> list:
        return [self._select_point(var) for var in self.problem.variables]


method_class = MethodClassification(
    name="random",
    func=MethodRandom,
    type_=MethodType.ACTIVE_SAMPLING
)
