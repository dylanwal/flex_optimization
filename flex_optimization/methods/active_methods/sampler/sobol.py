
from scipy.stats.qmc import Sobol

from flex_optimization.core.variable import ContinuousVariable, DiscreteVariable
from flex_optimization.methods import MethodClassification, MethodType
from flex_optimization.methods.active_methods._sampler import MethodSampler
from flex_optimization.methods.active_methods.sampler.utils import map_number


class MethodSobol(MethodSampler):
    """
    Method: Sobol

    """

    def method_init(self):
        self.sampler = Sobol(d=len(self.problem.variables), seed=self.seed)
        self._flag_init = True

    def get_point(self) -> list:
        var = self.sampler.random()[0]
        out = []
        for i, v in enumerate(self.problem.variables):
            if isinstance(v, ContinuousVariable):
                out.append(map_number(var[i], 0, 1, v.min_, v.max_))
            elif isinstance(v, DiscreteVariable):
                out.append(v.item[int(map_number(var[i], 0, 1, 0, len(v)-1))])
            else:
                raise NotImplementedError

        return out


method_class = MethodClassification(
    name="sobol",
    func=MethodSobol,
    type_=MethodType.ACTIVE_SAMPLING
)
