
from scipy.stats.qmc import MultivariateNormalQMC

from flex_optimization.core.variable import ContinuousVariable, DiscreteVariable
from flex_optimization.methods import MethodType, MethodClassification
from flex_optimization.methods.active_methods._sampler import MethodSampler
from flex_optimization.methods.active_methods.sampler.utils import map_number


class MethodMultiNormal(MethodSampler):
    """
    Method: MultiNormal

    """

    def method_init(self):
        self.sampler = MultivariateNormalQMC(mean=[0]*len(self.problem.variables), seed=self.seed)
        self._flag_init = True

    @staticmethod
    def _re_map(var, value):
        if isinstance(var, ContinuousVariable):
            return map_number(value, -3, 3, var.min_, var.max_)
        elif isinstance(var, DiscreteVariable):
            return var.item[int(map_number(value, -3, 3, 0, len(var)-1))]
        else:
            raise NotImplementedError

    def get_point(self) -> list:
        var = self.sampler.random()[0]
        out = []
        for i, v in enumerate(self.problem.variables):
            out.append(self._re_map(v, var[i]))

        return out


method_class = MethodClassification(
    name="multi-normal",
    func=MethodMultiNormal,
    type_=MethodType.ACTIVE_SAMPLING
)
