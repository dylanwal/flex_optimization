import enum


class MethodType(enum.Enum):
    PASSIVE_SAMPLING = 0
    ACTIVE_SAMPLING = 1
    ACTIVE_GRADIENT = 2
    ACTIVE_BAYESIAN = 3


class MethodClassification:
    population = []

    def __init__(self, name: str, func: callable, type_: MethodType):
        self.name = name
        self.func = func
        self.type_ = type_

        MethodClassification.population.append(self)

    def __repr__(self):
        return f"{self.name} | {self.type_}"


# passive sampling
from flex_optimization.methods.passive_methods.factorial import MethodFactorial
from flex_optimization.methods.passive_methods.covariance import MethodCovariance
from flex_optimization.methods.passive_methods.multi_covariance import MethodMultiCovariance
from flex_optimization.methods.passive_methods.star import MethodStar

# active sampling
from flex_optimization.methods.active_methods.random_pick import MethodRandom
from flex_optimization.methods.active_methods.sobol import MethodSobol
from flex_optimization.methods.active_methods.latin_hypercube import MethodLatinHypercube
from flex_optimization.methods.active_methods.halton import MethodHalton
from flex_optimization.methods.active_methods.multivariate_normal import MethodMultiNormal

# gradient based
from flex_optimization.methods.active_methods.BFGS import MethodBFGS

# active learning
from flex_optimization.methods.active_methods.baysian_dragon import MethodBODragon

