# Methods

----
----

This directory contains collection of methods for optimization.

----

# factorial

|key | value|
|---|---|
|name | factorial| 
|func | <class 'flex_optimization.methods.passive_methods.factorial.MethodFactorial'>| 
|type_ | MethodType.PASSIVE_SAMPLING| 

    Method: Factorial

    Creates a full factorial design.
    Every possible combination of factors

    Parameters
    ----------
    levels: int
        number of levels for each variable

    


----

# covariance

|key | value|
|---|---|
|name | covariance| 
|func | <class 'flex_optimization.methods.passive_methods.covariance.MethodCovariance'>| 
|type_ | MethodType.PASSIVE_SAMPLING| 

    Method: Covariance

    Method picks points by varying one or more factors linearly across the domain.


    


----

# factorial

|key | value|
|---|---|
|name | factorial| 
|func | <class 'flex_optimization.methods.passive_methods.multi_covariance.MethodMultiCovariance'>| 
|type_ | MethodType.PASSIVE_SAMPLING| 
None


----

# star

|key | value|
|---|---|
|name | star| 
|func | <class 'flex_optimization.methods.passive_methods.star.MethodStar'>| 
|type_ | MethodType.PASSIVE_SAMPLING| 

    Method: Star

    The Star algorithm chooses points on a star like pattern.

    


----

# random

|key | value|
|---|---|
|name | random| 
|func | <class 'flex_optimization.methods.active_methods.random_pick.MethodRandom'>| 
|type_ | MethodType.ACTIVE_SAMPLING| 
None


----

# sobol

|key | value|
|---|---|
|name | sobol| 
|func | <class 'flex_optimization.methods.active_methods.sobol.MethodSobol'>| 
|type_ | MethodType.ACTIVE_SAMPLING| 
None


----

# Latin hypercube

|key | value|
|---|---|
|name | Latin hypercube| 
|func | <class 'flex_optimization.methods.active_methods.latin_hypercube.MethodLatinHypercube'>| 
|type_ | MethodType.ACTIVE_SAMPLING| 

    Method: Latin Hypercube

    The latin hypercube  picks points by dividing the sample space into equally size intervals, and selecting points
    from the intervals.

    Parameters
    ----------
    seed: int
        seed

    


----

# Halton

|key | value|
|---|---|
|name | Halton| 
|func | <class 'flex_optimization.methods.active_methods.halton.MethodHalton'>| 
|type_ | MethodType.ACTIVE_SAMPLING| 
None


----

# multi-normal

|key | value|
|---|---|
|name | multi-normal| 
|func | <class 'flex_optimization.methods.active_methods.multivariate_normal.MethodMultiNormal'>| 
|type_ | MethodType.ACTIVE_SAMPLING| 
None


----

# Broyden–Fletcher–Goldfarb–Shanno (BFGS)

|key | value|
|---|---|
|name | Broyden–Fletcher–Goldfarb–Shanno (BFGS)| 
|func | <class 'flex_optimization.methods.active_methods.scipy.BFGS.MethodBFGS'>| 
|type_ | MethodType.ACTIVE_GRADIENT| 

    Method: Broyden–Fletcher–Goldfarb–Shanno algorithm (BFGS)

    * unconstrained
    * nonlinear

    steps:
    1) determines the descent direction by precondition the gradient with curvature information


    


----

# Nelder–Mead

|key | value|
|---|---|
|name | Nelder–Mead| 
|func | <class 'flex_optimization.methods.active_methods.scipy.nelder_mead.MethodNelderMead'>| 
|type_ | MethodType.ACTIVE_SIMPLEX| 

    Method: Nelder Mead algorithm
    * also known as downhill simplex method, amoeba method

    * unconstrained
    * nonlinear

    steps:
    1) determines the descent direction by precondition the gradient with curvature information


    


----

# Powell

|key | value|
|---|---|
|name | Powell| 
|func | <class 'flex_optimization.methods.active_methods.scipy.powell.MethodPowell'>| 
|type_ | MethodType.ACTIVE_LINESEARCH| 

    Method: Powell algorithm


    steps:
    The method minimises the function by a bi-directional search along each search vector, in turn.


    


----

# Trust Constraint

|key | value|
|---|---|
|name | Trust Constraint| 
|func | <class 'flex_optimization.methods.active_methods.scipy.trust_constraint.MethodTrustConstraint'>| 
|type_ | MethodType.ACTIVE_TRUST| 

    Method: Trust Constraint

    https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html#optimize-minimize-trustconstr

    steps:


    


----

# random

|key | value|
|---|---|
|name | random| 
|func | <class 'flex_optimization.methods.active_methods.baysian_dragon.MethodBODragon'>| 
|type_ | MethodType.ACTIVE_BAYESIAN| 
 Setup for single objective only


----

