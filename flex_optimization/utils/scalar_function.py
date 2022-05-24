from typing import Union, Callable

import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import LinearOperator

from flex_optimization.utils.hessian_strategy import HessianUpdateStrategy
from flex_optimization.utils.finite_differance import approx_derivative
from flex_optimization.problem_statement import Problem


FD_METHODS = ('2-point', '3-point', 'cs')


class ScalarFunction:
    """Scalar function and its derivatives.
    This class defines a scalar function F: R^n->R and methods for
    computing or approximating its first and second derivatives.

    Parameters
    ----------
    fun : callable
        evaluates the scalar function. Must be of the form ``fun(x, *args)``,
        where ``x`` is the argument in the form of a 1-D array and ``args`` is
        a tuple of any additional fixed parameters needed to completely specify
        the function. Should return a scalar.
        x0 : array-like
        Provides an initial set of variables for evaluating fun. Array of real
        elements of size (n,), where 'n' is the number of independent
        variables.
    args : tuple, optional
        Any additional fixed parameters needed to completely specify the scalar
        function.
        grad : {callable, '2-point', '3-point', 'cs'}
        Method for computing the gradient vector.
        If it is a callable, it should be a function that returns the gradient
    vector:
        ``grad(x, *args) -> array_like, shape (n,)``
    where ``x`` is an array with shape (n,) and ``args`` is a tuple with
    the fixed parameters.
    Alternatively, the keywords  {'2-point', '3-point', 'cs'} can be used
    to select a finite difference scheme for numerical estimation of the
    gradient with a relative step size. These finite difference schemes
    obey any specified `bounds`.
    hess : {callable, '2-point', '3-point', 'cs', HessianUpdateStrategy}
    Method for computing the Hessian matrix. If it is callable, it should
    return the  Hessian matrix:
        ``hess(x, *args) -> {LinearOperator, spmatrix, array}, (n, n)``
    where x is a (n,) ndarray and `args` is a tuple with the fixed
    parameters. Alternatively, the keywords {'2-point', '3-point', 'cs'}
    select a finite difference scheme for numerical estimation. Or, objects
    implementing `HessianUpdateStrategy` interface can be used to
    approximate the Hessian.
    Whenever the gradient is estimated via finite-differences, the Hessian
    cannot be estimated with options {'2-point', '3-point', 'cs'} and needs
    to be estimated using one of the quasi-Newton strategies.
    finite_diff_rel_step : None or array_like
    Relative step size to use. The absolute step size is computed as
    ``h = finite_diff_rel_step * sign(x0) * max(1, abs(x0))``, possibly
    adjusted to fit into the bounds. For ``method='3-point'`` the sign
    of `h` is ignored. If None then finite_diff_rel_step is selected
    automatically,
    finite_diff_bounds : tuple of array_like
    Lower and upper bounds on independent variables. Defaults to no bounds,
    (-np.inf, np.inf). Each bound must match the size of `x0` or be a
    scalar, in the latter case the bound will be the same for all
    variables. Use it to limit the range of function evaluation.
    epsilon : None or array_like, optional
    Absolute step size to use, possibly adjusted to fit into the bounds.
    For ``method='3-point'`` the sign of `epsilon` is ignored. By default
    relative steps are used, only if ``epsilon is not None`` are absolute
    steps used.
    Notes
    -----
    This class implements a memoization logic. There are methods `fun`,
    `grad`, hess` and corresponding attributes `f`, `g` and `H`. The following
    things should be considered:
    1. Use only public methods `fun`, `grad` and `hess`.
    2. After one of the methods is called, the corresponding attribute
       will be set. However, a subsequent call with a different argument
       of *any* of the methods may overwrite the attribute.
    """

    def __init__(self,
                 problem: Problem,
                 x0,
                 grad: Union[Callable, str] = None,
                 hess: Union[Callable, str] = None,
                 finite_diff_rel_step=None,
                 finite_diff_bounds=None,
                 epsilon=None):

        if hess is None:
            # ScalarFunction requires something for hess, so we give a dummy
            # implementation here if nothing is provided, return a value of None
            # so that downstream minimisers halt. The results of `fun.hess`
            # should not be used.
            def hess(x, *args):
                return None

        if finite_diff_bounds is None:
            finite_diff_bounds = (-np.inf, np.inf)

        if not callable(grad) and grad not in FD_METHODS:
            raise ValueError(
                f"`grad` must be either callable or one of {FD_METHODS}."
            )

        if not (callable(hess) or hess in FD_METHODS
                or isinstance(hess, HessianUpdateStrategy)):
            raise ValueError(
                f"`hess` must be either callable, HessianUpdateStrategy"
                f" or one of {FD_METHODS}."
            )

        if grad in FD_METHODS and hess in FD_METHODS:
            raise ValueError("Whenever the gradient is estimated via "
                             "finite-differences, we require the Hessian "
                             "to be estimated using one of the "
                             "quasi-Newton strategies.")

        # the astype call ensures that self.x is a copy of x0
        self.x = np.atleast_1d(x0).astype(float)
        self.n = self.x.size
        self.nfev = 0
        self.ngev = 0
        self.nhev = 0
        self.f_updated = False
        self.g_updated = False
        self.H_updated = False

        self._lowest_x = None
        self._lowest_f = np.inf

        finite_diff_options = {}
        if grad in FD_METHODS:
            finite_diff_options["method"] = grad
            finite_diff_options["rel_step"] = finite_diff_rel_step
            finite_diff_options["abs_step"] = epsilon
            finite_diff_options["bounds"] = finite_diff_bounds
        if hess in FD_METHODS:
            finite_diff_options["method"] = hess
            finite_diff_options["rel_step"] = finite_diff_rel_step
            finite_diff_options["abs_step"] = epsilon
            finite_diff_options["as_linear_operator"] = True

        # Function evaluation
        def fun_wrapped(x):
            self.nfev += 1
            # Send a copy because the user may overwrite it.
            # Overwriting results in undefined behaviour because
            # fun(self.x) will change self.x, with the two no longer linked.
            fx = problem.evaluate(np.copy(x))
            # Make sure the function returns a true scalar
            if not np.isscalar(fx):
                try:
                    fx = np.asarray(fx).item()
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        "The user-provided objective function "
                        "must return a scalar value."
                    ) from e

            if fx < self._lowest_f:
                self._lowest_x = x
                self._lowest_f = fx

            return fx

        def update_fun():
            self.f = fun_wrapped(self.x)

        self._update_fun_impl = update_fun
        self._update_fun()

        # Gradient evaluation
        if callable(grad):
            pass
            def grad_wrapped(x):
                self.ngev += 1
                return np.atleast_1d(grad(np.copy(x)))

            def update_grad():
                self.g = grad_wrapped(self.x)

        elif grad in FD_METHODS:
            def update_grad():
                self._update_fun()
                self.ngev += 1
                self.g = approx_derivative(problem.evaluate, self.x, f0=self.f,
                                           **finite_diff_options)

        self._update_grad_impl = update_grad
        self._update_grad()

        # Hessian Evaluation
        if callable(hess):
            self.H = hess(np.copy(x0))
            self.H_updated = True
            self.nhev += 1

            if sps.issparse(self.H):
                def hess_wrapped(x):
                    self.nhev += 1
                    return sps.csr_matrix(hess(np.copy(x)))

                self.H = sps.csr_matrix(self.H)

            elif isinstance(self.H, LinearOperator):
                def hess_wrapped(x):
                    self.nhev += 1
                    return hess(np.copy(x))

            else:
                def hess_wrapped(x):
                    self.nhev += 1
                    return np.atleast_2d(np.asarray(hess(np.copy(x))))

                self.H = np.atleast_2d(np.asarray(self.H))

            def update_hess():
                self.H = hess_wrapped(self.x)

        elif hess in FD_METHODS:
            def update_hess():
                self._update_grad()
                self.H = approx_derivative(grad_wrapped, self.x, f0=self.g,
                                           **finite_diff_options)
                return self.H

            update_hess()
            self.H_updated = True
        elif isinstance(hess, HessianUpdateStrategy):
            self.H = hess
            self.H.initialize(self.n, 'hess')
            self.H_updated = True
            self.x_prev = None
            self.g_prev = None

            def update_hess():
                self._update_grad()
                self.H.update(self.x - self.x_prev, self.g - self.g_prev)

        self._update_hess_impl = update_hess

        if isinstance(hess, HessianUpdateStrategy):
            def update_x(x):
                self._update_grad()
                self.x_prev = self.x
                self.g_prev = self.g
                # ensure that self.x is a copy of x. Don't store a reference
                # otherwise the memoization doesn't work properly.
                self.x = np.atleast_1d(x).astype(float)
                self.f_updated = False
                self.g_updated = False
                self.H_updated = False
                self._update_hess()
        else:
            def update_x(x):
                # ensure that self.x is a copy of x. Don't store a reference
                # otherwise the memoization doesn't work properly.
                self.x = np.atleast_1d(x).astype(float)
                self.f_updated = False
                self.g_updated = False
                self.H_updated = False
        self._update_x_impl = update_x

    def _update_fun(self):
        if not self.f_updated:
            self._update_fun_impl()
            self.f_updated = True

    def _update_grad(self):
        if not self.g_updated:
            self._update_grad_impl()
            self.g_updated = True

    def _update_hess(self):
        if not self.H_updated:
            self._update_hess_impl()
            self.H_updated = True

    def fun(self, x):
        if not np.array_equal(x, self.x):
            self._update_x_impl(x)
        self._update_fun()
        return self.f

    def grad(self, x):
        if not np.array_equal(x, self.x):
            self._update_x_impl(x)
        self._update_grad()
        return self.g

    def hess(self, x):
        if not np.array_equal(x, self.x):
            self._update_x_impl(x)
        self._update_hess()
        return self.H
