# Copyright 2019-- Derk Kappelle
#
# This file is part of MooPy, a Python package with
# Multi-Objective Optimization (MOO) tools.
#
# MooPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MooPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MooPy.  If not, see <http://www.gnu.org/licenses/>.

"""
Unified interfaces to a priori MOOP methods.

Functions
---------
- solve : Produces Pareto front for M > 2.
- solve_BOOP : Produces Pareto front for M = 2.
"""

from __future__ import absolute_import, division, print_function

__all__ = ['solve', 'solve_boop']


from warnings import warn

import numpy as np


def solve():
    pass


def solve_boop(funs, ds_ini=None, limits=None, constaints=None, jacobian=None,
               method='NC', options=None, *args, ** kwargs):

    """Producing the Pareto front of 2 objective functions.

    Parameters
    ----------
    funs : list
        List of two objective functions, callable.
        Each function must return a scalar.

    ds_ini : list or ndarray, shape (n,), optional
        If ndarray it is treated as initial guess.
        Array of real elements of size (n,),
        where 'n' is the number of independent variables.
        Else it is a list of initial Pareto points (ndarray).
        If the list contains only one Pareto point it will
        as a special initial guess. When the list contains
        2 or more points the Pareto front between these points
        is produced.

    limits : sequence or `design limits`, optional
        Sequence of ``(min, max)`` pairs for each element in `x`. None
               is used to specify no bound.

        See Scipy documentation

    constraints : {Constraint, dict} or List of {Constraint, dict}, optional
        Each dictionary with fields:

            type : str
                Constraint type: 'eq' for equality, 'ineq' for inequality.
            fun : callable
                The function defining the constraint.
            jac : callable, optional
                The Jacobian of `fun` (only for SLSQP).
            args : sequence, optional
                Extra arguments to be passed to the function and Jacobian.

        See Scipy documentation

    jacobian : list of {callable,  '2-point', '3-point', 'cs', bool}, optional
        Methods for computing the gradient vectors of each objective function.

    method : str or callable, optional
        Type of solver.  Should be one of
            - 'NC'        :ref:`(see here) <>`
            - 'PSE'       :ref:`(see here) <>`

        If not given, 'NC'

    options : dict, optional
        A dictionary of solver options. All methods accept the following
        generic options:

            maxiter : int
                Maximum number of iterations to perform. Depending on the
                method each iteration may use several function evaluations.
            disp : bool
                Set to True to print convergence messages.

        For method-specific options, see :func:`show_options()`.

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, ``success`` a
        Boolean flag indicating if the optimizer exited successfully and
        ``message`` which describes the cause of the termination. See
        `OptimizeResult` for a description of other attributes
    """

    x0 = np.asarray(x0)
    if x0.dtype.kind in np.typecodes["AllInteger"]:
        x0 = np.asarray(x0, dtype=float)

    if not isinstance(args, tuple):
        args = (args,)

    if method is None:
        method = 'NC'

    if callable(method):
        meth = "_custom"
    else:
        meth = method.lower()

    self.ninp = len(ds_ini[0][0])
    if not isinstance(funs, FunctionWrapper):
        self.funcs = FunctionWrapper(funs)
    else:
        self.funcs = funs
    if not isinstance(lims, LimitWrapper):
        self.limits = LimitWrapper(lims, self.ninp)
    else:
        self.limits = lims
    if not isinstance(cons, ConstraintWrapper):
        self.constraints = ConstraintWrapper(cons)
    else:
        self.constraints = cons
    if not isinstance(jac, JacobianWrapper):
        self.jac = JacobianWrapper(self.funcs.funcs_ini, jac)
    else:
        self.jac = jac

    # self.ds_ini = []
    # for xi in ds_ini:
    #     self.ds_ini.append(self.get_dp(xi[0]))

    if options is None:
        self.options = {}
    else:
        self.options = dict(options)

    self.fit_func = self.options.pop('fit_func', 'pol2')
    self.npar = self.get_npar(self.fit_func)
    self.nsamp = self.options.pop('nsample', 5)
    self.Npar = self.options.pop('Npar', 20)
    self.tol = self.options.pop('tol', 1e-6)
    self.finish_min = self.options.pop('finish_min', True)
    self.prt_info = self.options.get('print_info', False)
    self.prt_steps = self.options.get('print_steps', False)

    dd_method = self.options.pop('dd_method', {})
    if not isinstance(dd_method, dict):
        dd_method = dict(dd_method)

    if not dd_method:
        self.dx = (self.limits.upb - self.limits.lob) / self.Npar
        self.cal_dd = self.dd_dx
    elif 'c' in dd_method:
        self.dx = dd_method['c']
        self.cal_dd = self.dd_c
    elif 'dx' in dd_method:
        self.dx = dd_method['dx']
        self.cal_dd = self.dd_dx
    elif 'df' in dd_method:
        self.dx = dd_method['df']
        self.cal_dd = self.dd_df
    else:
        raise ValueError

    self.d1 = self.options.pop('d1', 0.01)
    self.d2 = self.options.pop('d2', 0.01)

    SOOP_options = self.options.pop('SOOP_options', {})
    if not isinstance(SOOP_options, dict):
        SOOP_options = dict(SOOP_options)

    if not SOOP_options:
        self.perform_SOOP = self.SOOP_EC
        self.restricted = False
    else:
        if 'SOOP_method' in SOOP_options:
            SOOP_method = SOOP_options['SOOP_method']
            if SOOP_method == 'NC':
                self.perform_SOOP = self.SOOP_EC
            elif callable(SOOP_method):
                self.perform_SOOP = SOOP_method
            else:
                self.perform_SOOP = self.SOOP_EC
        else:
            self.perform_SOOP = self.SOOP_EC
        if 'Restricted' in SOOP_options:
            self.restricted = SOOP_options['Restricted']
        else:
            self.restricted = False

    # dp0 = self.get_dp(x_ini)
    dp0 = self.get_dp(ds_ini[0][0])

    self.ninp = len(ds_ini[0][0])
    if not isinstance(funs, FunctionWrapper):
        self.funcs = FunctionWrapper(funs)
    else:
        self.funcs = funs
    if not isinstance(lims, LimitWrapper):
        self.limits = LimitWrapper(lims, self.ninp)
    else:
        self.limits = lims
    if not isinstance(cons, ConstraintWrapper):
        self.constraints = ConstraintWrapper(cons)
    else:
        self.constraints = cons
    if not isinstance(jac, JacobianWrapper):
        self.jac = JacobianWrapper(self.funcs.funcs_ini, jac)
    else:
        self.jac = jac

    if options is None:
        self.options = {}
    else:
        self.options = dict(options)

    self.Npar = int(self.options.pop('delta', 20))
    self.tol = self.options.pop('tol', 1e-6)
    self.prt_info = self.options.get('print_info', False)

    self.ds_ini = []
    for xi in ds_ini:
        self.ds_ini.append(self.get_dp(xi[0]))

    self.v = []
    for i, dp in enumerate(self.ds_ini):
        if i != 1:
            self.v.append(self.ds_ini[1].f - dp.f)

    self.delta = 1 / self.Npar
    a1 = 1.
    a2 = 0.
    self.p = []
    for i in range(self.Npar):
        self.p.append(a1 * self.ds_ini[0].f + a2 * self.ds_ini[1].f)
        a1 -= self.delta
        a2 += self.delta

    dp0 = self.ds_ini[0]
    self.ds = []
    self.it = 1




    if not isinstance(args, tuple):
        args = (args,)

    if callable(method):
        meth = "_custom"
    else:
        meth = method.lower()
    if options is None:
        options = {}

    if tol is not None:
        options = dict(options)
        if meth == 'bounded' and 'xatol' not in options:
            warn("Method 'bounded' does not support relative tolerance in x; "
                 "defaulting to absolute tolerance.", RuntimeWarning)
            options['xatol'] = tol
        elif meth == '_custom':
            options.setdefault('tol', tol)
        else:
            options.setdefault('xtol', tol)

    if meth == '_custom':
        return method(fun, args=args, bracket=bracket, bounds=bounds, **options)
    elif meth == 'brent':
        return _minimize_scalar_brent(fun, bracket, args, **options)
    elif meth == 'bounded':
        if bounds is None:
            raise ValueError('The `bounds` parameter is mandatory for '
                             'method `bounded`.')
        # replace boolean "disp" option, if specified, by an integer value, as
        # expected by _minimize_scalar_bounded()
        disp = options.get('disp')
        if isinstance(disp, bool):
            options['disp'] = 2 * int(disp)
        return _minimize_scalar_bounded(fun, bounds, args, **options)
    elif meth == 'golden':
        return _minimize_scalar_golden(fun, bracket, args, **options)
    else:
        raise ValueError('Unknown solver %s' % method)


# def standardize_bounds(bounds, x0, meth):
#     """Converts bounds to the form required by the solver."""
#     if meth == 'trust-constr':
#         if not isinstance(bounds, Bounds):
#             lb, ub = old_bound_to_new(bounds)
#             bounds = Bounds(lb, ub)
#     elif meth in ('l-bfgs-b', 'tnc', 'slsqp'):
#         if isinstance(bounds, Bounds):
#             bounds = new_bounds_to_old(bounds.lb, bounds.ub, x0.shape[0])
#     return bounds
#
#
# def standardize_constraints(constraints, x0, meth):
#     """Converts constraints to the form required by the solver."""
#     all_constraint_types = (NonlinearConstraint, LinearConstraint, dict)
#     new_constraint_types = all_constraint_types[:-1]
#     if isinstance(constraints, all_constraint_types):
#         constraints = [constraints]
#     constraints = list(constraints)  # ensure it's a mutable sequence
#
#     if meth == 'trust-constr':
#         for i, con in enumerate(constraints):
#             if not isinstance(con, new_constraint_types):
#                 constraints[i] = old_constraint_to_new(i, con)
#     else:
#         # iterate over copy, changing original
#         for i, con in enumerate(list(constraints)):
#             if isinstance(con, new_constraint_types):
#                 old_constraints = new_constraint_to_old(con, x0)
#                 constraints[i] = old_constraints[0]
#                 constraints.extend(old_constraints[1:])  # appends 1 if present
#
#     return constraints



