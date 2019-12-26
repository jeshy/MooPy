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
from __future__ import absolute_import, division, print_function

import sys
import copy
import math
import time
import logging
import datetime
import operator
import functools
import itertools
import numpy as np
from abc import ABCMeta, abstractmethod

from cachetools import cached, TTLCache
cache = TTLCache(maxsize=10000, ttl=1000)
cache2 = TTLCache(maxsize=10000, ttl=1000)

LOGGER = logging.getLogger("MooPy")
EPSILON = sys.float_info.epsilon
POSITIVE_INFINITY = float("inf")

##############################################################################
# Tools (classes)
# functional wrappers
class FunctionWrapper(object):
    def __init__(self, funcs):
        self.funcs_ini = funcs
        self.num_sing_eva = 0
        self.num_arr_eva = 0
        self.noutp = len(funcs)

        self.x = None
        self.f = None
        self.i = None

    # returns solution fi(x)
    @cached(cache)
    def cahched_func(self, x, i):
        self.num_sing_eva += 1
        return float(self.funcs_ini[self.i](self.x))


    def evaluate_func(self, x, i, **kwargs):
        self.x = x
        self.i = i
        if isinstance(x, list):
            x = np.asarray(x)
        return self.cahched_func(hash(x.tostring()), i)

    # returns function fi
    def single_func(self, i, **kwargs):
        def func(x):
            return self.evaluate_func(x, i)

        return func

    # returns solution f(x), as array
    def evaluate_funcs(self, x, f=None, ind=None, **kwargs):
        self.num_arr_eva += 1
        if f is None:
            return np.asarray([self.evaluate_func(x, i) for i in range(self.noutp)], dtype=float)
        else:
            l = []
            for i in range(self.noutp):
                if i != ind:
                    l.append(self.evaluate_func(x, i))
                else:
                    l.append(f)
            return np.asarray(l)

    # returns function f, as array
    def array_funcs(self, **kwargs):
        def funcs(x):
            return self.evaluate_funcs(x)
        return funcs

    # returns function mu, for a specific lamb
    def combine_funcs(self, lamb, **kwargs):
        self.num_arr_eva += 1

        def objfunc(x):
            return sum(la * self.evaluate_func(x, i) for i, la in enumerate(lamb) if la != 0)

        return objfunc

    def clear(self):
        self.num_sing_eva = 0
        self.num_arr_eva = 0
        self.x = None
        self.f = None
        self.i = None
        cache.clear()

    def revers(self):
        self.funcs_ini = list(reversed(self.funcs_ini))


# constraints wrappers
class ConstraintWrapper(object):
    def __init__(self, constraints, eps=1e-8):
        if constraints is None:
            self.con_ini = []
            self.eps = eps
            self.cons = None
            self.x = None
            self.f = None
            self.fun = None
            self.ncons = 0
        else:
            # Constraints are triaged per type into a dictionnary of tuples
            self.con_ini = copy.copy(constraints)
            if isinstance(constraints, dict):
                constraints = (constraints,)

            self.cons = {'eq': (), 'ineq': ()}
            for ic, con in enumerate(constraints):
                # check type
                try:
                    ctype = con['type'].lower()
                except KeyError:
                    raise KeyError('Constraint %d has no type defined.' % ic)
                except TypeError:
                    raise TypeError('Constraints must be defined using a '
                                    'dictionary.')
                except AttributeError:
                    raise TypeError("Constraint's type must be a string.")
                else:
                    if ctype not in ['eq', 'ineq']:
                        raise ValueError("Unknown constraint type '%s'." % con['type'])

                # check function
                if 'fun' not in con:
                    raise ValueError('Constraint %d has no function defined.' % ic)

                # check jacobian
                cjac = con.get('jac')
                if cjac is None or cjac == '2-point':
                    def cjac_factory(fun):
                        def cjac(x, *args):
                            return self.grad_fw(x, fun, *args)

                        return cjac

                    cjac = cjac_factory(con['fun'])

                elif cjac == '3-point':
                    def cjac_factory(fun):
                        def cjac(x, *args):
                            return self.grad_mid(x, fun, *args)

                        return cjac

                    cjac = cjac_factory(con['fun'])

                # update constraints' dictionary
                self.cons[ctype] += ({'fun': con['fun'],
                                      'jac': cjac,
                                      'args': con.get('args', ())},)

                self.eps = eps

                self.x = None
                self.f = None
                self.fun = None

                self.ncons = len(constraints)

    # returns solution fprimei(x), 2-point
    def grad_fw(self, x, fun, *args):
        if np.all(self.x == x) and self.fun == fun:
            pass
        else:
            self.x = np.copy(x)
            self.fun = copy.copy(fun)
            self.f = copy.copy(fun(x))
        grad = np.zeros(len(x), float)
        dx = np.zeros(len(x), float)
        for j in range(len(x)):
            dx[j] = self.eps
            grad[j] = (fun(x + dx) - self.f) / self.eps
            dx[j] = 0.0
        return grad

    # returns solution fprimei(x), 3-point
    def grad_mid(self, x, fun, *args):
        if np.all(self.x == x) and self.fun == fun:
            pass
        else:
            self.x = np.copy(x)
            self.fun = copy.copy(fun)
            self.f = copy.copy(fun(x))
        grad = np.zeros(len(x), float)
        dx = np.zeros(len(x), float)
        for j in range(len(x)):
            dx[j] = self.eps
            grad[j] = (fun(x + dx) - fun(x - dx)) / (2 * self.eps)
            dx[j] = 0.0
        return grad

    # returns solution fi(x)
    def evaluate_con(self, x, i, **kwargs):
        return float(self.cons["ineq"][i]["fun"](x))

    # returns function fi
    def single_con(self, i, **kwargs):
        def con(x):
            return self.evaluate_con(x, i)

        return con

    # returns solution f(x), as array
    def evaluate_cons(self, x, f=None, ind=None, **kwargs):
        # self.num_arr_eva += 1
        if f is None:
            return np.asarray([self.evaluate_con(x, i) for i in range(self.ncons)], dtype=float)
        else:
            l = []
            for i in range(self.ncons):
                if i != ind:
                    l.append(self.evaluate_con(x, i))
                else:
                    l.append(f)
            return np.asarray(l)

    # returns function f, as array
    def array_cons(self, **kwargs):
        def cons(x):
            return self.evaluate_cons(x)

        return cons

    def evaluate_conj(self, x, i, **kwargs):
        return self.cons["ineq"][i]["jac"](x)

    def add_constraint(self, cons, **kwargs):
        for ic, con in enumerate(cons):
            cjac = con.get('jac')
            if cjac is None or cjac == '2-point':
                def cjac_factory(fun):
                    def cjac(x, *args):
                        return self.grad_fw(x, fun, *args)

                    return cjac

                cjac = cjac_factory(con['fun'])

            elif cjac == '3-point':
                def cjac_factory(fun):
                    def cjac(x, *args):
                        return self.grad_mid(x, fun, *args)

                    return cjac

                cjac = cjac_factory(con['fun'])

            con['jac'] = cjac

        return self.con_ini + cons


# limits wrappers
class LimitWrapper(object):
    def __init__(self, limits, n_input):
        if limits is None:
            self.lims_ini = None
        else:
            self.lims_ini = limits
        self.lob, self.upb = self.get_limits(limits, n_input)

    def get_limits(self, bounds, n_input):
        if bounds is None or len(bounds) == 0:
            return np.array([-1.0E12] * n_input), np.array([1.0E12] * n_input)
        else:
            bnds = np.array(bounds, float)
            if bnds.shape[0] != n_input:
                raise IndexError('Error: the length of bounds is not'
                                 'compatible with that of x0.')

            bnderr = np.where(bnds[:, 0] > bnds[:, 1])[0]
            if bnderr.any():
                raise ValueError('Error: lb > ub in bounds %s.' %
                                 ', '.join(str(b) for b in bnderr))
            return bnds[:, 0], bnds[:, 1]


# jacobian wrappers
class JacobianWrapper(object):
    def __init__(self, funcs, jac, eps=1e-8):
        if jac is None:
            self.jac_ini = None
        else:
            self.jac_ini = copy.copy(jac)
        self.funcs = funcs
        self.noutp = len(funcs)

        self.jac = []
        if jac == '2-point':
            self.jac = [self.gradi_fw(i) for i in range(self.noutp)]
        elif jac == '3-point':
            self.jac = [self.gradi_mid(i) for i in range(self.noutp)]
        elif isinstance(jac, (list,)):
            for i, grad in enumerate(jac):
                if callable(grad):
                    self.jac.append(grad)
                elif grad == '2-point' or grad is None:
                    self.jac.append(self.gradi_fw(i))
                elif grad == '3-point':
                    self.jac.append(self.gradi_mid(i))
                else:
                    # warn('Gradient %i is not correct, 2-point is used.' % i)
                    self.jac.append(self.gradi_fw(i))
        else:
            # warn('Jacobian is not correct, 2-point is used.')
            self.jac = [self.gradi_fw(i) for i in range(self.noutp)]

        self.num_jac_eva = 0
        self.num_grad_eva = 0
        self.num_sing_eva = 0

        self.eps = eps

        self.x = None
        self.f = None
        self.i = None

    # returns solution fi(x)
    @cached(cache2)
    def cahched_func(self, x, i):
        self.num_sing_eva += 1
        return float(self.funcs[self.i](self.x))


    def evaluate_func(self, x, i, **kwargs):
        self.x = x
        self.i = i
        if isinstance(x, list):
            x = np.asarray(x)
        return self.cahched_func(hash(x.tostring()), i)

    # returns solution fprimei(x), callable
    def evaluate_grad(self, x, i):
        self.num_grad_eva += 1
        return np.asarray(self.jac[i](x), dtype=float)

    # returns function fprimei, callable
    def single_grad(self, i):
        def func(x):
            return self.evaluate_grad(x, i)

        return func

    # returns solution fprimei(x), callable
    def evaluate_grad_neg(self, x, i):
        self.num_grad_eva += 1
        return np.asarray(-self.jac[i](x), dtype=float)

    # returns function fprimei, callable
    def single_grad_neg(self, i):
        def func(x):
            return self.evaluate_grad_neg(x, i)

        return func

    # returns solution fprimei(x), 2-point
    def grad_fw(self, x, i):
        if np.all(self.x == x) and self.i == i:
            pass
        else:
            self.x = np.copy(x)
            self.i = copy.copy(i)
            self.f = copy.copy(self.evaluate_func(x, i))
        grad = np.zeros(len(x), float)
        dx = np.zeros(len(x), float)
        for j in range(len(x)):
            dx[j] = self.eps
            grad[j] = (self.evaluate_func((x + dx), i) - self.f) / self.eps
            dx[j] = 0.0
        return grad

    # returns solution fprimei(x), 3-point
    def grad_mid(self, x, i):
        grad = np.zeros(len(x), float)
        dx = np.zeros(len(x), float)
        for j in range(len(x)):
            dx[j] = self.eps
            grad[j] = (self.evaluate_func((x + dx), i) - self.evaluate_func((x - dx), i)) / (2 * self.eps)
            dx[j] = 0.0
        return grad

    # returns function fprimei, callable 2-point
    def gradi_fw(self, i):
        def func(x):
            return self.grad_fw(x, i)

        return func

    # returns function fprimei, callable 3-point
    def gradi_mid(self, i):
        def func(x):
            return self.grad_mid(x, i)

        return func

    # returns solution jac(x)
    def evaluate_jac(self, x, grad=None, ind=None, **kwargs):
        jac = np.zeros((len(self.funcs), len(x)))
        for i in range(self.noutp):
            if i == ind and grad is not None:
                jac[i] = grad
            else:
                jac[i] = self.evaluate_grad(x, i)
        return jac

    # returns function mu, for a specific lamb
    def combine_grad(self, lamb, **kwargs):
        def objfunc(x):
            return np.sum(la * self.evaluate_grad(x, i) for i, la in enumerate(lamb) if la != 0)

        return objfunc

    # update jac for non-smooth
    def update_jac(self, jac, **kwargs):
        self.jac = []
        if jac == '2-point':
            self.jac = [self.gradi_fw(i) for i in range(self.noutp)]
        elif jac == '3-point':
            self.jac = [self.gradi_mid(i) for i in range(self.noutp)]
        elif isinstance(jac, (list,)):
            for i, grad in enumerate(jac):
                if callable(grad):
                    self.jac.append(grad)
                elif grad == '2-point' or grad is None:
                    self.jac.append(self.gradi_fw(i))
                elif grad == '3-point':
                    self.jac.append(self.gradi_mid(i))
                else:
                    # warn('Gradient %i is not correct, 2-point is used.' % i)
                    self.jac.append(self.gradi_fw(i))
        else:
            # warn('Jacobian is not correct, 2-point is used.')
            self.jac = [self.gradi_fw(i) for i in range(self.noutp)]

    def clear(self):
        self.num_jac_eva = 0
        self.num_grad_eva = 0
        self.num_sing_eva = 0
        self.x = None
        self.f = None
        self.i = None
        cache2.clear()

    def revers(self):
        self.funcs = list(reversed(self.funcs))



