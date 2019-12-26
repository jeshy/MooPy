import numpy as np
from math import pi


def case(n):
    if n == 'sim_lin':
        return sim_lin()
    elif n == 'sim_lin_2':
        return sim_lin_2()
    elif n == 'sim_nlin':
        return sim_nlin()
    elif n == 'sim_lim':
        return sim_lim()
    elif n == 'sim_lim_2':
        return sim_lim_2()
    elif n == 'sim_con':
        return sim_con()
    elif n == 'sim_con_2':
        return sim_con_2()
    elif n == 'lim_con_nat':
        return lim_con_nat()
    elif n == 'disco':
        return disco()
    elif n == 'tintout':
        return tintout()
    elif n == 'KUR':
        return KUR()
    elif n == 'KUR_2':
        return KUR_2()
    elif n == 'ZTD2':
        return ZTD2()
    elif n == 'ZTD2_2':
        return ZTD2_2()
    elif n == 'ZTD3':
        return ZTD3()
    elif n == 'ZTD3_2':
        return ZTD3_2()
    elif n == 'BaK':
        return BaK()
    elif n == 'CaH':
        return CaH()

def sim_lin():  # Simple linear
    # Define objective functions
    def fun1(x):
        return (x[0] - 3)**2 + (x[1] - 3)**2 + 1

    def fun2(x):
        return (x[0] - 1)**2 + (x[1] - 1)**2 + 1

    funcs = [fun1, fun2]

    # Initial input
    x_ini = np.array([1, 1])

    # Limits
    limits = ([0, 5], [0, 5])

    # Constraints
    constraints = None

    # Initial Data set
    ds_ini = [[np.array([3.0, 3.0]), 0],
              [np.array([1.0, 1.0]), 1]]

    # Jacobian
    def jac1(x):
        return np.asarray([2 * (x[0] - 3), 2 * (x[1] - 3)], dtype=float)

    def jac2(x):
        return np.asarray([2 * (x[0] - 1), 2 * (x[1] - 1)], dtype=float)

    jac = [jac1, jac2]

    # Hessian
    hess = None

    # MOO method of updating sections
    moo_method = None
    moo_options = {}

    # Method for initialization of the data set
    ini_method = None
    ini_options = {}

    return funcs, x_ini, ds_ini, limits, constraints, jac, \
           hess, moo_method, ini_method, moo_options, ini_options


def sim_lin_2():  # Simple non-linear
    # Define objective functions
    def fun1(x):
        return (x[0] - 3)**2 + (x[1] - 3)**2 + 1

    def fun2(x):
        return (x[0] - 1)**4 + (x[1] - 1)**4 + 1

    funcs = [fun1, fun2]

    # Initial input
    x_ini = np.array([1, 1])

    # Limits
    limits = ([0, 5], [0, 5])

    # Constraints
    constraints = None

    # Initial Data set
    ds_ini = [[np.array([3.0, 3.0]), 0],
              [np.array([1.0, 1.0]), 1]]

    # Jacobian
    def jac1(x):
        return np.asarray([2 * (x[0] - 3), 2 * (x[1] - 3)], dtype=float)

    def jac2(x):
        return np.asarray([4 * (x[0] - 1)**3, 4 * (x[1] - 1)**3], dtype=float)

    jac = [jac1, jac2]

    # Hessian
    hess = None

    # MOO method of updating sections
    moo_method = None
    moo_options = {}

    # Method for initialization of the data set
    ini_method = None
    ini_options = {}

    return funcs, x_ini, ds_ini, limits, constraints, jac, \
           hess, moo_method, ini_method, moo_options, ini_options


def sim_nlin():  # Simple non-linear
    # Define objective functions
    def fun1(x):
        return (x[0] - 3)**2 + (x[1] - 3)**2 + 1

    def fun2(x):
        return 0.25*(x[0] - 1)**2 + (x[1] - 1)**2 + 1

    funcs = [fun1, fun2]

    # Initial input
    x_ini = np.array([1, 1])

    # Limits
    limits = ([0, 5], [0, 5])

    # Constraints
    constraints = None

    # Initial Data set
    ds_ini = [[np.array([3.0, 3.0]), 0],
              [np.array([1.0, 1.0]), 1]]

    # Jacobian
    def jac1(x):
        return np.asarray([2 * (x[0] - 3), 2 * (x[1] - 3)], dtype=float)

    def jac2(x):
        return np.asarray([0.5 * (x[0] - 1), 2 * (x[1] - 1)], dtype=float)

    jac = [jac1, jac2]

    # Hessian
    hess = None

    # MOO method of updating sections
    moo_method = None
    moo_options = {}

    # Method for initialization of the data set
    ini_method = None
    ini_options = {}

    return funcs, x_ini, ds_ini, limits, constraints, jac, \
           hess, moo_method, ini_method, moo_options, ini_options


def sim_lim():  # Simple and limited
    # Define objective functions
    def fun1(x):
        return (x[0] - 3)**2 + (x[1] - 3)**2 + 2

    def fun2(x):
        return x[0] + 2*x[1]

    funcs = [fun1, fun2]

    # Initial input
    x_ini = np.array([1, 1])

    # Limits
    limits = ([0, 5], [0, 5])

    # Constraints
    constraints = None

    # initial Data set
    ds_ini = [[np.array([3.0, 3.0]), 0],
              [np.array([0.0, 0.0]), 1]]

    # Jacobian
    def jac1(x):
        return np.asarray([2 * (x[0] - 3), 2 * (x[1] - 3)], dtype=float)

    def jac2(x):
        return np.array([1., 2.])

    jac = [jac1, jac2]

    # Hessian
    hess = None

    # MOO method of updating sections
    moo_method = None
    moo_options = {}

    # Method for initialization of the data set
    ini_method = None
    ini_options = {}

    return funcs, x_ini, ds_ini, limits, constraints, jac, \
           hess, moo_method, ini_method, moo_options, ini_options


def sim_lim_2():  # Simple and limited
    # Define objective functions
    def fun1(x):
        return x[0] + 2 * x[1]

    def fun2(x):
        return (x[0] - 3) ** 2 + (x[1] - 3) ** 2 + 2


    funcs = [fun1, fun2]

    # Initial input
    x_ini = np.array([1, 1])

    # Limits
    limits = ([0, 5], [0, 5])

    # Constraints
    constraints = None

    # initial Data set
    ds_ini = [[np.array([0.0, 0.0]), 0],
              [np.array([3.0, 3.0]), 1]]

    # Jacobian
    def jac1(x):
        return np.array([1., 2.])

    def jac2(x):
        return np.asarray([2 * (x[0] - 3), 2 * (x[1] - 3)], dtype=float)

    jac = [jac1, jac2]

    # Hessian
    hess = None

    # MOO method of updating sections
    moo_method = None
    moo_options = {}

    # Method for initialization of the data set
    ini_method = None
    ini_options = {}

    return funcs, x_ini, ds_ini, limits, constraints, jac, \
           hess, moo_method, ini_method, moo_options, ini_options


def sim_con():   # Simple and constrained
    # Define objective functions
    def fun1(x):
        return (x[0] - 3)**2 + (x[1] - 3)**2 + 2

    def fun2(x):
        return x[0] + 2*x[1]

    funcs = [fun1, fun2]

    # Initial input
    x_ini = np.array([1, 1])

    # Limits
    limits = ([0, 5], [0, 5])

    # Constraints
    constraints = [{'type': 'ineq', 'fun': lambda x: -(0.1*x[0]**2 - x[1] + 0.5)}]

    # initial Data set
    ds_ini = [[np.array([3.0, 3.0]), 0],
              [np.array([0.0, 0.5]), 1]]

    # Jacobian
    def jac1(x):
        return np.asarray([2 * (x[0] - 3), 2 * (x[1] - 3)], dtype=float)

    def jac2(x):
        return np.array([1., 2.], dtype=float)

    jac = [jac1, jac2]

    # Hessian
    hess = None

    # MOO method of updating sections
    moo_method = None
    moo_options = {}

    # Method for initialization of the data set
    ini_method = None
    ini_options = {}

    return funcs, x_ini, ds_ini, limits, constraints, jac, \
           hess, moo_method, ini_method, moo_options, ini_options


def sim_con_2():   # Simple and constrained
    # Define objective functions
    def fun1(x):
        return x[0] + 2 * x[1]

    def fun2(x):
        return (x[0] - 3) ** 2 + (x[1] - 3) ** 2 + 2

    funcs = [fun1, fun2]

    # Initial input
    x_ini = np.array([1, 1])

    # Limits
    limits = ([0, 5], [0, 5])

    # Constraints
    constraints = [{'type': 'ineq', 'fun': lambda x: -(0.1*x[0]**2 - x[1] + 0.5)}]

    # initial Data set
    ds_ini = [[np.array([0.0, 0.5]), 0],
              [np.array([3.0, 3.0]), 1]]

    # Jacobian
    def jac1(x):
        return np.array([1., 2.], dtype=float)

    def jac2(x):
        return np.asarray([2 * (x[0] - 3), 2 * (x[1] - 3)], dtype=float)

    jac = [jac1, jac2]

    # Hessian
    hess = None

    # MOO method of updating sections
    moo_method = None
    moo_options = {}

    # Method for initialization of the data set
    ini_method = None
    ini_options = {}

    return funcs, x_ini, ds_ini, limits, constraints, jac, \
           hess, moo_method, ini_method, moo_options, ini_options


def lim_con_nat():   # Simple and constrained
    # Define objective functions
    def fun1(x):
        return x[0] + 2 * x[1]

    def fun2(x):
        return (x[0] - 3) ** 2 + (x[1] - 3) ** 2 + 2

    funcs = [fun1, fun2]

    # Initial input
    x_ini = np.array([1, 1])

    # Limits
    limits = ([0, 5], [0, 5])

    # Constraints
    constraints = [{'type': 'ineq', 'fun': lambda x: -(0.1*x[0]**2 - x[1] - 0.1)}]

    # initial Data set
    ds_ini = [[np.array([0.0, 0.0]), 0],
              [np.array([3.0, 3.0]), 1]]

    # Jacobian
    def jac1(x):
        return np.array([1., 2.], dtype=float)

    def jac2(x):
        return np.asarray([2 * (x[0] - 3), 2 * (x[1] - 3)], dtype=float)

    jac = [jac1, jac2]

    # Hessian
    hess = None

    # MOO method of updating sections
    moo_method = None
    moo_options = {}

    # Method for initialization of the data set
    ini_method = None
    ini_options = {}

    return funcs, x_ini, ds_ini, limits, constraints, jac, \
           hess, moo_method, ini_method, moo_options, ini_options


def disco():   # Discontinuous
    # Define objective functions
    def a(x):
        return 1 + 9*(x[1] - 1)**2

    def fun1(x):
        return x[0]

    def h(x):
        return 1 - np.sqrt(fun1(x) / a(x)) - (fun1(x) / a(x)) * np.sin((1.5/2.5) * pi * fun1(x))

    def fun2(x):
        f2 = 0.1*a(x)*h(x)
        return f2

    funcs = [fun1, fun2]

    # Initial input
    x_ini = np.array([3, 4])

    # Limits
    limits = ([0, 5], [0, 5])

    # Constraints
    constraints = None

    # initial Data set
    ds_ini = [[np.array([0.0, 1.0]), 0],
              [np.array([1.163, 1.0]), 1],
              [np.array([4.2485, 1.0]), 1]]

    # ds_ini = [[np.array([0.0, 0.9939631]), 0],
    #           [np.array([1.16359019,  1.00024886]), 1],
    #           [np.array([4.24886734,  1.0653759]), 1]]

    # Jacobian
    jac = None

    # Hessian
    hess = None

    # MOO method of updating sections
    moo_method = None
    moo_options = {}

    # Method for initialization of the data set
    ini_method = None
    ini_options = {}

    return funcs, x_ini, ds_ini, limits, constraints, jac, \
           hess, moo_method, ini_method, moo_options, ini_options


def tintout():  # 2 in 3 out
    # Define objective functions
    def fun1(x):
        return (x[0] - 1)**2 + (x[1] - 1)**2 + 1

    def fun2(x):
        return (x[0] - 4)**2 + (x[1] - 2)**2 + 1

    def fun3(x):
        return (x[0] - 2)**2 + (x[1] - 4)**2 + 1

    funcs = [fun1, fun2, fun3]

    # Initial input
    x_ini = np.array([1, 1])

    # Limits
    limits = ([0, 5], [0, 5])

    # Constraints
    constraints = None

    # Initial Data set
    ds_ini = [[np.array([1.0, 1.0]), 0],
              [np.array([4.0, 2.0]), 1],
              [np.array([2.0, 4.0]), 2]]

    # Jacobian
    def jac1(x):
        return np.asarray([2 * (x[0] - 1), 2 * (x[1] - 1)], dtype=float)

    def jac2(x):
        return np.asarray([2 * (x[0] - 4), 2 * (x[1] - 2)], dtype=float)

    def jac3(x):
        return np.asarray([2 * (x[0] - 2), 2 * (x[1] - 4)], dtype=float)

    jac = [jac1, jac2, jac3]

    # Hessian
    hess = None

    # MOO method of updating sections
    moo_method = None
    moo_options = {}

    # Method for initialization of the data set
    ini_method = None
    ini_options = {}

    return funcs, x_ini, ds_ini, limits, constraints, jac, \
           hess, moo_method, ini_method, moo_options, ini_options


def KUR():  # KUR problem
    # Define objective functions
    def fun1(x):
        sum = 0
        for i in range(0, len(x) - 1):
            sum = sum - 10 * np.exp(-0.2 * np.sqrt((x[i]) ** 2 + (x[i + 1]) ** 2))
        f1 = sum
        return f1

    def fun2(x):
        sum = 0
        for i in range(0, len(x)):
            sum = sum + (np.abs(x[i]) ** 0.8 + 5 * (np.sin((x[i]))) ** 3)
        f2 = sum
        return f2

    funcs = [fun1, fun2]

    # Initial input
    # x0 = np.asarray([-1.5216198, 0.0, -1.5216198])
    x_ini = np.asarray([0.0, 0.0, 0.0])
    # x0 = np.asarray([4.0, 4.0, 4.0])

    # Limits
    limits = ([-5, 5], [-5, 5], [-5, 5])

    # Constraints
    constraints = None
    # initial Data set
    # ds = [[np.asarray([0.0, 0.0, 0.0]),0],
    #       [np.asarray([-1.5216198, 0.0, 0.0]),1],
    #       [np.asarray([0.0, 0.0, -1.5216198]),1],
    #       [np.asarray([-1.5216198, 0.0, -1.5216198]),1],
    #       [np.asarray([-1.5216198, -1.5216198, -1.5216198]),1]]

    ds_ini = [[np.asarray([-1.00940081e-08,   8.19811851e-09,   5.71849748e-09]),0],
          # [np.asarray([-1.5216198, 0.0, 0.0]),1],
          [np.asarray([-8.56028533e-14,  -5.08347373e-14,  -1.52161916e+00]),1],
          [np.asarray([ -1.52161912e+00,  -1.10468205e-13,  -1.52161897e+00]),1],
          [np.asarray([-1.52161954, -1.52161821, -1.5216178 ]),1]]

    # ds = [[np.asarray([-1.00940081e-08,   8.19811851e-09,   5.71849748e-09]),0],
    #       # [np.asarray([-1.5216198, 0.0, 0.0]),1],
    #       [np.asarray([-8.56028533e-14,  -5.08347373e-14,  -1.52161916e+00]),1]]

    # Jacobian
    jac = None

    # Hessian
    hess = None

    # MOO method of updating sections
    moo_method = None
    moo_options = {}

    # Method for initialization of the data set
    ini_method = None
    ini_options = {}

    return funcs, x_ini, ds_ini, limits, constraints, jac, \
           hess, moo_method, ini_method, moo_options, ini_options


def KUR_2():  # KUR problem
    # Define objective functions
    def fun1(x):
        sum = 0
        for i in range(0, len(x) - 1):
            sum = sum - 10 * np.exp(-0.2 * np.sqrt((x[i]) ** 2 + (x[i + 1]) ** 2))
        f1 = sum
        return f1

    def fun2(x):
        sum = 0
        for i in range(0, len(x)):
            sum = sum + (np.abs(x[i]) ** 0.8 + 5 * (np.sin((x[i]))) ** 3)
        f2 = sum
        return f2

    funcs = [fun1, fun2]

    # Initial input
    # x0 = np.asarray([-1.5216198, 0.0, -1.5216198])
    x_ini = np.asarray([0.0, 0.0, 0.0])
    # x0 = np.asarray([4.0, 4.0, 4.0])

    # Limits
    limits = ([-5, 5], [-5, 5], [-5, 5])

    # Constraints
    constraints = None

    # initial Data set
    ds_ini = [[np.asarray([-1.00940081e-08,   8.19811851e-09,   5.71849748e-09]), 0],
              [np.asarray([-1.52161954, -1.52161821, -1.5216178]), 1]]

    # Jacobian
    jac = None

    # Hessian
    hess = None

    # MOO method of updating sections
    moo_method = None
    moo_options = {}

    # Method for initialization of the data set
    ini_method = None
    ini_options = {}

    return funcs, x_ini, ds_ini, limits, constraints, jac, \
           hess, moo_method, ini_method, moo_options, ini_options


def ZTD2():
    # Define objective functions
    def g(x):
        sum = 0
        for i in range(1, len(x)):
            sum = sum + x[i]
        return 1 + 9*(sum**2)/29

    def fun1(x):
        return x[0]

    def h(x):
        return 1 - (fun1(x) / g(x))**2

    def fun2(x):
        f2 = g(x)*h(x)
        return f2

    funcs = [fun1, fun2]

    # Initial input
    x_ini = np.asarray([0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0.])
    # Limits
    limits = ([0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
              [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
              [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
              [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
              [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
              [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
              )

    # Constraints
    constraints = None

    # initial Data set
    ds_ini = [[np.asarray([0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.]), 0],
              [np.asarray([1., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.]), 1]]

      # Jacobian
    jac = None

    # Hessian
    hess = None

    # MOO method of updating sections
    moo_method = None
    moo_options = {}

    # Method for initialization of the data set
    ini_method = None
    ini_options = {}

    return funcs, x_ini, ds_ini, limits, constraints, jac, \
           hess, moo_method, ini_method, moo_options, ini_options


def ZTD2_2():
    # Define objective functions
    def g(x):
        sum = 0
        for i in range(1, len(x)):
            sum = sum + x[i]
        # return 1 + 9*(sum**2)/29
        return 1 + 9*sum/29

    def fun1(x):
        return x[0]

    def h(x):
        return 1 - np.sqrt(fun1(x) / g(x)) - (fun1(x) / g(x)) * np.sin(10 * pi * fun1(x))

    def fun2(x):
        f2 = g(x)*h(x)
        return f2

    funcs = [fun1, fun2]

    # Initial input
    x_ini = np.asarray([0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0.])
    # Limits
    limits = ([0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
              [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
              [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
              [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
              [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
              [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
              )

    # Constraints
    constraints = None

    # initial Data set
    ds_ini = [[np.asarray([0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.]), 0],
              [np.asarray([0.083, 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.]), 1],
              [np.asarray([0.257763, 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.]), 1],
              [np.asarray([0.45388, 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.]), 1],
              [np.asarray([0.65251159, 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.]), 1],
              [np.asarray([0.85183286, 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.]), 1]]

      # Jacobian
    jac = None

    # Hessian
    hess = None

    # MOO method of updating sections
    moo_method = None
    moo_options = {}

    # Method for initialization of the data set
    ini_method = None
    ini_options = {}

    return funcs, x_ini, ds_ini, limits, constraints, jac, \
           hess, moo_method, ini_method, moo_options, ini_options


def ZTD3():
    # Define objective functions
    def g(x):
        sum = 0
        for i in range(1, len(x)):
            sum = sum + x[i]
        # return 1 + 9*(sum**2)/29
        return 1 + 9*sum/29

    def fun1(x):
        return x[0]

    def h(x):
        return 1 - np.sqrt(fun1(x) / g(x)) - (fun1(x) / g(x)) * np.sin(10 * pi * fun1(x))

    def fun2(x):
        f2 = g(x)*h(x)
        return f2

    funcs = [fun1, fun2]

    # Initial input
    x_ini = np.asarray([0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0.])
    # Limits
    limits = ([0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
              [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
              [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
              [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
              [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
              [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
              )

    # Constraints
    constraints = None

    # initial Data set
    ds_ini = [[np.asarray([0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.]), 0],
              [np.asarray([0.083, 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.]), 1],
              [np.asarray([0.257763, 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.]), 1],
              [np.asarray([0.45388, 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.]), 1],
              [np.asarray([0.65251159, 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.]), 1],
              [np.asarray([0.85183286, 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.]), 1]]

      # Jacobian
    jac = None

    # Hessian
    hess = None

    # MOO method of updating sections
    moo_method = None
    moo_options = {}

    # Method for initialization of the data set
    ini_method = None
    ini_options = {}

    return funcs, x_ini, ds_ini, limits, constraints, jac, \
           hess, moo_method, ini_method, moo_options, ini_options


def ZTD3_2():
    # Define objective functions
    def g(x):
        sum = 0
        for i in range(1, len(x)):
            sum = sum + x[i]
        # return 1 + 9*(sum**2)/29
        return 1 + 9*sum/29

    def fun1(x):
        return x[0]

    def h(x):
        return 1 - np.sqrt(fun1(x) / g(x)) - (fun1(x) / g(x)) * np.sin(10 * pi * fun1(x))

    def fun2(x):
        f2 = g(x)*h(x)
        return f2

    funcs = [fun1, fun2]

    # Initial input
    x_ini = np.asarray([0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0.])
    # Limits
    limits = ([0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
              [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
              [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
              [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
              [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
              [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
              )

    # Constraints
    constraints = None

    # initial Data set
    ds_ini = [[np.asarray([0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.]), 0],
              [np.asarray([0.85183286, 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.]), 1]]

      # Jacobian
    jac = None

    # Hessian
    hess = None

    # MOO method of updating sections
    moo_method = None
    moo_options = {}

    # Method for initialization of the data set
    ini_method = None
    ini_options = {}

    return funcs, x_ini, ds_ini, limits, constraints, jac, \
           hess, moo_method, ini_method, moo_options, ini_options

def BaK():
    # Define objective functions
    def fun1(x):
        return 4*x[0]**2 + 4*x[1]**2

    def fun2(x):
        return (x[0] - 5)**2 + (x[1] - 5)**2

    funcs = [fun1, fun2]

    # Initial input
    x_ini = np.asarray([0., 0.])

    # Limits
    limits = ([0., 5.], [0., 3.])

    # Constraints
    constraints = [{'type': 'ineq', 'fun': lambda x: -((x[0] - 5)**2 + x[1]**2) + 25},
                   {'type': 'ineq', 'fun': lambda x: (x[0] - 8)**2 + (x[1] + 3)**2 - 7.7}]


    # initial Data set
    ds_ini = [[np.asarray([0., 0.]), 0],
              [np.asarray([5., 3.]), 1]]

      # Jacobian
    jac = None

    # Hessian
    hess = None

    # MOO method of updating sections
    moo_method = None
    moo_options = {}

    # Method for initialization of the data set
    ini_method = None
    ini_options = {}

    return funcs, x_ini, ds_ini, limits, constraints, jac, \
           hess, moo_method, ini_method, moo_options, ini_options

def CaH():
    # Define objective functions
    def fun1(x):
        return 2 + (x[0] - 2)**2 + (x[1] - 1)**2

    def fun2(x):
        return 9*x[0] - (x[1] - 1)**2

    funcs = [fun1, fun2]

    # Initial input
    x_ini = np.asarray([0., 0.])

    # Limits
    limits = ([-20., 20.], [-20., 20.])

    # Constraints
    constraints = [{'type': 'ineq', 'fun': lambda x: -(x[0]**2 + x[1]**2) + 225},
                   {'type': 'ineq', 'fun': lambda x: -(x[0] - 3*x[1] + 10.)}]


    # initial Data set
    # ds_ini = [[np.asarray([1.1, 3.7]), 0],
    #           [np.asarray([ -4.84097722932 , 14.1973569421 ]), 1]]
    # ds_ini = [[np.asarray([1.09558503219 , 3.69852834406 ]), 0],
    #           [np.asarray([ -4.84097722932 , 14.1973569421 ]), 1]]
    ds_ini = [[np.asarray([1.09558503219 , 3.69852834406 ]), 0],
              [np.asarray([-2.19940578791 , 2.6001980707]), 0],
              [np.asarray([ -4.84097722932 , 14.1973569421 ]), 1]]
      # Jacobian
    jac = None

    # Hessian
    hess = None

    # MOO method of updating sections
    moo_method = None
    moo_options = {}

    # Method for initialization of the data set
    ini_method = None
    ini_options = {}

    return funcs, x_ini, ds_ini, limits, constraints, jac, \
           hess, moo_method, ini_method, moo_options, ini_options

def LZF4():
    # Define objective functions
    def fun1(x):
        return 2 + (x[0] - 2)**2 + (x[1] - 1)**2

    def fun2(x):
        return 9*x[0] - (x[1] - 1)**2

    funcs = [fun1, fun2]

    # Initial input
    x_ini = np.asarray([0., 0.])

    # Limits
    limits = ([-20., 20.], [-20., 20.])

    # Constraints
    constraints = [{'type': 'ineq', 'fun': lambda x: -(x[0]**2 + x[1]**2) + 225},
                   {'type': 'ineq', 'fun': lambda x: -(x[0] - 3*x[1] + 10.)}]


    # initial Data set
    # ds_ini = [[np.asarray([1.1, 3.7]), 0],
    #           [np.asarray([ -4.84097722932 , 14.1973569421 ]), 1]]
    # ds_ini = [[np.asarray([1.09558503219 , 3.69852834406 ]), 0],
    #           [np.asarray([ -4.84097722932 , 14.1973569421 ]), 1]]
    ds_ini = [[np.asarray([1.09558503219 , 3.69852834406 ]), 0],
              [np.asarray([-2.19940578791 , 2.6001980707]), 0],
              [np.asarray([ -4.84097722932 , 14.1973569421 ]), 1]]
      # Jacobian
    jac = None

    # Hessian
    hess = None

    # MOO method of updating sections
    moo_method = None
    moo_options = {}

    # Method for initialization of the data set
    ini_method = None
    ini_options = {}

    return funcs, x_ini, ds_ini, limits, constraints, jac, \
           hess, moo_method, ini_method, moo_options, ini_options
