# start_imports
from MooPy.moopy import PSE, show_results
import numpy as np
# end_imports

#### -- Test problem formulation -- ####
# Define objective functions
def fun1(x):
    return (x[0] - 3)**2 + (x[1] - 3)**2 + 1

def fun2(x):
    return (x[0] - 1)**2 + (x[1] - 1)**2 + 1

funs = [fun1, fun2]

# Limits
limits = ([0, 5], [0, 5])

# Constraints
constraints = [{'type': 'ineq', 'fun': lambda x: -(0.1*x[0]**2 - x[1] - 0.1)}]

# Initial Data set
ds_ini = [np.array([3.0, 3.0]),
          np.array([1.0, 1.0])]

# Jacobian
def jac1(x):
    return np.asarray([2 * (x[0] - 3), 2 * (x[1] - 3)], dtype=float)

def jac2(x):
    return np.asarray([2 * (x[0] - 1), 2 * (x[1] - 1)], dtype=float)

jac = [jac1, jac2]

# MOO method of updating sections
moo_options = {}


### -- Start main -- ####
# --------------------------------------------
print('--------- Start PSE method -----------')

PSE_method = PSE(funs=funs,
                 ds_ini=ds_ini,
                 lims=limits,
                 cons=constraints,
                 jac=jac,
                 options=moo_options,
                 )
res_PSE, info_PSE = PSE_method.solve()


#### -- Show results -- ####
# Show output MOOP solver
plot = show_results.ShowPareto()
plot.output(res_PSE, opt=1)
show_results.print_Pf(res_PSE)
plot = show_results.ShowPareto()
plot.input(res_PSE, opt=1)
show_results.print_Px(res_PSE)

# --------------------------------------------
print('--------- End PSE method -----------')
print('')

