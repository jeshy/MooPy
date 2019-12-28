# start_imports
from MooPy.moopy import test_cases, show_results
from MooPy.moopy import solve_boop
# end_imports

#### -- Test problem formulation -- ####
#          sim_lin
#          sim_lin_2
#          sim_nlin
#          sim_lim
#          sim_lim_2
#          sim_con
#          sim_con_2
#          lim_con_nat

#          BaK
#          CaH
#          Sf1
#          FFf
#          LZf4
#          ZDT1
#          ZDT1_extra
#          ZDT2
#          ZDT2_extra
#          ZDT4
#          ZDT6
#          OaK
#          CPT1
#          CEx
#          Tf4

test_case = 'sim_lin'

#### -- Solving techniques -- ####
#          WS
#          EC
#          NC
#          PSE
#          PSEVariableStep

method = 'PSEVariableStep'

funs, x_ini, ds_ini, limits, constraints, jac, moo_options = test_cases.case(test_case)


### -- Start main -- ####
# --------------------------------------------
print('--------- Start MOOP method -----------')

# Solve the MOOP with method
res_ds, info_method = solve_boop(funs, ds_ini, limits, constraints, jac, method, moo_options)

# --------------------------------------------
print('--------- End MOOP method -----------')
print('')


#### -- Select presentation -- ####
plot_res_Pf = True
pr_res_Pf = True

plot_res_Px = True
pr_res_Px = True

#### -- Show results -- ####
# Show results objective space of MOOP solver
if plot_res_Pf:
    plot = show_results.ShowPareto()
    plot.output(res_ds, opt=1)
if pr_res_Pf:
    show_results.print_Pf(res_ds)

# Show results design space of MOOP solver
if plot_res_Px:
    plot = show_results.ShowPareto()
    plot.input(res_ds, opt=1)
if pr_res_Px:
    show_results.print_Px(res_ds)


