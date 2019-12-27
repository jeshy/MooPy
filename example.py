# start_imports
from MooPy.moopy import PSE, test_cases, show_results
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

case = 'LZf4'

plot_res_Pf = [True, False, False, False, False]
pr_res_Pf = [True, True, True, True, True]

plot_res_Px = [True, False, False, False, False]
pr_res_Px = [True, True, True, True, True]

funs, x_ini, ds_ini, limits, constraints, jac, moo_options = test_cases.case(case)

# ds_ini[1][0] = x_ini
revers = moo_options.pop('reversed', False)
print_info = False
moo_options['print_info'] = print_info

### -- Start main -- ####
# --------------------------------------------
print('--------- Start PSE method -----------')

if revers:
    funs.revers()
    ds_ini = list(reversed(ds_ini))
    jac.revers()


PSE_method = PSE()
# Solve the MOOP with the PSE method
res_PSE, info_PSE = PSE_method.solve(funs=funs,
                                     ds_ini=ds_ini,
                                     lims=limits,
                                     cons=constraints,
                                     jac=jac,
                                     options=moo_options,
                                     )

info_PSE.append(show_results.Eveness(res_PSE))
if print_info:
    print('Eveness:', "%.5f" % info_PSE[4])

#### -- Show results -- ####
# Show output MOOP solver
if plot_res_Pf[0]:
    plot = show_results.ShowPareto()
    plot.output(res_PSE, opt=1)
if pr_res_Pf[0]:
    show_results.print_Pf(res_PSE)

if plot_res_Px[0]:
    plot = show_results.ShowPareto()
    plot.input(res_PSE, opt=1)
if pr_res_Px[0]:
    show_results.print_Px(res_PSE)

print('PSE &', info_PSE[0], '&', info_PSE[3] , '&', "%.5f" % info_PSE[4], '\\\\')

# --------------------------------------------
print('--------- End PSE method -----------')
print('')

