import pickle as pic
import pandas as pd
import numpy as np
import statsmodels.api as sm


from sklearn import linear_model
from tabulate import tabulate
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import save_graph
from datetime import datetime
with open('run20240513180559.pic', 'rb') as fh:
    info=pic.load(fh)

print(info[0])
#inf_cols = [d, s0, n_stopping_rights, step, k,  i, samples_est]
#[f'{label_} {grid}', *info_cols, lowerbound, lowerbound_std, upperbound, upperbound_std,time_training,time_upperbound.total_seconds(), CV_lowerbound, CV_lowerbound_std, gap]
list_relevant=[]
for i in info:
    label_=i[0]
    time_steps=i[4]
    k_used=i[5]
    i_try=i[6]
    samples_used=i[7]
    times_solver=i[12]
    for solver_model in range(5):
        list_relevant.append([label_, time_steps, k_used, i_try, samples_used, solver_model, times_solver[solver_model]])

DATA=pd.DataFrame.from_records(list_relevant, columns=['label', 'timesteps', 'k', 'try', 'samplesize', 'linearmodel', 'time'])
DATA['variables']=2*DATA.k*DATA.timesteps+DATA.samplesize
DATA['constraints']=DATA.samplesize*(DATA.timesteps+1)
DATA['constraints2']=2*(DATA.timesteps)*(DATA.k)
DATA.loc[DATA.label.str.contains('bhs'), 'variables']=DATA.loc[DATA.label.str.contains('bhs'), 'variables']+1-DATA.loc[DATA.label.str.contains('bhs'), 'samplesize']

DATA['logtime']=np.log(DATA.time)
DATA['logconstraints']=np.log(DATA.constraints)
DATA['logconstraints2']=np.log(DATA.constraints2)
DATA['logvariables']=np.log(DATA.variables)

DATA['logk']=np.log(DATA.k)
DATA['logtimesteps']=np.log(DATA.timesteps)
DATA['logsamplesize']=np.log(DATA.samplesize)
result_list=[]

for label_ in np.unique(DATA.label):
    if 'bhs' in label_:
        label_formal ='Belomestny, Hildebrand, Schoenmakers (2019)'
    elif 'bbs' in label_:
        label_formal='Belomestny, Bender, Schoenmakers (2023)'
    else:
        label_formal='Desai et al. (2012)'
    print(label_formal)

    DATA_selection = DATA.loc[(DATA.label==label_)]
    print(len(DATA_selection))
    val, counts = np.unique(DATA_selection.timesteps.values, return_counts=True)
    # print(list(zip(val, counts)))
    reg = sm.OLS(DATA_selection.logtime.values, np.hstack((np.ones((len(DATA_selection), 1)), DATA_selection[['logk','logtimesteps', 'logsamplesize']].values))).fit()
    reg_coeff=reg.params
    se_coeff = reg.HC0_se
    r_squared = reg.rsquared
    # print(reg.summary())
    time_step_plots= 7# Largest sample w.r.t. time steps. Used for plots.
    DATA_grouped = (DATA_selection.groupby(['label', 'timesteps', 'k', 'try', 'samplesize', 'logk','logsamplesize', 'logvariables'])['logtime'].mean()).reset_index()
    DATA_grouped = DATA_grouped.loc[DATA_grouped.timesteps==time_step_plots] # Largest sample w.r.t. time steps. Used for plots.

    
    # regression_info = [*[str(np.round(coef_,4))+ f' ({np.round(se_,4)})' for coef_,se_ in list(zip(reg_coeff, se_coeff))], r_squared]
    # result_list.append([label_formal,*regression_info])

    reg_coeff = np.round(reg_coeff, 4)
    result_list.append([label_formal,*reg_coeff, np.round(r_squared,4)]) # Coefficients+R^2
    result_list.append(['',  *[f'({np.round(se_,4)})' for se_ in se_coeff], '']) # S.e. on seperate line in table

    fig = plt.figure()
    ## Empirical
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(DATA_grouped.logk, DATA_grouped.logsamplesize, DATA_grouped.logtime, label='Realised')
    ## Estimated line OLS
    x = np.linspace(min(DATA_grouped.logk.values), max(DATA_grouped.logk.values), 2)
    y = np.linspace(min(DATA_grouped.logsamplesize.values), max(DATA_grouped.logsamplesize.values), 2)
    X, Y = np.meshgrid(x, y)
    input_data = np.vstack((np.ones((np.size(X))), X.flatten(), np.ones((np.size(X)))*np.log(time_step_plots), Y.flatten())).T
    fitted_function = reg.predict(input_data)
    Z = fitted_function.reshape(X.shape)

    # Plot the estimated function 
    ax.plot_surface(X, Y, Z, alpha=0.7,color='C1', label='Estimated')

    ax.set_title(label_formal)
    ax.set_xlabel(r'$\log(K)$')
    ax.set_ylabel(r'$\log(N_{train})$')
    ax.set_zlabel(r'$\log(time)$')
    ax.set_box_aspect(aspect=None, zoom=0.95)
    ax.legend()
    ax.view_init(elev=17, azim=-53, roll=0)

    save_graph(f'{label_formal}.pdf')

    #### ERROR PLOT
    fig = plt.figure()
    ## Empirical
    ax = fig.add_subplot(111, projection='3d')
    error=DATA_grouped.logtime- reg.predict(np.vstack([np.ones((len(DATA_grouped))), DATA_grouped.logk.values,np.ones((len(DATA_grouped)))*np.log(time_step_plots) , DATA_grouped.logsamplesize.values]).T)

    ax.scatter(DATA_grouped.logk, DATA_grouped.logsamplesize, error ,label='Error')

    ax.set_title(label_formal)
    ax.set_xlabel(r'$\log(K)$')
    ax.set_ylabel(r'$\log(N_{train})$')
    ax.set_zlabel(r'$\log(time)$')
    ax.set_box_aspect(aspect=None, zoom=0.95)
    ax.view_init(elev=17, azim=-53, roll=0)
    ax.legend()

    save_graph(f'{label_formal}-error.pdf')
    abs_error= np.abs(error)

    print('q 0.5; 0.75; 0.8; 0.9 ', np.quantile(abs_error.values, [0.5, 0.75, 0.8, 0.9]))
    print('perc <0.35', np.mean(abs_error<0.35))

table_ = tabulate(result_list, headers=['Approach', r'$\delta_0$', r'$\delta_{k}$',r'$\delta_{timesteps}$',r'$\delta_{samplesize}$', r'$R^{2}$'], tablefmt="latex_raw", floatfmt=".4f")
print(table_)