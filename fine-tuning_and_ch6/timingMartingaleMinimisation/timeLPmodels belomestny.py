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

with open('run20240522000533.pic', 'rb') as fh:
    info=pic.load(fh)

list_relevant=[]
for i in info:
    label_=i[0]
    time_steps=i[4]
    k_used=i[5]
    i_try=i[6]
    samples_used=i[7]
    times_solver=i[12]
    if (time_steps>7 and k_used>120) or (time_steps*k_used>samples_used) or (time_steps>7 and k_used==120 and samples_used>5000) or (time_steps==7 and (k_used>120 and samples_used>=10000)):
        print('ignore')
    else:
        for solver_model in range(5):
            list_relevant.append([label_, time_steps, k_used, i_try, samples_used, solver_model, times_solver[solver_model]])

DATA=pd.DataFrame.from_records(list_relevant, columns=['label', 'timesteps', 'k', 'try', 'samplesize', 'linearmodel', 'time'])
DATA['timesteps'] = DATA['timesteps']- DATA['linearmodel']
DATA['variables']=2*DATA.k*DATA.timesteps+DATA.samplesize
DATA['constraints']=DATA.samplesize*(DATA.timesteps+1)
DATA['constraints2']=2*(DATA.timesteps)*(DATA.k)
# DATA.loc[DATA.label.str.contains('bhs'), 'variables']=DATA.loc[DATA.label.str.contains('bhs'), 'variables']+1-DATA.loc[DATA.label.str.contains('bhs'), 'samplesize']

DATA['logtime']=np.log(DATA['time'].dt.total_seconds())
DATA=DATA.loc[~(DATA.logtime==-1*np.inf)]
DATA['logconstraints']=np.log(DATA.constraints)
DATA['logconstraints2']=np.log(DATA.constraints2)
DATA['logvariables']=np.log(DATA.variables)

DATA['logk']=np.log(DATA.k)
DATA['logtimesteps']=np.log(DATA.timesteps)
DATA['logsamplesize']=np.log(DATA.samplesize)

### TABLE IMPACT lambda and p computation time
DATA_mean = DATA[['label', 'timesteps', 'k', 'try', 'samplesize', 'linearmodel', 'time']].groupby(['label', 'timesteps', 'k', 'samplesize', 'linearmodel'])['time'].mean().dt.total_seconds()
mean_time_label_A = DATA_mean[DATA_mean.index.get_level_values('label') == 'B2013-0-100 200']
Rel_time=DATA_mean.reset_index().apply(lambda x: x.time/mean_time_label_A['B2013-0-100 200', x.timesteps, x.k, x.samplesize, x.linearmodel], axis=1)
Rel_time.index= DATA_mean.index
Rel_time=Rel_time.reset_index()
Rel_time= Rel_time.groupby(['label'])[0].mean().reset_index()
Rel_time['lambda_']=Rel_time['label'].apply(lambda x: x.split('-')[1])
Rel_time['p_']= Rel_time['label'].apply(lambda x: (x.split('-')[2]).split()[0])
Rel_time['label']= Rel_time.apply(lambda x: fr'Belomestny (2013), $\lambda={x.lambda_}$, $p={x.p_}$', axis=1)
Rel_time = Rel_time[['label', 0]].rename({0:'Relative time'}, axis=1)
table_rel = tabulate(Rel_time.values, headers=['Approach', 'Relative Time'], tablefmt="latex_raw", floatfmt=".4f")
print(table_rel)

result_list=[]


for label_ in np.unique(DATA.label):
    print(label_)
    lambda_=label_.split('-')[1]
    p_ = (label_.split('-')[2]).split()[0]
    label_formal= fr'Belomestny (2013), $\lambda={lambda_}$, $p={p_}$' 
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

    save_graph(f'{label_}.pdf')
    #### ERROR PLOT
    fig = plt.figure()
    ## Empirical
    ax = fig.add_subplot(111, projection='3d')
    #realised_data = np.vstack((DATA_grouped.logk, DATA_grouped.logsamplesize, np.ones(DATA_grouped.shape[0])*np.log(time_step_plots))).T
    #estimated_values = reg.predict(np.hstack((np.ones((realised_data.shape[0], 1)), realised_data)))
    #error = DATA_grouped.logtime.values - estimated_values

    error= DATA_grouped.logtime.values- reg.predict(np.vstack([np.ones((len(DATA_grouped))), DATA_grouped.logk.values,np.ones((len(DATA_grouped)))*np.log(time_step_plots) , DATA_grouped.logsamplesize.values]).T)

    ax.scatter(DATA_grouped.logk, DATA_grouped.logsamplesize, error ,label='Error')

    ax.set_title(label_formal)
    ax.set_xlabel(r'$\log(K)$')
    ax.set_ylabel(r'$\log(N_{train})$')
    ax.set_zlabel(r'$\log(time)$')
    ax.set_box_aspect(aspect=None, zoom=0.95)
    ax.view_init(elev=17, azim=-53, roll=0)
    ax.legend()

    save_graph(f'{label_}-error.pdf')
    abs_error= np.abs(error)

    print('q 0.5; 0.75; 0.8; 0.9 ', np.quantile(abs_error, [0.5, 0.75, 0.8, 0.9]))
    print('perc <0.35', np.mean(abs_error<0.35))

table_ = tabulate(result_list, headers=['Approach', r'$\delta_0$', r'$\delta_{k}$',r'$\delta_{timesteps}$',r'$\delta_{samplesize}$', r'$R^{2}$'], tablefmt="latex_raw", floatfmt=".4f")
print(table_)