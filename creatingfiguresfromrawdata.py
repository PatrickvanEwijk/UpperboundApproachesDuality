import numpy as np
np.set_printoptions(linewidth=200, edgeitems=8)
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import pickle as pic
import pandas as pd
from utils import save_graph
SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 18

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title




file_='run20240417110433.pic' # MultipleStopping
#file_ ='run20240416210454.pic' #fBM.
file_='run20240418180419.pic'
file_basename=file_.split('.')[0]
with open(file_, 'rb') as file:
    information=pic.load(file)
df = pd.DataFrame.from_records(information)
df.iloc[:,0] = (df.iloc[:,0]).str.replace('Belom. et al 1', ' Belom. et al. (2009)')
type_fbm= df.iloc[:,2].max()>=0 and df.iloc[:,2].max()<=1
print(type_fbm)


if type_fbm:
    lower_bounds = df.iloc[:,7]
    lower_bounds_cv =  df.iloc[:,13]
    methods = df.iloc[:,0]
    upper_bounds = df.iloc[:,9]
    hurst = df.iloc[:, 2]
    dataframe_sel = df.iloc[:, [0, 2, 9, 7]]
    gap = upper_bounds-lower_bounds
    print(dataframe_sel)
    unique_methods= list(np.unique(methods))
    #unique_methods = [m for m in unique_methods if ('Emp' in m or '2017' in m or 'Desai' in m or '2024' in m)]

    ### FIG GAP
    for method in unique_methods:
        mask_method = methods.index[(methods==method)]
        plt.scatter(hurst.loc[mask_method], gap.loc[mask_method], label=method)

    plt.xlabel('Hurst')
    plt.ylabel('GAP')
    plt.title('GAP vs Hurst')
    legend = plt.legend(bbox_to_anchor=(1.2, 1.2), facecolor='white')
    plt.gca().add_artist(legend)
    save_graph(f'GAP{file_basename}.pdf')

    ### FIG Bound
    fig, ax1 = plt.subplots(1, 1) 
    lb=dict()
    ub=dict()
    for number_method, method in enumerate(unique_methods):
        mask_method = methods.index[(methods==method)]
        lb[method]=ax1.scatter(hurst.loc[mask_method], lower_bounds.loc[mask_method], label=method, marker='v', c=f'C{number_method}', linewidths=.5)
        ub[method]=ax1.scatter(hurst.loc[mask_method], upper_bounds.loc[mask_method], label=method, marker='^', c=f'C{number_method}', linewidths=.5)
    dict_bounds ={method: (lb[method], ub[method]) for method in lb.keys()}
    l = ax1.legend( list(dict_bounds.values()), list(dict_bounds.keys()), handler_map={tuple: HandlerTuple(ndivide=None)},bbox_to_anchor=(1.2, 1.2)) # bbox_to_anchor=(1, 0.5),
    legend_frame = l.get_frame()
    legend_frame.set_facecolor('white')  # Set face color to white
    plt.gca().add_artist(l)
    ax1.set_xlabel('Hurst')
    ax1.set_ylabel('Expectation')
    ax1.set_title('Lowerbound and Upperbound vs Hurst')
    save_graph(f'BOUNDS{file_basename}.pdf')
    

else:
    lower_bounds = df.iloc[:,8]
    lower_bounds_cv=df.iloc[:,14]
    methods = df.iloc[:,0]
    upper_bounds = df.iloc[:, 10]
    exercise_right = df.iloc[:, 3]
    dataframe_sel = df.iloc[:, [0, 3, 8, 10]]
    gap = upper_bounds-lower_bounds_cv#upper_bounds-lower_bounds
    print(dataframe_sel)
    unique_methods= list(np.unique(methods))
    # ### to be deleted
    # unique_methods=[m for m in unique_methods if 'HK' not in m]
    # ###
    
    ### FIG GAP
    for method in unique_methods:
        mask_method = methods.index[(methods==method)]
        plt.scatter(exercise_right.loc[mask_method], gap.loc[mask_method], label=method)

    plt.xlabel('Exercise Right')
    plt.ylabel('GAP')
    plt.title('GAP vs Exercise Right')
    legend = plt.legend(bbox_to_anchor=(1.1, 0.5), facecolor='white')
    plt.gca().add_artist(legend)
    save_graph(f'GAP{file_basename}.pdf')

    ### FIG Bound
    fig, ax1 = plt.subplots(1, 1) 
    lb=dict()
    ub=dict()
    for number_method, method in enumerate(unique_methods):
        mask_method = methods.index[(methods==method)]
        lb[method]=ax1.scatter(exercise_right.loc[mask_method], lower_bounds.loc[mask_method], label=method, marker='v', c=f'C{number_method}', linewidths=.5)
        ub[method]=ax1.scatter(exercise_right.loc[mask_method], upper_bounds.loc[mask_method], label=method, marker='^', c=f'C{number_method}', linewidths=.5)
    dict_bounds ={method: (lb[method], ub[method]) for method in lb.keys()}
    l = ax1.legend( list(dict_bounds.values()), list(dict_bounds.keys()), handler_map={tuple: HandlerTuple(ndivide=None)},bbox_to_anchor=(1.1, 0.5)) # bbox_to_anchor=(1, 0.5),
    legend_frame = l.get_frame()
    legend_frame.set_facecolor('white')  # Set face color to white
    plt.gca().add_artist(l)
    ax1.set_xlabel('Exercise Right')
    ax1.set_ylabel('Value Option')
    ax1.set_title('Lowerbound and Upperbound vs Exercise Right')
    save_graph(f'BOUNDS{file_basename}.pdf')
    