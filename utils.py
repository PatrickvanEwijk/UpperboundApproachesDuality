import numpy as np
import os
import random
from matplotlib import pyplot as plt
import re

payoff_maxcal=  lambda x, strike: np.maximum(np.max(x, axis=-1) - strike,0)

def set_seeds(seed=0):
    from tensorflow import random as random_tf
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    random_tf.set_seed(seed)
    np.random.seed(seed)

def smoothen_label(i):
    ## Smoothen name of methods in Tables.
    pattern_remove_labels=r'((-\d+)?)\s(\d{1,5})$'
    method=re.sub(pattern_remove_labels, '', i)
    pattern_remove_labels2=r'KU\d{1,5}'   
    method=re.sub(pattern_remove_labels2, '', method)
    method=method.replace('lambda', r'$\lambda$')
    if 'desai' in method.lower() and 'high test' not in method:
        method+=' [I]'
    method=method.replace('high test', r' [II]')
    method=method.replace('Belomestny', 'Belom.')
    method=method.replace('Belomesty', 'Belom.')
    method=method.replace('Schoenmakers', 'Schoenm.')
    method=method.replace('Anderson', 'Andersen')
    method=method.replace('best', '[I]')
    method=method.replace('original', '[II]')
    pattern_remove_labels=r'(-\d+)'
    method=re.sub(pattern_remove_labels, '', method)
    return method
    ##

len_inf = 6
len_inf_fbm = 7
delta_len = len_inf_fbm-len_inf
information_format= lambda information: [[smoothen_label(i[0]), *i[1:4], *["%.4f" %np.round(i[j],4) + ' (' + "%.4f" %np.round(i[j+1],4) + ')' for j in [8, 10, 14]], i[16], i[13], i[12]+i[13] ] for i in information]
def information_format_fbm(information):
    try: 
        return [[smoothen_label(i[0]), i[1],  str(np.round(i[2],3)) + '\\', *["%.4f" %np.round(i[j],4) + ' (' + "%.4f" %np.round(i[j+1],4) + ')' for j in np.array([7, 9, 13, 16])], i[15], i[12], i[11]+i[12] ] for i in information] 
    except Exception: 
        return [[smoothen_label(i[0]), i[1],  str(np.round(i[2],3)) + '\\', *["%.4f" %np.round(i[j],4) + ' (' + "%.4f" %np.round(i[j+1],4) + ')' for j in np.array([7, 9, 13, 16])+1], i[16], i[13], i[12]+i[13] ] for i in information]  

def information_format_fbm_no_fu(information):
    try: 
        return [[smoothen_label(i[0]), i[1], str(np.round(i[2],3)) + '\\', *["%.4f" %np.round(i[j],4) + ' (' + "%.4f" %np.round(i[j+1],4) + ')' for j in np.array([7, 9, 13])], i[15], i[12], i[11]+i[12] ] for i in information] 
    except Exception: 
        return [[smoothen_label(i[0]), i[1], str(np.round(i[2],3)) + '\\', *["%.4f" %np.round(i[j],4) + ' (' + "%.4f" %np.round(i[j+1],4) + ')' for j in np.array([7, 9, 13])+1], i[16], i[13], i[12]+i[13] ] for i in information]  

def information_format_fbm_BBS(information):
    try: 
        val_= str(np.round((information[0])[7],4))
        return [[smoothen_label(i[0]), *i[1:3], *[i[9], i[10]*1000], i[15], i[-1]] for i in information] 
    except Exception: 
        return [[smoothen_label(i[0]), *i[1:3], "%.5f" %np.round(i[10],5)+ ' (' + "%.5f" %np.round(i[11],5) + ')', i[16], i[-1]] for i in information]  
information_format_maxcall_BBS= lambda information: [[smoothen_label(i[0]), *i[1:3], "%.4f" %np.round(i[10], 4) + ' (' + "%.4f" %np.round(i[10+1],4) +')', i[16], i[-1]] for i in information]

def process_function_output(lowerbound, lowerbound_std, upperbound, upperbound_std,time_training,time_upperbound, CV_lowerbound, CV_lowerbound_std, up_fuiji=None, up_fuiji_std=None, label_='', grid=None, info_cols=None):
    gap = upperbound-lowerbound
    if up_fuiji is not None:
        return [f'{label_} {grid}', *info_cols, lowerbound, lowerbound_std, upperbound, upperbound_std,time_training.total_seconds(),time_upperbound.total_seconds(), CV_lowerbound, CV_lowerbound_std, gap, up_fuiji, up_fuiji_std]
    try:
        return [f'{label_} {grid}', *info_cols, lowerbound, lowerbound_std, upperbound, upperbound_std,time_training.total_seconds(),time_upperbound.total_seconds(), CV_lowerbound, CV_lowerbound_std, gap]
    except:                                     #0.1659, 0.001547426, 0.1669724, 0.001420824203190345, 18.448083,                     476.318604,                  0.16521417,        0.001427427, 0.00104, 0.162593767, 0.00143019814]
        pass
    finally:
        return [f'{label_} {grid}', *info_cols, lowerbound, lowerbound_std, upperbound, upperbound_std,time_training.total_seconds(), time_upperbound.total_seconds(), CV_lowerbound, CV_lowerbound_std, gap]
header_ = ['Method', '$d$', '$S_0$', 'ex rights', 'Lower bound (s.e.)','Upper bound (s.e.)', 'Lowerbound CV (s.e.)' , 'GAP', 'Time Dual (s)', 'Time Primal \& Dual (s)',]
header_fbm = ['Method', '$d$', '$H$', 'Lower bound (s.e.)','Upper bound (s.e.)',  'Lower bound CV (s.e.)', 'Upperbound Fujii (s.e.)' , 'GAP', 'Time Dual (s)', 'Time Primal \& Dual (s)']
header_fbm_no_fu=['Method', '$d$', '$H$', 'Lower bound (s.e.)','Upper bound (s.e.)',  'Lower bound CV (s.e.)', 'GAP', 'Time Dual (s)', 'Time Primal \& Dual (s)']
header_bbs= ['Method', r'$\mu_A$', r'$\sigma_A$', 'Upper bound (s.e.)', 'GAP', '95\% Upper Quantile']
header_bbs_fbm= ['Method', r'$\mu_A$', r'$\sigma_A$', 'Upper bound (s.e.)' , 'GAP', '95\% Upper Quantile']
fol = r'figures\\'   # If plot_save=True, plots are saved in this Folder on PC.
#plt.rcParams['font.family'] = 'Calibri' 
plt.rcParams['text.usetex'] = True
def save_graph(fn, scale=500, plot_save=True):
    """
    Function to save graphs on PC. Used if plot_save=True
    """
    #scale: standard 300. DPI of image
    plt.rc('legend',fontsize=8) 
    if plot_save:
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
        plt.savefig( fol+ '\\'+ fn  ,dpi=scale, bbox_inches='tight', pad_inches=0.35)
        plt.close()
    else:
        plt.show()
