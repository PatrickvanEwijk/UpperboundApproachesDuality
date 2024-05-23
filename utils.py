import numpy as np
import os
import random
from tensorflow import random as random_tf
from matplotlib import pyplot as plt
def set_seeds(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    random_tf.set_seed(seed)
    np.random.seed(seed)


len_inf = 6
len_inf_fbm = 7
delta_len = len_inf_fbm-len_inf
information_format= lambda information: [[*i[:4], *[str(np.round(i[j],4)) + ' (' + str(np.round(i[j+1],4)) + ')' for j in [8, 10, 14]], i[16], i[12], i[13] ] for i in information]
information_format_fbm= lambda information: [[*i[:3], *[str(np.round(i[j],4)) + ' (' + str(np.round(i[j+1],4)) + ')' for j in np.array([7, 9, 13, 16])], i[15], i[11], i[12] ] for i in information]

def process_function_output(lowerbound, lowerbound_std, upperbound, upperbound_std,time_training,time_upperbound, CV_lowerbound, CV_lowerbound_std, up_fuiji=None, up_fuiji_std=None, label_='', grid=None, info_cols=None):
    gap = upperbound-lowerbound
    if up_fuiji is not None:
        return [f'{label_} {grid}', *info_cols, lowerbound, lowerbound_std, upperbound, upperbound_std,time_training.total_seconds(),time_upperbound.total_seconds(), CV_lowerbound, CV_lowerbound_std, gap, up_fuiji, up_fuiji_std]
    try:
        return [f'{label_} {grid}', *info_cols, lowerbound, lowerbound_std, upperbound, upperbound_std,time_training.total_seconds(),time_upperbound.total_seconds(), CV_lowerbound, CV_lowerbound_std, gap]
    except:
        pass
    finally:
        return [f'{label_} {grid}', *info_cols, lowerbound, lowerbound_std, upperbound, upperbound_std,time_training,time_upperbound.total_seconds(), CV_lowerbound, CV_lowerbound_std, gap]
header_ = ['Method', 'd', 'S0', 'ex rights', 'Lowerbound (s.e.)','Upperbound (s.e.)', 'Lowerbound CV (s.e.)' , 'GAP', 'Time training LB', 'Time UB martingale']
header_fbm = ['Method', 'd', 'H', 'Lowerbound (s.e.)','Upperbound (s.e.)',  'Upperbound Fujii (s.e.)', 'Lowerbound CV (s.e.)' , 'GAP', 'Time training LB', 'Time UB martingale']


fol = r'figures\\'   # If plot_save=True, plots are saved in this Folder on PC.
#plt.rcParams['font.family'] = 'Calibri' 
plt.rcParams['text.usetex'] = True
def save_graph(fn, scale=300, plot_save=True):
    """
    Function to save graphs on PC. Used if plot_save=True
    """
    #scale: standard 300. DPI of image
    if plot_save:
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
        plt.savefig( fol+ '\\'+ fn  ,dpi=scale, bbox_inches='tight', pad_inches=0.35)
        plt.close()
    else:
        plt.show()
