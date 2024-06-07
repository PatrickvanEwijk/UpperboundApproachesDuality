import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import utils

from tensorflow import keras, constant_initializer, compat, random as random_tf
import numpy as np
from fBMSAA import main as mainSAA
compat.v1.logging.set_verbosity(compat.v1.logging.ERROR)
import pickle as pic
from tabulate import tabulate
from datetime import datetime
from itertools import product
information=[]
if __name__=='__main__':
    for d,H in [ (1,0.2)]:
        for grid in [700]:
            print(''.join(['*' for j in range(10)]), grid ,''.join(['*' for j in range(10)]))
            for mean_A, std_A in [*list(product([.2, .6, 1], [.01, 0.05, .3, 1.0, 1.75])), (1.0, 0)]:
                print(f'mean {mean_A} - std {std_A}')
                list_inf = mainSAA(d, True, grid=grid, K_low=300,K_up=200, steps=9, traj_est=200000, traj_test_ub=13000, traj_train_ub=8000, traj_test_lb=200000, seed=8, mode_desai_BBS_BHS='bbs', mean_BBS=mean_A, std_BBS=std_A, mode_kaggle=False) 
                label_= f'bbs fbm,13000-{H}'
                inf_cols = [mean_A, std_A , '', '', '', '', '']
                inf_list=utils.process_function_output(*list_inf, label_ = label_, grid= grid, info_cols=inf_cols)
                information.append(inf_list)
    information.sort(key=lambda x: (x[1], x[2]))
    with open(f'run{datetime.now().strftime("%Y%m%d%H%m%S")}.pic', 'wb') as fh:
        pic.dump(information, fh)
   
    table_ = tabulate(utils.information_format_fbm(information), headers=utils.header_fbm, tablefmt="latex_raw", floatfmt=".4f")
    print(table_)
