import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import utils
from datetime import datetime
from tabulate import tabulate
from tensorflow import keras, constant_initializer, compat, random as random_tf
import numpy as np
import pickle as pic
from multiplestoppingSAA import main as mainSAA
from itertools import product
information=[]
if __name__=='__main__':
    for d,s0,n_stopping_rights in [ (2, 90, 3)]:#, (2, 90, 6), (2,90, 5), (2, 90, 1)]:
        for grid in [700]:
            print(''.join(['*' for j in range(10)]), grid ,''.join(['*' for j in range(10)]))
            for i in range(1):                
                print(''.join(['-' for j in range(10)]), i , ''.join(['-' for j in range(10)]))
                for mean_A, std_A in [*list(product([.2, .6, 1, 1.2], [0.01, 0.05, .3, 1.0, 1.75])), (1.0, 0)]:
                    print(f'mean {mean_A} - std {std_A}')
                    list_inf = mainSAA(d, n_stopping_rights, True, grid=grid, K_low=300,K_up=200, traj_est=200000, traj_test_ub=13000, traj_train_ub=8000, traj_test_lb=200000, S_0=s0, seed=i+8, mode_desai_BBS_BHS='bbs', mean_BBS=mean_A, std_BBS=std_A) 
                    label_= f'bbs 90,{n_stopping_rights},13000'
                    inf_cols = [mean_A, std_A , n_stopping_rights, '', '', '', '']
                    inf_list=utils.process_function_output(*list_inf, label_ = label_, grid= grid, info_cols=inf_cols)
                    information.append(inf_list)
    information.sort(key=lambda x: (x[1], x[2]))
    with open(f'run{datetime.now().strftime("%Y%m%d%H%m%S")}.pic', 'wb') as fh:
        pic.dump(information, fh)
   
    table_ = tabulate(utils.information_format(information), headers=utils.header_, tablefmt="latex_raw", floatfmt=".4f")
    print(table_)
    # folder_txt_log = '/content/drive/MyDrive/'#Tilburg/msc/Thesis/Log'
    # fh = open(f'logresults.txt', 'a')
    # fh.write(f'{datetime.now()}\n ')
    # fh.writelines(table_)
    # line="".join(np.repeat('*',75))
    # fh.write(f'\n {line} \n')
    # fh.close()