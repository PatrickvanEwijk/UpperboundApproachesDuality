"""
File which executes Schoenmakers et al. (2013) primal-dual upper bound approach to a fractional brownian motion.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import utils

from tensorflow import keras, constant_initializer, compat, random as random_tf
import numpy as np
from modelRrobust2fBM import model_Schoenmakers
compat.v1.logging.set_verbosity(compat.v1.logging.ERROR)
import pickle as pic
from tabulate import tabulate
from datetime import datetime


def main(d=1,print_progress=True,steps= 100, T=1, traj_est=80000, grid=100, step_inner=True, traj_test_lb=150000, traj_test_ub=10000, K_low=200, K_up=50, hurst=0.7, seed=0):
    """
    Main function, which executes Schoenmakers et al. (2013) approach.

    Function also calculates a lower biased estimate based on primal LSMC with randomised neural network.

    Input:
        d: dimension of the fractional brownian motion. Stopping maximum out of d. Only d=1 is considered in report.
        print_progress: If True: printing results at the end. If False: Only printing times during loops execution algoritm 
        steps: N_T in report, number of possible future stopping dates.
        T: Time of the final possible stopping date.
        traj_est: Trajectories used for estimation of primal LSMC.
        grid: Number of inner simulations.
        traj_test_lb: Number of testing trajectories for primal LSMC.
        traj_test_ub: Number of testring trajectories to evaluate upper bound.
        K_low: Number of nodes,=#basis functions -1, in randomised neural network primal LSMC.
        K_up:  Number of nodes in dual randomised neural network, =# basis functions on martingale term.
        hurst: Hurst parameter of fbm.
        seed: Seed for randomised neural network and simulating trajectories.
       

    Output:
        lowerbound: Lower biased estimate
        lowerbound_std: standard error of lower biased estimate 
        upperbound: Upper biased estimate
        upperbound_std: Standard error of upper biased estimate
        np.mean(np.array(time_training)): Training time (list with 1 value so automatically total time)
        np.mean(np.array(time_ub)):  Testing time (list with 1 value so automatically total time)
        CV_lowerbound: Lower biased estimate using constructed martingale as control variate.
        CV_lowerbound_std: Standard error of lower biased estimate using constructed martingale as control variate.
        upperbound2: Upper bound estimate corresponding to Fuiji et al. (2011) simple improvement algorithm. Not used in report. Using at own caution.
        upperbound_std2: standard error upper bound estimate corresponding to Fuiji et al. (2011) simple improvement algorithm. Not used in report. Using at own caution.
    
    """
    time_training=[]
    time_testing=[]
    time_ub=[]
    utils.set_seeds(seed)
    dt = T/steps
    traj=traj_est
    time_ = np.arange(0,T+0.5*dt, dt)
    r=0.0

    train_rng= np.random.default_rng(seed)
    test_rng =  np.random.default_rng(seed+2000)
    model_nn_rng = np.random.default_rng(seed+4000)

    discount_f= np.exp(-r*dt)
    # 
    hurst2=2*hurst
    time_s=time_[:,None]

    var_ = (time_s[1:]**hurst2 + time_[1:]**hurst2 - np.abs(time_[1:]-time_s[1:])**hurst2)/2
    B= np.linalg.cholesky(var_)
    Z = train_rng.normal(size=(steps,traj, d))
    S =  np.vstack((np.zeros((1, traj, d)), np.tensordot(B, Z, axes=(1,0)))).transpose(1, 0, 2)
    
    ## ALTERNATIVE SIMULATING fBm but takes longer and requires fbm package, from fbm import FBM; generator = FBM(n=steps, hurst=hurst, length=T, method='daviesharte')
    # S2=[generator.fbm() for sample in range(traj_test_lb)]
    # S3 =[generator.fbm() for sample in range(traj_test_ub)]
    S2 = np.vstack((np.zeros((1, traj_test_lb, d)), np.tensordot(B, test_rng.standard_normal((steps,traj_test_lb, d)), axes=(1,0)))).transpose(1, 0, 2)

    ### Finer samples ###
    steps_fine = steps*grid
    dt_fine = T/(steps_fine)
    time_fine = np.arange(0,T+0.5*dt_fine, dt_fine, dtype=np.float32)
    time_s_fine=time_fine[:,None]
    var_fine = (time_s_fine[1:]**hurst2 + time_fine[1:]**hurst2 - np.abs(time_fine[1:]-time_s_fine[1:])**hurst2)/2
    B_fine=np.linalg.cholesky(var_fine)

    dW_testingub = test_rng.normal(size=(steps_fine,traj_test_ub, d))
    S4 =   np.vstack((np.zeros((1, traj_test_ub, d)), np.tensordot(B_fine, dW_testingub, axes=(1,0)))).transpose(1, 0, 2)
    # Clear from RAM:
    del B_fine
    B_fine= None

    discount_f= np.exp(-r*dt)
    payoff_fracBM = lambda x: np.max(x) if x.ndim==1 else np.max(x[:,0], -1) if x.ndim==2 else  np.max(x[:,0] ,-1)
    payoff_option =payoff_fracBM

    K_lower= K_low
    K_upper = K_up*d
    mode_= 0
    input_size_lb= (1)*(d)
    input_size_ub=(1)*(d)
    model_= model_Schoenmakers(model_nn_rng, seed, input_size_lb, K_lower, steps,d, K_upper, input_size_ub, L=1)
    inner=grid
    
    M_incr=np.zeros((traj_test_ub, steps))
    if step_inner is True:
        step_inner = 2500*100 // traj_test_ub
    else:
        step_inner=inner
    inner_ = np.arange(0, inner+step_inner, step_inner, dtype=int)
    inner_[inner_>=inner] = inner
    inner_=np.unique(inner_)
    inner_ = [inner_[i: i+2] for i in range(len(inner_)-1)]

 
    #### TRAINING #########
    delta_SZH_stopping = np.max(S[:,-1,:], -1)*discount_f**steps
    stop_val_testing= np.max(S2[:,-1,:], -1)*discount_f**steps
    t0_training=datetime.now()
    for time in range(steps)[::-1]:    
        print(time)
        ### TRAINING
        underlying = S[:,time::-1,:]
        reg_m= underlying.reshape(traj_est, (time+1)*d)#np.hstack((underlying, payoff_underlying[:,None]))
        payoff_underlying=payoff_option(underlying)*discount_f**time
        con_val = model_.train_finallayer_continuationvalue(reg_m,delta_SZH_stopping, time, traj, Z[time], dt, mode_)
        xi_=model_.prediction_Z_model_upper(reg_m, traj, time, Z[time])        
        delta_SZH_stopping = np.where(payoff_underlying<con_val,  delta_SZH_stopping-xi_, payoff_underlying)
  
    t1_training=datetime.now()
    time_training.append(t1_training-t0_training)        
    
    ## TESTING (Lowerbound)
    stop_times=np.repeat(steps,traj_test_lb)   
    for time in range(steps)[::-1]:
        ## TESTING (Lowerbound)
        underlying_test = S2[:,time::-1,:]
        cur_payoff_testing = payoff_option(underlying_test)*discount_f**(time)
        reg_m_testing= underlying_test.reshape(traj_test_lb, (time+1)*d)#np.hstack((underlying_test, cur_payoff_testing[:,None]))
        con_val_testing= model_.prediction_conval_model1(reg_m_testing, traj_test_lb, time)
        stop_val_testing = np.where(cur_payoff_testing<con_val_testing, stop_val_testing, cur_payoff_testing)
        stop_times = np.where(cur_payoff_testing<con_val_testing,stop_times, time)
    mean_stop_times=np.mean(stop_times)   
    std_stop_times=np.std(stop_times)

    lowerbound = np.mean(stop_val_testing)
    lowerbound_std = np.std(stop_val_testing)/ (traj_test_lb**0.5)


    #CONSTRUCT MARTINGALE
    discount_f_time = np.exp(-r)
    discount_f_fine = np.exp(-r*dt_fine)

    t0_fineincrement =datetime.now()
    M_incr=np.zeros((traj_test_ub, steps_fine))
    ## Loop
    for t_fine in range(steps_fine):
        print(t_fine)
        underlying_upperbound_test = S4[:,t_fine::-1,:] #if (t_fine+1)//grid<steps_fine else S4[:,t_fine,:]
        # cur_payoff_ub_test = payoff_option(underlying_upperbound_test)*discount_f_fine**t_fine 
        reg_m_=underlying_upperbound_test.reshape(traj_test_ub, (t_fine+1)*d) # np.hstack((underlying_upperbound_test[:, None], cur_payoff_ub_test[:,None]))
        M_incr[:, t_fine]= model_.prediction_Z_model_upper(reg_m_, traj_test_ub, t_fine//grid, dW_testingub[t_fine,:, :])#M_incr[:,ex_right, t_fine]= model_.prediction_Z_model_upper(reg_m_, traj_test_ub, t_fine//grid, dW_testingub[:,t_fine,:], ex_right)
    t1_fineincrement =datetime.now()
    time_ub.append(t1_fineincrement - t0_fineincrement)
    
    grid_consideration = np.arange(0, steps_fine+.5*grid, grid, dtype=int)
    stopping_process= np.max(S4, -1)*discount_f_time**time_fine
    M=np.hstack((np.zeros((traj_test_ub,1)), np.cumsum(M_incr, axis=-1)))[:, grid_consideration]
    M_incr= np.diff(M, axis=-1)
    S4=S4[:,grid_consideration, :]
    stopping_process=stopping_process[:,grid_consideration]

    max_traj_minus_martingale= np.max(stopping_process - M, axis=1)
    upperbound = np.mean(max_traj_minus_martingale)
    upperbound_std = np.std(max_traj_minus_martingale)/ (traj_test_ub**0.5)

    #### TESTING - UB ####
    exp_max = np.maximum.accumulate((stopping_process-M)[:, ::-1], axis=1)[:, ::-1]
    U = exp_max +M

    U_w1 = np.maximum(np.max(S4[:,-1,:],-1)*discount_f**(steps), U[:,-1] )-M[:,-1]
    for t_rough in range(steps)[::-1]:
        underlying_upperbound_test = S4[:,t_rough::-1,:]
        cur_payoff_ub_test = payoff_option(underlying_upperbound_test)*discount_f**(t_rough)
        reg_m_=underlying_upperbound_test.reshape(traj_test_ub, (t_rough+1)*d)
        con_val = model_.prediction_conval_model1(reg_m_, traj_test_ub, t_rough)
        ind_payoffnow=(con_val<=cur_payoff_ub_test)
        U_w1= np.where(ind_payoffnow, np.maximum(cur_payoff_ub_test, U[:, t_rough])-M[:,t_rough], U_w1)

    upperbound2 = np.mean(U_w1)
    upperbound_std2= np.std(U_w1)/(traj_test_ub)**.5

    # Control variate Martingale
    stop_val_testingcv = np.max(S4[:,-1,:], -1)*discount_f**steps-M[:,-1] #payoff_option()*discount_f**steps-M[:,-1]
    for time in range(steps)[::-1]:
        ## TESTING (Lowerbound CV)
        underlying_testCV = S4[:,time::-1,:]
        cur_payoff_testingCV = payoff_option(underlying_testCV)*discount_f**(time)
        reg_m_testingCV=underlying_testCV.reshape(traj_test_ub, (time+1)*d)
        con_val_testingCV= model_.prediction_conval_model1(reg_m_testingCV, traj_test_ub, time)
        stop_val_testingcv = np.where(cur_payoff_testingCV<con_val_testingCV, stop_val_testingcv, cur_payoff_testingCV-M[:,time])
    CV_lowerbound=np.mean(stop_val_testingcv)
    CV_lowerbound_std= np.std(stop_val_testingcv)/(traj_test_ub**0.5)
    if print_progress:
        print('Lowerbound')
        print('Value', lowerbound)
        print('Std',lowerbound_std)
        print('Stop time-mean', mean_stop_times)
        print('Stop time-std', std_stop_times)

        print('')
        print('Upperbound')
        print('up', upperbound)
        print('std',upperbound_std)
        print('up2', upperbound2)
        print('std2', upperbound_std2)

        print('CV est',CV_lowerbound )
        print('CV std', CV_lowerbound_std)
        print('time avg training', np.mean(np.array(time_training)))
        print('time avg ub', np.mean(np.array(time_ub)))
    return lowerbound, lowerbound_std, upperbound, upperbound_std,  np.mean(np.array(time_training)), np.mean(np.array(time_ub)), CV_lowerbound, CV_lowerbound_std, upperbound2, upperbound_std2




information=[]
if __name__=='__main__':
    for d,H in [ (1,0.45)]:
        for grid in [1]:
            print(''.join(['*' for j in range(10)]), grid ,''.join(['*' for j in range(10)]))
            for i in range(1):                
                print(''.join(['-' for j in range(10)]), i , ''.join(['-' for j in range(10)]))
                list_inf=main(d, True, grid=grid, K_low=200,K_up=70, traj_est=200000, traj_test_ub=1000, traj_test_lb=50000, hurst=H, seed=i+8, steps=9)
                label_ = 'SM LS'
                inf_cols = [d, H, '', '', '', '']
                inf_list=utils.process_function_output(*list_inf, label_ = label_, grid= grid, info_cols=inf_cols)
                information.append(inf_list)

    # with open(f'run{datetime.now().strftime("%Y%m%d%H%m%S")}.pic', 'wb') as fh:
    #     pic.dump(information, fh)
   
    table_ = tabulate(utils.information_format_fbm(information), headers=utils.header_fbm, tablefmt="latex_raw", floatfmt=".4f")
    print(table_)
    # folder_txt_log = '/content/drive/MyDrive/'#Tilburg/msc/Thesis/Log'
    # fh = open(f'logresults.txt', 'a')
    # fh.write(f'{datetime.now()}\n ')
    # fh.writelines(table_)
    # line="".join(np.repeat('*',75))
    # fh.write(f'\n {line} \n')
    # fh.close()