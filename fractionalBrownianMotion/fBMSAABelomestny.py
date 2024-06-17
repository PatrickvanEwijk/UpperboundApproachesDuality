"""
File which executes Belomestny (2013) non-linear pure dual upper bound approach to a fractional brownian motion.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import utils

from tensorflow import keras, constant_initializer, compat, random as random_tf
import numpy as np
from modelRrobust2fBM import model_SAA
compat.v1.logging.set_verbosity(compat.v1.logging.ERROR)
import pickle as pic
from tabulate import tabulate
from datetime import datetime


def main(d=1,print_progress=True, steps= 9, T=1, traj_est=80000, grid=100, traj_test_lb=150000,traj_est_ub=8000, traj_test_ub=10000, K_low=200, K_up=100, hurst=0.7, seed=0, mode_kaggle=False, lambda_=1, L=1, p=100):
    """
    Main function, which executes Belomestny (2013) approach.

    Function also calculates a lower biased estimate based on primal LSMC with randomised neural network.

    Input:
        d: dimension of the fractional brownian motion. Stopping maximum out of d. Only d=1 is considered in report.
        print_progress: If True: printing results at the end. If False: Only printing times during loops execution algoritm 
        steps: N_T in report, number of possible future stopping dates.
        T: Time of the final possible stopping date.
        traj_est: Trajectories used for estimation of primal LSMC.
        grid: Number of inner simulations.
        traj_test_lb: Number of testing trajectories for primal LSMC.
        traj_est_ub: Number of training trajectories for the Dual problem.
        traj_test_ub: Number of testring trajectories to evaluate upper bound.
        K_low: Number of nodes,=#basis functions -1, in randomised neural network primal LSMC.
        K_up:  Number of nodes in dual randomised neural network, =# basis functions to construct martingale family.
        hurst: Hurst parameter of fbm.
        seed: Seed for randomised neural network and simulating trajectories.
        mode_kaggle: Boolean.
            Used to avoid loops to calculate martingale increments-> cloud has much more RAM.
        lambda_: Lambda parameter in Belomestny (2013). Determines weight on minimising empirical variance of dual pathwise maxima.
        L=1: NOT IMPLEMENTED, restricted to 1. Number of stopping rights.
        p: Smoothening parameter in Belomestny (2013) and Dickmann (2014) (referred to as glättungs parameter in his code). Default set to 100.

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
    traj_train_ub=traj_est_ub
    time_ = np.arange(0,T+0.5*dt, dt)
    r=0.0

    train_rng= np.random.default_rng(seed)
    test_rng =  np.random.default_rng(seed+2000)
    model_nn_rng = np.random.default_rng(seed+4000)

    discount_f= np.exp(-r*dt)
    # generator = FBM(n=steps, hurst=hurst, length=T, method='daviesharte')
    hurst2=2*hurst
    time_s=time_[:,None]
    var_ = (time_s[1:]**hurst2 + time_[1:]**hurst2 - np.abs(time_[1:]-time_s[1:])**hurst2)/2
    B=np.linalg.cholesky(var_)
    Z= train_rng.normal(size=(steps,traj, d))

    S = np.swapaxes(np.vstack((np.zeros((1, traj, d)), np.einsum('sj,jkd->skd', B, Z))), 0,1)
    
    S2 = np.swapaxes(np.vstack((np.zeros((1, traj_test_lb, d)), np.einsum('sj,jkd->skd', B, test_rng.standard_normal((steps,traj_test_lb, d))))), 0,1)
    Z3 = train_rng.normal(size=(steps,traj_train_ub, d))
    S3 = np.swapaxes(np.vstack((np.zeros((1, traj_train_ub, d)), np.einsum('sj,jkd->skd', B, Z3))), 0,1)

    Z4 = test_rng.normal(size=(steps,traj_test_ub, d))
    S4 = np.swapaxes(np.vstack((np.zeros((1, traj_test_ub, d)), np.einsum('sj,jkd->skd', B, Z4))), 0,1)

    discount_f= np.exp(-r*dt)

    payoff_fracBM = lambda x: np.max(x) if x.ndim==1 else np.max(x[:,0], -1) if x.ndim==2 else  np.max(x[:,0] ,-1)
    payoff_option =payoff_fracBM


    K_lower= K_low
    K_upper = K_up


    input_size_lb= (1)*(d)
    input_size_ub=(1)*(d)

    model_ = model_SAA(model_nn_rng, seed, input_size_lb, K_lower, steps, d, K_upper, input_size_ub, L=1, K_noise=None, mode_kaggle=mode_kaggle)
    inner=grid
    
    M_incr=np.zeros((traj_test_ub, steps))
    if mode_kaggle:
        step_size=traj_test_ub
    else:
        step_size = (1000*150*500)//K_up//grid
    create_loop_grid = lambda traj: list(np.unique(np.clip(np.arange(0, traj+step_size, step_size, dtype=int), 0, traj)))
    create_loop_grid_slice = lambda traj: [slice for slice in zip(create_loop_grid(traj), create_loop_grid(traj)[1:])]
    #### TRAINING #########
    stop_val = S[:,-1,0]*discount_f**steps
    stop_val_testing= S2[:,-1,0]*discount_f**steps
    t0_training=datetime.now()
    for time in range(steps)[::-1]:    
        print(time)
        underlying = S[:,time::-1,:]
        reg_m= underlying.reshape(traj_est, (time+1)*d) #np.hstack((underlying, payoff_underlying[:,None]))
        payoff_underlying=payoff_option(underlying)*discount_f**time
        con_val = model_.train_finallayer_continuationvalue(reg_m,stop_val, time, traj, Z[time])        
        stop_val = np.where(payoff_underlying<con_val, stop_val, payoff_underlying)
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

   ### Constructing family of martingale increments
    M2_train_basisfunctions = np.zeros((traj_train_ub, steps, K_up))
    M2_test_basisfunctions = np.zeros((traj_test_ub, steps, K_up))
    t0_ub = datetime.now()
    for time in range(steps):
        for paths_t, paths_t_p1, M, Z_t, traj_ in [(S3[:,:time+1,:],S3[:, :time+2,:], M2_train_basisfunctions, Z3[:time, :, :], traj_train_ub), (S4[:,:time+1,:],S4[:, :time+2,:], M2_test_basisfunctions,  Z4[:time, :, :], traj_test_ub)]:
            underlying_upperbound = paths_t[:,::-1,:]
            underlying_upperbound_next = paths_t_p1[:,::-1,:]
            reg_m_t_p1= underlying_upperbound_next.reshape(traj_,(time+2)*d)#np.hstack((paths_t_p1, cur_payoff_t_p1[:,None]))
        
            z_inner = np.random.randn(traj_, d,inner)
            traj_inner = np.einsum('t, tid->id', B[time, :time], Z_t)[:,:,None] + B[time,time] * z_inner # (traj_ub, d, inner) vector with realisations of next period underlying
            reg_m_inner= [np.hstack(((traj_inner[:,:,i])[:,None,:], underlying_upperbound)).reshape(traj_, (time+2)*d) for i in range(inner)] # stack underlying and previous underlying trajectory -> time t+1 : 0 trajectory used to predict continuation value.
            reg_m_inner=np.dstack(reg_m_inner).transpose(0,2,1).reshape(traj_*inner, (time+2)*d)
            
            basis_f_inner= []
            inner_= create_loop_grid_slice(traj_)
            for index_start, index_end in inner_:
                slice_length = index_end-index_start
                exp_basis_func_tp1_inner= np.mean(model_.random_basis_LS_upper(reg_m_inner[index_start*grid:index_end*grid,:], time+1).reshape(slice_length,grid, K_up), axis=1)
                basis_f_inner.append(exp_basis_func_tp1_inner)
            exp_basis_func_tp1_inf_t=np.vstack((basis_f_inner))
            M[:,time, :] = model_.random_basis_LS_upper(reg_m_t_p1, time+1) - exp_basis_func_tp1_inf_t

    

    payoff_process_manipulated = np.max(S3, axis=-1)*discount_f**np.arange(steps+1) 
    r_opt, theta_ub_nonsmoothened, fun_smoothened, solver_time= model_.LP_BELOM_BFGS_multiple(payoff_process_manipulated, M2_train_basisfunctions, print_progress=True, lambda_=lambda_, p=p, calc_nonsmoothenedTrainingUB=True, ridge_penalty=1/100)
    print(fun_smoothened)
    ##################################

    #CONSTRUCT MARTINGALE
    payoff_process_= np.max(S4, -1)*discount_f**np.arange(steps+1) 
    steps_fine=steps
    fine_grid=1
    M_incr=np.zeros((traj_test_ub, L, steps_fine))
    for t_fine in range(steps_fine):
        M2_test_bf_t = M2_test_basisfunctions[:,t_fine//1,:]
        for ex_right in range(L):
            M_incr[:,ex_right, t_fine]= M2_test_bf_t@ r_opt[:,ex_right, t_fine]
    t1_ub = datetime.now()
    time_ub.append(t1_ub-t0_ub)
    # M_test = np.cumsum( np.dstack((np.zeros(M_incr.shape[:2]), M_incr)), axis=-1).reshape(M_incr.shape[0], M_incr.shape[-1]+1)
    # ub_testdata= np.mean((payoff_process_-M_test).max(1))
    # print('ub test', ub_testdata)
    #### TESTING - UB ####
    terminal_payoff=np.max(S4[:,-1,:], -1)*np.exp(-r*T)
    theta_upperbound= np.zeros((traj_test_ub, L, steps+1))
    theta_upperbound[:,:,-1]=terminal_payoff[:,None]
    for ex_right in range(L):
        print('right=',ex_right)
        theta_next_samelevel=terminal_payoff
        for time in range(steps)[::-1]:
            underlying_test = S4[:,time::-1,:]
            cur_payoff_testing = payoff_option(underlying_test)*discount_f**time
            theta_next_prevlevel= theta_upperbound[:,ex_right-1, time+1] if ex_right>0 else 0
            M_incr_prevlevel= M_incr[:,ex_right-1,time] if ex_right>0 else 0
            M_incr_samelevel= M_incr[:,ex_right,time]
            theta_next_samelevel= np.maximum(cur_payoff_testing - M_incr_prevlevel + theta_next_prevlevel, -M_incr_samelevel +theta_next_samelevel)  if steps-ex_right>time else cur_payoff_testing - M_incr_prevlevel + theta_next_prevlevel
            theta_upperbound[:, ex_right, time] = np.copy(theta_next_samelevel)
    


    stopping_process=np.max(S4[:,:,:], axis=-1)
    M=np.dstack((np.zeros((traj_test_ub,L,1)), np.cumsum(M_incr, axis=-1)))
    M_2d = M.reshape(M.shape[0], M.shape[-1])
    print(np.mean(M_2d[:,-3:], axis=0))
    max_traj=np.max((stopping_process -M_2d), axis=1)
    upperbound = np.mean(max_traj)
    upperbound_std= np.std(max_traj)/(traj_test_ub)**.5

    ## LOWERBOUND
    lowerbound= np.mean(stop_val_testing)
    lowerbound_std = np.std(stop_val_testing)/(traj_test_lb)**.5
 
    #CONSTRUCT MARTINGALE
    Dual_max_traj=theta_upperbound[:, -1,0]
    upperbound = np.mean(Dual_max_traj)
    upperbound_std= np.std(Dual_max_traj)/(traj_test_ub)**.5
    


    exp_max = np.maximum.accumulate((stopping_process-M_2d)[:, ::-1], axis=1)[:, ::-1]
    U = exp_max +M_2d

    U_w1 = np.maximum(np.max(S4[:,-1,:],-1)*discount_f**(steps), U[:,-1] )-M_2d[:,-1]
    for t_rough in range(steps)[::-1]:
        underlying_upperbound_test = S4[:,t_rough::-1,:]
        cur_payoff_ub_test = payoff_option(underlying_upperbound_test)*discount_f**(t_rough)
        reg_m_=underlying_upperbound_test.reshape(traj_test_ub, (t_rough+1)*d)
        con_val = model_.prediction_conval_model1(reg_m_, traj_test_ub, t_rough)
        ind_payoffnow=(con_val<=cur_payoff_ub_test)
        U_w1= np.where(ind_payoffnow, np.maximum(cur_payoff_ub_test, U[:, t_rough])-M_2d[:,t_rough], U_w1)

    upperbound2 = np.mean(U_w1)
    upperbound_std2= np.std(U_w1)/(traj_test_ub)**.5

    #### CONTROL VARIATE#####
    stop_val_testingcv=0
    prev_stop_timeCV=np.repeat(-1, traj_test_ub)  
    for ex_right in range(L):
        print('right=',ex_right)
        stop_timeCV= steps+1-(L-ex_right)
        stop_val_testingCV_round=payoff_option(S4[:,stop_timeCV::-1,:])*discount_f**stop_timeCV 
        for time in range(steps)[-(L-ex_right)::-1]:
            underlying_test = S4[:,time::-1,:]
            cur_payoff_testing = payoff_option(underlying_test)*discount_f**time
            reg_m_testing=underlying_test.reshape(traj_test_ub, (time+1)*d) # np.hstack((underlying_test, cur_payoff_testing[:,None]))
            con_val_no_ex = model_.prediction_conval_model1(reg_m_testing, traj_test_ub, time, L-1-ex_right)      
            con_val_ex= model_.prediction_conval_model1(reg_m_testing, traj_test_ub, time, L-2-ex_right) if L-1-ex_right>0 else 0
            ex_ind = ((cur_payoff_testing+con_val_ex>=con_val_no_ex) & (time> prev_stop_timeCV))
            stop_val_testingCV_round = np.where(ex_ind, cur_payoff_testing, stop_val_testingCV_round)
            stop_timeCV = np.where(ex_ind, time, stop_timeCV)
        
        samples_array=np.arange(traj_test_ub)
        m_incr = M[samples_array, L-1-ex_right,stop_timeCV] - M[samples_array, L-1-ex_right,prev_stop_timeCV] if ex_right>0 else M[samples_array, L-1-ex_right,stop_timeCV]
        stop_val_testingcv+=stop_val_testingCV_round -m_incr
        prev_stop_timeCV = np.copy(stop_timeCV)
    CV_lowerbound=np.mean(stop_val_testingcv)
    CV_lowerbound_std= np.std(stop_val_testingcv)/(traj_test_ub**0.5)
    if print_progress:
        print('Lowerbound')
        print('Value', lowerbound)
        print('Std',lowerbound_std)
        print('Stop time-mean', mean_stop_times)
        print('Stop time-std', std_stop_times)

        print('Upperbound')
        print('up', upperbound)
        print('std',upperbound_std)
        print('up2', upperbound2)
        print('std2', upperbound_std2)

        print('CV est',CV_lowerbound )
        print('CV std', CV_lowerbound_std)
        print('time avg training', np.mean(np.array(time_training)))
        print('time avg ub', np.mean(np.array(time_ub)))
    return lowerbound, lowerbound_std, upperbound, upperbound_std, np.mean(np.array(time_training)), np.mean(np.array(time_ub)), CV_lowerbound, CV_lowerbound_std, upperbound2, upperbound_std2


information=[]
if __name__=='__main__':
    for d,H in [ (2,0.2), (1,0.3), (1, 0.7)]:
        for grid in [100]:
            print(''.join(['*' for j in range(10)]), grid ,''.join(['*' for j in range(10)]))
            for i in range(1):                
                print(''.join(['-' for j in range(10)]), i , ''.join(['-' for j in range(10)]))
                list_inf=main(d, True, grid=grid, K_low=150,K_up=50, traj_est=100000, traj_test_ub=40000, traj_test_lb=50000, hurst=H, seed=i+8, steps=9, lambda_=1/2)
                label_='HK LS'
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