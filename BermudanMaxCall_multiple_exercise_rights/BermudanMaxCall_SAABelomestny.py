"""
File which executes the non-linear pure dual upper bound approach by Belomestny (2013) to pricing a Bermudan Max Call option, with possibly multiple exercise rights.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import utils

from tensorflow import keras, constant_initializer, compat, random as random_tf
import numpy as np
from  BermudanMaxCall_model import model_SAA
compat.v1.logging.set_verbosity(compat.v1.logging.ERROR)
import pickle as pic
from tabulate import tabulate
from datetime import datetime
import gurobipy as gb

time_training=[]
time_testing=[]
time_ub=[]
def main(d=3, L=3, print_progress=True, steps=9,T=3, r=0.05, delta_dividend=0.1, traj_est=80000, grid=100, mode_kaggle=False, traj_test_lb=150000, traj_test_ub=10000, traj_est_ub=20000, K_low=200, K_up=200, K_noise=None, S_0=110, strike=100, seed=0, lambda_=1, p=100, g_tol=0.001, payoff_=lambda x, strike: utils.payoff_maxcal(x, strike)):
    """
    Main function, which executes the Belomestny (2013) approach.


    Function also calculates a lower biased estimate based on primal LSMC with randomised neural network.

    Input:
        d: dimension of the fractional brownian motion. Stopping maximum out of d. Only d=1 is considered in report.
        L: Number of exercise rights of the Bermudan Max Call option.
        print_progress: If True: printing results at the end. If False: Only printing times during loops execution algoritm.
        steps: N_T in report, number of possible future stopping dates.
        T: Time of the final possible stopping date.
        r: Interest rate.
        delta_dividend: The dividend rate on each stock in the underlying. 
            Interest rate r is fixed to 0.05; volatility is fixed to 0.2.
        traj_est: Trajectories used for estimation of primal LSMC.
        grid: Number of inner simulations.
        mode_kaggle: Boolean. Set True if running on cloud, as need a cloud license for gurobi. If False, using license on local machine. 
            Also used to avoid loops to calculate martingale increments-> cloud has much more RAM.
        traj_test_lb: Number of testing trajectories for primal LSMC.
        traj_est_ub: Number of training trajectories for the Dual problem.
        traj_test_ub: Number of testring trajectories to evaluate upper bound.

        K_low: Number of nodes,=#basis functions -1, in randomised neural network primal LSMC.
        K_up:  Number of nodes in dual randomised neural network, =# basis functions to construct martingale family.
        S_0: Time-0 stock price of all stocks in the underlying.
        strike: Strike price of the Bermudan Max Call option.
        seed: Seed for randomised neural network and simulating trajectories.
        lambda_: Lambda parameter in Belomestny (2013). Determines weight on minimising empirical variance of dual pathwise maxima.
        p: Smoothening parameter in Belomestny (2013) and Dickmann (2014) (referred to as glättungs parameter in his code). Default set to 100.
        g_tol: Gradient tolerance L_2 norm for stopping optimisation (also limit on 500 iterations). Default 0.001. 
        payoff_: In principal the code works for all payoff functions (so not just a max call, could in principal implement basket-option or min-put etc.).
            Nevertheless, default set to max call and kept to max call: payoff_= lambda x, strike: utils.payoff_maxcal(x, strike))
    Output:
        lowerbound: Lower biased estimate
        lowerbound_std: standard error of lower biased estimate 
        upperbound: Upper biased estimate
        upperbound_std: Standard error of upper biased estimate
        np.mean(np.array(time_training)): Training time (list with 1 value so automatically total time)
        np.mean(np.array(time_ub)):  Testing time (list with 1 value so automatically total time)
        CV_lowerbound: Lower biased estimate using constructed martingale as control variate.
        CV_lowerbound_std: Standard error of lower biased estimate using constructed martingale as control variate.
    
    """

    time_training=[]
    time_testing=[]
    time_ub=[]
    utils.set_seeds(seed)

    dt = T/steps
    traj=traj_est
    sigma = 0.2
    time_ = np.arange(0,T+0.5*dt, dt)
    


    train_rng= np.random.default_rng(seed)
    test_rng =  np.random.default_rng(seed+2000)
    model_nn_rng = np.random.default_rng(seed+4000)
    sim_s = .5*dt
    S_0_train=S_0 *np.exp( train_rng.normal(size=(traj, 1, d))*sigma*(sim_s)**.5 - .5*sigma**2*sim_s)
    discount_f= np.exp(-r*dt)
    dWS= train_rng.normal(size=(traj, steps, d)).astype(np.float32)*((dt)**0.5)
    S = S_0_train*np.exp( np.cumsum(np.hstack((np.zeros((traj,1,d)), dWS*sigma)), axis=1) + np.repeat( (r - delta_dividend - sigma**2/2)*time_, d).reshape(steps+1,d))
    S2 = S_0*np.exp( np.cumsum(np.hstack((np.zeros((traj_test_lb,1,d)), test_rng.normal(size=(traj_test_lb, steps, d)).astype(np.float32)*((dt)**0.5)*sigma)), axis=1) + np.repeat( (r- delta_dividend - sigma**2/2)*time_, d).reshape(steps+1,d))
    S3 = S_0*np.exp( np.cumsum(np.hstack((np.zeros((traj_est_ub,1,d)),  train_rng.normal(size=(traj_est_ub, steps, d)).astype(np.float32)*((dt)**0.5)*sigma)), axis=1) + np.repeat( (r- delta_dividend - sigma**2/2)*time_, d).reshape(steps+1,d))
    S4 = S_0*np.exp( np.cumsum(np.hstack((np.zeros((traj_test_ub,1,d)),  test_rng.normal(size=(traj_test_ub, steps, d)).astype(np.float32)*((dt)**0.5)*sigma)), axis=1) + np.repeat( (r- delta_dividend - sigma**2/2)*time_, d).reshape(steps+1,d))


    discount_f= np.exp(-r*dt)

    payoff_option = lambda x: payoff_(x, strike)

    K_lower= K_low

    input_size_lb= (1)*(d+1)
    input_size_ub=(1)*(d+1)

    model_ = model_SAA(model_nn_rng, seed, input_size_lb, K_lower, steps, d, K_up, input_size_ub, L=L, K_noise=K_noise, mode_kaggle=mode_kaggle)
    stop_val = payoff_option(S[:,-1,:])*discount_f**steps
    stop_val_testing = payoff_option(S2[:,-1,:])*discount_f**steps
    if mode_kaggle:
        step_size=traj_test_ub
    else:
        step_size = (1000*150*600)//K_low//grid
    inner_ = np.arange(0, traj_test_ub+step_size, step_size, dtype=int)
    inner_[inner_>=traj_test_ub] = traj_test_ub
    inner_= list(np.unique(inner_))
    inner_ = [inner_[i: i+2] for i in range(len(inner_)-1)]
    create_loop_grid = lambda traj: list(np.unique(np.clip(np.arange(0, traj+step_size, step_size, dtype=int), 0, traj)))
    create_loop_grid_slice = lambda traj: [slice for slice in zip(create_loop_grid(traj), create_loop_grid(traj)[1:])]
    stop_val_archive= np.zeros((traj_est, steps+1, 2))
    #### TRAINING #########
    t0_training=datetime.now()
    for ex_right in range(L):
        stop_val_archive=stop_val_archive[:,:,::-1] # Update stop_val_archive to set previous ex_right level to position stop_val_archive[:,:,0] and write new level at stop_val_archive[:,:,1]
        print('right=',ex_right)
        # final_time_exercising= steps-ex_right
        stop_val = payoff_option(S[:,-1,:])*discount_f**steps
        stop_val_archive[:, -1, 1] = np.copy(stop_val)
        for time in range(steps)[::-1]:
            print('t=',time)
            underlying = S[:,time,:]
            payoff_underlying = payoff_option(underlying)*discount_f**time
            reg_m=np.hstack((underlying, payoff_underlying[:,None]))
            prev_right_c = stop_val_archive[:,time+1, 0]
            con_val = model_.train_finallayer_continuationvalue(reg_m,  stop_val, time, traj, dWS[:,time,:], ex_right)#stop_val_archive[:,time+1, 1]
            con_val_no_ex = model_.prediction_conval_model1(reg_m, traj_est, time, ex_right-1) if ex_right>0 else 0
            ex_ind = (payoff_underlying+con_val_no_ex>con_val)|(steps-time<=ex_right) # (steps-time<=ex_right): due to nonnegative payoff
            stop_val = np.where(ex_ind, payoff_underlying+prev_right_c, stop_val) #stop_val_archive[:,time+1, 1]) # stop_val)
            stop_val_archive[:,time, 1]=np.copy(stop_val)
    t1_training=datetime.now()
    time_training.append(t1_training-t0_training)  

    S = S_0*np.exp( np.cumsum(np.hstack((np.zeros((traj,1,d)), dWS*sigma)), axis=1) + np.repeat( (r - delta_dividend - sigma**2/2)*time_, d).reshape(steps+1,d)) # start from time-0 in optimisation problems rather than -1/2.
    #### TESTING - LB ####
    stop_val_testing=0
    prev_stop_time=-1  
    mean_stop_times=np.zeros(L)
    std_stop_times=np.zeros(L)  
    for ex_right in range(L):
        print('right=',ex_right)
        stop_time=steps+1-(L-ex_right)
        stop_val_testing_round=payoff_option(S2[:,stop_time,:])*discount_f**stop_time
        for time in range(steps)[-(L-ex_right)::-1]:
            underlying_test = S2[:,time,:]
            cur_payoff_testing = payoff_option(underlying_test)*discount_f**time
            reg_m_testing=np.hstack((underlying_test, cur_payoff_testing[:,None]))
            con_val_no_ex = model_.prediction_conval_model1(reg_m_testing, traj_test_lb, time, L-1-ex_right)      
            con_val_ex= model_.prediction_conval_model1(reg_m_testing, traj_test_lb, time, L-2-ex_right) if L-1-ex_right>0 else 0
            ex_ind = ((cur_payoff_testing+con_val_ex>=con_val_no_ex) & (time> prev_stop_time))
            stop_val_testing_round = np.where(ex_ind, cur_payoff_testing, stop_val_testing_round)
            stop_time = np.where(ex_ind, time, stop_time)
        mean_stop_times[ex_right]=np.mean(stop_time)
        std_stop_times[ex_right]=np.std(stop_time)
        prev_stop_time = np.copy(stop_time)
        stop_val_testing+=stop_val_testing_round

    ### Constructing family of martingale increments
    M2_train_basisfunctions = np.zeros((traj_est_ub, steps, K_up))
    M2_test_basisfunctions = np.zeros((traj_test_ub, steps, K_up))
    t0_ub = datetime.now()
    for time in range(steps):  
            cur_payoff_t_p1 = payoff_option(S3[:, time+1,:])*discount_f**(time+1)
            reg_m_t_p1= np.hstack((S3[:, time+1,:], cur_payoff_t_p1[:,None]))
            traj_inner_ = np.exp( sigma*np.random.randn(traj_est_ub, d, grid)*(dt)**.5  + (r-delta_dividend-sigma**2/2)*dt)*(S3[:,time,:])[:,:,None]
            cur_payoff_inner = payoff_option(np.swapaxes(traj_inner_, 1,2))*discount_f**(time+1) 
            reg_m_t_sim_p1 = np.hstack((traj_inner_, cur_payoff_inner[:,None, :])).transpose(0, 2, 1)
            reg_m_inner = reg_m_t_sim_p1.reshape(-1, reg_m_t_sim_p1.shape[-1])
            basis_f_inner= []
            inner_train = create_loop_grid_slice(traj_est_ub)
            for index_start, index_end in inner_train:
                slice_length = index_end-index_start
                exp_basis_func_tp1_inner= np.mean(model_.random_basis_LS_upper(reg_m_inner[index_start*grid:index_end*grid,:]).reshape(slice_length,grid, K_up), axis=1)
                basis_f_inner.append(exp_basis_func_tp1_inner)
            exp_basis_func_tp1_inf_t=np.vstack((basis_f_inner))
            M2_train_basisfunctions[:,time, :] = model_.random_basis_LS_upper(reg_m_t_p1) - exp_basis_func_tp1_inf_t

            cur_payoff_t_p1 = payoff_option(S4[:, time+1,:])*discount_f**(time+1)
            reg_m_t_p1= np.hstack((S4[:, time+1,:], cur_payoff_t_p1[:,None]))
            traj_inner_ = np.exp( sigma*np.random.randn(traj_test_ub, d, grid)*(dt)**.5  + (r-delta_dividend-sigma**2/2)*dt)*(S4[:,time,:])[:,:,None]
            cur_payoff_inner = payoff_option(np.swapaxes(traj_inner_, 1,2))*discount_f**(time+1) 
            reg_m_t_sim_p1 = np.hstack((traj_inner_, cur_payoff_inner[:,None, :])).transpose(0, 2, 1)
            reg_m_inner = reg_m_t_sim_p1.reshape(-1, reg_m_t_sim_p1.shape[-1])
            basis_f_inner= []
            inner_test = create_loop_grid_slice(traj_test_ub)
            for index_start, index_end in inner_test:
                slice_length = index_end-index_start
                exp_basis_func_tp1_inner= np.mean(model_.random_basis_LS_upper(reg_m_inner[index_start*grid:index_end*grid,:]).reshape(slice_length,grid, K_up), axis=1)
                basis_f_inner.append(exp_basis_func_tp1_inner)
            exp_basis_func_tp1_inf_t=np.vstack((basis_f_inner))
            M2_test_basisfunctions[:,time, :] = model_.random_basis_LS_upper(reg_m_t_p1) - exp_basis_func_tp1_inf_t
    
    payoff_process_manipulated = payoff_option(S3)*discount_f**np.arange(steps+1) # Desai et al. 2012
    r_opt, theta_ub_nonsmoothened, fun_smoothened, solver_time, status_= model_.LP_BELOM_BFGS_multiple(payoff_process_manipulated, M2_train_basisfunctions, print_progress=True, lambda_=lambda_, p=p, calc_nonsmoothenedTrainingUB=False, ridge_penalty=1/100, g_tol=g_tol)
    if 'gradient' not in status_.lower():
        print(np.repeat('!', 50))
        print('problem, Number iterations was used as stopping criterion')
        print(np.repeat('!', 50))
    print('time solver', solver_time)
    ##################################

    #CONSTRUCT MARTINGALE
    steps_fine=steps
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
    terminal_payoff=payoff_option(S4[:,-1,:])*np.exp(-r*T)
    theta_upperbound= np.zeros((traj_test_ub, L, steps+1))
    theta_upperbound[:,:,-1]=terminal_payoff[:,None]
    for ex_right in range(L):
        print('right=',ex_right)
        theta_next_samelevel=terminal_payoff
        for time in range(steps)[::-1]:
            underlying_test = S4[:,time,:]
            cur_payoff_testing = payoff_option(underlying_test)*discount_f**time
            theta_next_prevlevel= theta_upperbound[:,ex_right-1, time+1] if ex_right>0 else 0
            M_incr_prevlevel= M_incr[:,ex_right-1,time] if ex_right>0 else 0
            M_incr_samelevel= M_incr[:,ex_right,time]
            theta_next_samelevel= np.maximum(cur_payoff_testing - M_incr_prevlevel + theta_next_prevlevel, -M_incr_samelevel +theta_next_samelevel)  if steps-ex_right>time else cur_payoff_testing - M_incr_prevlevel + theta_next_prevlevel
            theta_upperbound[:, ex_right, time] = np.copy(theta_next_samelevel)

    ## LOWERBOUND
    lowerbound= np.mean(stop_val_testing)
    lowerbound_std = np.std(stop_val_testing)/(traj_test_lb)**.5

    #CONSTRUCT MARTINGALE
    Dual_max_traj=theta_upperbound[:, -1,0]
    upperbound = np.mean(Dual_max_traj)
    upperbound_std= np.std(Dual_max_traj)/(traj_test_ub)**.5
    M=np.dstack((np.zeros(M_incr.shape[:2])[:,:,None], M_incr ))
    M= np.cumsum(M, axis=-1)

    #### CONTROL VARIATE#####
    stop_val_testingcv=0
    prev_stop_timeCV=np.repeat(-1, traj_test_ub)  
    for ex_right in range(L):
        print('right=',ex_right)
        stop_timeCV= steps+1-(L-ex_right)
        stop_val_testingCV_round=payoff_option(S4[:,stop_timeCV,:])*discount_f**stop_timeCV 
        for time in range(steps)[-(L-ex_right)::-1]:
            underlying_test = S4[:,time,:]
            cur_payoff_testing = payoff_option(underlying_test)*discount_f**time
            reg_m_testing=np.hstack((underlying_test, cur_payoff_testing[:,None]))
            con_val_no_ex = model_.prediction_conval_model1(reg_m_testing, traj_test_ub, time, L-1-ex_right)      
            con_val_ex= model_.prediction_conval_model1(reg_m_testing, traj_test_ub, time, L-2-ex_right) if L-1-ex_right>0 else 0
            ex_ind = ((cur_payoff_testing+con_val_ex>=con_val_no_ex) & (time> prev_stop_timeCV))
            stop_val_testingCV_round = np.where(ex_ind, cur_payoff_testing, stop_val_testingCV_round)
            stop_timeCV = np.where(ex_ind, time, stop_timeCV)
        
        samples_array=np.arange(traj_test_ub)
        m_incr_level = M[samples_array, L-1-ex_right,stop_timeCV] - M[samples_array, L-1-ex_right,prev_stop_timeCV] if ex_right>0 else M[samples_array, L-1-ex_right,stop_timeCV]
        stop_val_testingcv+=stop_val_testingCV_round -m_incr_level
        prev_stop_timeCV = np.copy(stop_timeCV)
    

    CV_lowerbound=np.mean(stop_val_testingcv)
    CV_lowerbound_std= np.std(stop_val_testingcv)/(traj_test_ub**0.5)
    if print_progress:
        print(np.mean(M[:,-3:],axis=0))
        print('Lowerbound')

        print('Value', lowerbound)
        print('Std',lowerbound_std)
        print('Stop time-mean', mean_stop_times)
        print('Stop time-std', std_stop_times)
        

        print('Upperbound')
        print('up', upperbound)
        print('std',upperbound_std)
        # print('up2', upperbound2)
        # print('std2', upperbound_std2)

        print('CV est',CV_lowerbound )
        print('CV std', CV_lowerbound_std)
        # print('time avg testing', np.mean(np.array(time_testing)))
        print('time avg training', np.mean(np.array(time_training)))
        print('time avg ub', np.mean(np.array(time_ub))) #  np.mean(np.array(time_training))
    return lowerbound, lowerbound_std, upperbound, upperbound_std, np.mean(np.array(time_training)), np.mean(np.array(time_ub)), CV_lowerbound, CV_lowerbound_std


information=[]
if __name__=='__main__':
    for d,s0,n_stopping_rights, lambda_i in [ (2, 90, 2, 0), (2, 90, 2, 1), (2, 90, 2, 1/20), (2, 90, 2, 1/3), (2, 90, 2, 1/6)]:# (2,90, 1, 0),(2, 90, 1, 1), (2,90,1, 2), (2, 90, 1, 3), (2,90, 1, 20)
        for grid in [600]:
            print(''.join(['*' for j in range(10)]), grid ,''.join(['*' for j in range(10)]))
            for i in range(1):                
                print(''.join(['-' for j in range(10)]), i ,''.join(['-' for j in range(10)]))
                list_inf = main(d, n_stopping_rights, True, grid=grid, K_low=150, K_up=450, traj_est=10000, traj_test_ub=20000, traj_est_ub=8000, traj_test_lb=200000, S_0=s0, seed=i+8, lambda_= lambda_i, p=100) 
                label_= f'SAA {lambda_i}'
                inf_cols = [d, s0, n_stopping_rights, '', '', '', '']
                inf_list=utils.process_function_output(*list_inf, label_ = label_, grid= grid, info_cols=inf_cols)
                information.append(inf_list)

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