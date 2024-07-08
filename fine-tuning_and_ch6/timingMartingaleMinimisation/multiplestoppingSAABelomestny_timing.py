import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import utils

from tensorflow import keras, constant_initializer, compat, random as random_tf
import numpy as np
from modelRrobust2MS import model_SAA
compat.v1.logging.set_verbosity(compat.v1.logging.ERROR)
import pickle as pic
from tabulate import tabulate
from datetime import datetime
import gurobipy as gb

time_training=[]
time_testing=[]
time_ub=[]
def main(d=3, L=3,  print_progress=True, traj_est=80000, grid=100, mode_kaggle=False, traj_test_lb=150000,steps= 9, traj_test_ub=10000, traj_est_ub=20000, K_low=200, K_noise=10, S_0=110, strike=100, seed=0, step_size=None, lambda_=1, p=100):
    time_training=[]
    time_testing=[]
    time_ub=[]
    utils.set_seeds(seed)
    
    T=3
    dt = T/steps
    traj=traj_est
    sigma = 0.2
    time_ = np.arange(0,T+0.5*dt, dt)
    r=0.05
    delta_dividend= 0.1

    train_rng= np.random.default_rng(seed)
    test_rng =  np.random.default_rng(seed+2000)
    model_nn_rng = np.random.default_rng(seed+4000)
    sim_s = 3.5*dt
    S_0_train=S_0 *np.exp( train_rng.normal(size=(traj, 1, d))*sigma*(sim_s)**.5 - .5*sigma**2*sim_s)
    discount_f= np.exp(-r*dt)
    dWS= train_rng.normal(size=(traj, steps, d)).astype(np.float32)*((dt)**0.5)
    S = S_0_train*np.exp( np.cumsum(np.hstack((np.zeros((traj,1,d)), dWS*sigma)), axis=1) + np.repeat( (r - delta_dividend - sigma**2/2)*time_, d).reshape(steps+1,d))
    S2 = S_0*np.exp( np.cumsum(np.hstack((np.zeros((traj_test_lb,1,d)), test_rng.normal(size=(traj_test_lb, steps, d)).astype(np.float32)*((dt)**0.5)*sigma)), axis=1) + np.repeat( (r- delta_dividend - sigma**2/2)*time_, d).reshape(steps+1,d))
    S3 = S_0*np.exp( np.cumsum(np.hstack((np.zeros((traj_est_ub,1,d)),  train_rng.normal(size=(traj_est_ub, steps, d)).astype(np.float32)*((dt)**0.5)*sigma)), axis=1) + np.repeat( (r- delta_dividend - sigma**2/2)*time_, d).reshape(steps+1,d))
    S4 = S_0*np.exp( np.cumsum(np.hstack((np.zeros((traj_test_ub,1,d)),  test_rng.normal(size=(traj_test_ub, steps, d)).astype(np.float32)*((dt)**0.5)*sigma)), axis=1) + np.repeat( (r- delta_dividend - sigma**2/2)*time_, d).reshape(steps+1,d))


    discount_f= np.exp(-r*dt)

    payoff_maxcal=  lambda x: np.maximum(np.max(x, axis=-1) - strike,0)
    payoff_basketcall = lambda x: np.maximum(np.mean(x, axis=-1) - strike,0)
    payoff_option =payoff_maxcal

    K_lower= K_low

    input_size_lb= (1)*(d+1)
    input_size_ub=(1)*(d+1)

    model_ = model_SAA(model_nn_rng, seed, input_size_lb, K_lower, steps, d, 0, input_size_ub, L=L, K_noise=K_noise, mode_kaggle=mode_kaggle)
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
        # print('right=',ex_right)
        # final_time_exercising= steps-ex_right
        stop_val = payoff_option(S[:,-1,:])*discount_f**steps
        stop_val_archive[:, -1, 1] = np.copy(stop_val)
        for time in range(steps)[::-1]:
            # print('t=',time)
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

    #### TESTING - LB ####
    stop_val_testing=0
    prev_stop_time=-1    
    for ex_right in range(L):
        # print('right=',ex_right)
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
        prev_stop_time = np.copy(stop_time)
        stop_val_testing+=stop_val_testing_round

    ### Constructing family of martingale increments
    M2_train_basisfunctions = np.zeros((traj_est_ub, steps, K_low))
    M2_test_basisfunctions = np.zeros((traj_test_ub, steps, K_low))
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
                exp_basis_func_tp1_inner= np.mean(model_.random_basis_LS(reg_m_inner[index_start*grid:index_end*grid,:]).reshape(slice_length,grid, K_low), axis=1)
                basis_f_inner.append(exp_basis_func_tp1_inner)
            exp_basis_func_tp1_inf_t=np.vstack((basis_f_inner))
            M2_train_basisfunctions[:,time, :] = model_.random_basis_LS(reg_m_t_p1) - exp_basis_func_tp1_inf_t

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
                exp_basis_func_tp1_inner= np.mean(model_.random_basis_LS(reg_m_inner[index_start*grid:index_end*grid,:]).reshape(slice_length,grid, K_low), axis=1)
                basis_f_inner.append(exp_basis_func_tp1_inner)
            exp_basis_func_tp1_inf_t=np.vstack((basis_f_inner))
            M2_test_basisfunctions[:,time, :] = model_.random_basis_LS(reg_m_t_p1) - exp_basis_func_tp1_inf_t
    
    payoff_process_manipulated = payoff_option(S3)*discount_f**np.arange(steps+1) # Desai et al. 2012
    r_opt, theta_ub_nonsmoothened, fun_smoothened, solver_time= model_.LP_BELOM_BFGS_multiple(payoff_process_manipulated, M2_train_basisfunctions, print_progress=False, lambda_=lambda_, p=p, calc_nonsmoothenedTrainingUB=False, ridge_penalty=1/100)
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
        # print('right=',ex_right)
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
        # print(np.mean(M[:,-3:],axis=0))
        print('Lowerbound')

        print('Value', lowerbound)
        print('Std',lowerbound_std)
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
    return lowerbound, lowerbound_std, upperbound, upperbound_std, solver_time, np.mean(np.array(time_ub)), CV_lowerbound, CV_lowerbound_std

from itertools import product
information=[]
if __name__=='__main__':
    for d,s0,n_stopping_rights in [ (2, 90, 5)]:
        for grid in [200]:
            steps_list = [7,9]
            k_list = [35, 60, 80, 120, 200]
            samples_est_list = [1000,  2000, 2500, 5000, 10000, 15000]
            high_dim_comb=  list(product(steps_list, k_list, samples_est_list))
            low_dim_comb = list(product([5, 7, 8], [5, 10, 20, 30], [i//20 for i in samples_est_list]))
            high_k_dim_comb =[ (step, k, k+j) for step, k in product([7], [500, 600, 900]) for j in [2, 50,100]]
            high_sample_low_k = list(product([7], [3, 5, 15, 20], [15000, 20000,17000]))
            for sample_set in [high_k_dim_comb, high_sample_low_k]: # [low_dim_comb, high_dim_comb,
                for step, k, samples_est in sample_set:
                    if (step>7 and k>120) or (step*k>samples_est) or (step>7 and k==120 and samples_est>5000) or (step==7 and (k>120 and samples_est>=10000)): # Filter these out as computation time takes too long.
                        print('ignore')
                    else:
                        for i in range(3):                
                            # print(''.join(['-' for j in range(10)]), i , ''.join(['-' for j in range(10)]))
                            # list_inf = main(d, n_stopping_rights, True, grid=grid, K_low=100,K_up=20, traj_est=100000, traj_test_ub=10000,traj_est_ub=10000, traj_test_lb=100000, S_0=s0, seed=i+8, mode_desai_BBS_BHS='bbs') 
                            # label_= 'bbs'
                            # inf_cols = [d, s0, n_stopping_rights, '', '', '', '']
                            # inf_list=utils.process_function_output(*list_inf, label_ = label_, grid= grid, info_cols=inf_cols)
                            # information.append(inf_list)
                            for lambda_, p in [(0,50), (0,100), (1,50), (1,100)]:
                                print(''.join(['-' for j in range(10)]), i, step, k, samples_est , ''.join(['-' for j in range(10)]))
                                list_inf = main(d, n_stopping_rights, True, grid=grid, K_low=k, K_noise=0, traj_est=500, traj_test_ub=100,traj_est_ub=samples_est, traj_test_lb=1000, S_0=s0, seed=i+8, steps=step, p=p, lambda_=lambda_) 
                                label_= f'B2013-{lambda_}-{p}'
                                inf_cols = [d, s0, n_stopping_rights, step, k,  i, samples_est]
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