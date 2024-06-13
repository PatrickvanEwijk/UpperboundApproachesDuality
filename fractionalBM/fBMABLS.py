import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import utils

from tensorflow import keras, constant_initializer, compat, random as random_tf
import numpy as np
from modelRrobust2fBM import model_HaughKaugen
compat.v1.logging.set_verbosity(compat.v1.logging.ERROR)
import pickle as pic
from tabulate import tabulate
from datetime import datetime

def main(d=1,print_progress=True, steps= 100, T=1, traj_est=80000, grid=100, step_inner=True, traj_test_lb=150000, traj_test_ub=10000, K_low=200, K_up=10, hurst=0.7, seed=0, mode_kaggle=False):
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
    # sim_s = 1*dt
    discount_f= np.exp(-r*dt)
    hurst2=2*hurst
    time_s=time_[:,None]
    var_ = (time_s[1:]**hurst2 + time_[1:]**hurst2 - np.abs(time_[1:]-time_s[1:])**hurst2)/2
    B=np.linalg.cholesky(var_)
    Z= train_rng.normal(size=(steps,traj, d))
    S = np.vstack((np.zeros((1, traj, d)), np.tensordot(B, Z, axes=(1,0)))).transpose(1, 0, 2)
    
    S2 = np.vstack((np.zeros((1, traj_test_lb, d)), np.tensordot(B,test_rng.standard_normal((steps,traj_test_lb, d)), axes=(1,0)))).transpose(1, 0, 2) 
    Z3 = test_rng.normal(size=(steps,traj_test_ub, d))
    S3 = np.vstack((np.zeros((1, traj_test_ub, d)), np.tensordot(B, Z3, axes=(1,0)))).transpose(1, 0, 2)

    discount_f= np.exp(-r*dt)

    payoff_fracBM = lambda x: np.max(x) if x.ndim==1 else np.max(x[:,0], -1) if x.ndim==2 else  np.max(x[:,0] ,-1)
    payoff_option =payoff_fracBM


    K_lower= K_low
    K_upper = K_up

    input_size_lb= (1)*(d)
    input_size_ub=(1)*(d)


    model_= model_HaughKaugen(model_nn_rng, seed, input_size_lb, K_lower, steps,d, K_upper, input_size_ub, L=1, mode_kaggle=mode_kaggle)
    inner=grid
    
    M_incr=np.zeros((traj_test_ub, steps))
    if step_inner is True:
        step_inner = 2500*100 // traj_test_ub
    else:
        step_inner=inner

    #### TRAINING #########
    stop_val = np.max(S[:,-1,:], -1)*discount_f**steps
    stop_val_testing= np.max(S2[:,-1,:], -1)*discount_f**steps
    t0_training=datetime.now()
    for time in range(steps)[::-1]:    
        print(time)
        ### TRAINING
        underlying = S[:,time::-1,:]
        reg_m= underlying.reshape(traj_est, (time+1)*d) #np.hstack((underlying, payoff_underlying[:,None]))
        payoff_underlying=payoff_option(underlying)*discount_f**time
        con_val = model_.train_finallayer_continuationvalue(reg_m,stop_val, time, traj, Z[time])        
        stop_val = np.where(payoff_underlying<con_val, stop_val, payoff_underlying)
    t1_training=datetime.now()
    time_training.append(t1_training-t0_training)        

    ###### UPPERBOUND ####
    t0_upperbound = datetime.now()
    bool_n_r = np.zeros((traj_test_ub, steps+1), dtype=bool)
    bool_n_r[:, -1] =True # Always exercise in final step
    C_bar=np.zeros((traj_test_ub, steps+1))
    # C_bar[:,-1,:]=0 # Unnecesary as initialised as 0.
    Y_comp = np.zeros((traj_test_ub, steps))
    for time in range(steps-1, -1,-1):
        print(time)
        underlying_upperbound= S3[:,time::-1,:]  
        ## Bool nrl
        cur_payoff_testing_ub_nrl = payoff_option(underlying_upperbound)*discount_f**time
        regm_boolnrl= underlying_upperbound.reshape(traj_test_ub, (time+1)*d) # np.hstack((underlying_upperbound, cur_payoff_testing_ub_nrl[:,None]))
        con_val_boolnrl=model_.prediction_conval_model1(regm_boolnrl, traj_test_ub, time)
        bool_n_r[:,time]=np.where(0+cur_payoff_testing_ub_nrl.flatten()>=con_val_boolnrl, True, False)
            
        # inner trajectories y^(l)_n,r
        dW_inner = test_rng.standard_normal(size=(traj_test_ub, steps-time, d,inner)).astype(np.float32)
        # np.tensordot(B[time:, :time] , Z3[:time, :, :], axes=(1,0)): explainable part from past. Note B[time:, :time] contains rows (time, ... , T) and columns (0, ... , time-1)
        # np.tensordot(B[time:, time:] , dW_inner, axes=(1,0)) : Unexplainable orthogonal part.
        # past_path = underlying_upperbound , past trajectories. = np.tensordot(B[:time, :time+1] , Z3[:time+1, :, :], axes=(1,0))
        predictable_future_path =  np.tensordot(B[time:, :time] , Z3[:time, :, :], axes=(1,0)).transpose(1, 0, 2) # output shape: (traj_ub, time dimension, d). 
        orthogonal_future_path = np.tensordot(B[time:, time:] , dW_inner, axes=(1,1)).transpose(1, 3, 0, 2) # output shape: (traj_ub, inner, time dimension, d). 
        traj_inner_future =  predictable_future_path[:,None,:,:] + orthogonal_future_path
        traj_inner_future = traj_inner_future.reshape(traj_test_ub*inner, steps-time, d)
        mask_not_exercised = np.repeat(True, traj_test_ub*inner)
        stop_val_testing_round= np.zeros((traj_test_ub*inner))
        for time_inner in range(time+1,steps): # loop forward to (T) computing continuation value.
            underlying_test_future = traj_inner_future[mask_not_exercised,time_inner-time-1::-1, :]
            cur_payoff_testing_ub = (payoff_option(underlying_test_future)*discount_f**time_inner).flatten()
            reg_m_testing_future=underlying_test_future.reshape(-1, d*(time_inner-time)) # reshape to d*(time_inner+1-time) columns. First d columns for final considered time.
            reg_m_testing_past = underlying_upperbound.reshape(traj_test_ub, (time+1)*d) # past trajectories to be used.
            # Computing continuation value current level 
            con_val_no_ex=[]
            ## inner loop grid for calculations
            num_instances_calc= np.sum(mask_not_exercised)
            mask_not_exercised_loc = np.where(mask_not_exercised)[0]
            size_loop= int(traj_test_ub*step_inner)
            inner_ = np.arange(0, num_instances_calc+size_loop, size_loop, dtype=int)
            inner_[inner_>=num_instances_calc] = num_instances_calc
            inner_=list(np.unique(inner_))
            inner_ = [inner_[i: i+2] for i in range(len(inner_)-1)]
            past_paths_reg_m=np.repeat(reg_m_testing_past, inner, axis=0).reshape(-1, d*(1+time))[mask_not_exercised_loc] # Stack previous paths to inner trajectories to construct regression matrix
            reg_m  = np.hstack((reg_m_testing_future, past_paths_reg_m))
            # Clear RAM
            del past_paths_reg_m
            del reg_m_testing_future
            past_paths_reg_m=None
            reg_m_testing_future=None
            #
            for slice_start, slice_end in inner_:
                slice_length=slice_end-slice_start
                con_val_no_ex.append(model_.prediction_conval_model1(reg_m[slice_start:slice_end], slice_length, time_inner))
            con_val_no_ex=np.hstack(con_val_no_ex)
            ex_ind = (cur_payoff_testing_ub>=con_val_no_ex)
            stop_val_testing_round[mask_not_exercised] = np.where(ex_ind, cur_payoff_testing_ub, 0) # stop_val_testing_round[mask_not_exercised])
            mask_not_exercised[mask_not_exercised]=np.where(ex_ind==True, False, True) # Update mask_not_exercised 'archive' by setting positions to true which are exercised.
        stop_val_testing_round[mask_not_exercised] = payoff_option(traj_inner_future[mask_not_exercised,-1,:][:,None,:])*discount_f**steps # Never exercised: get terminal payoff.     
        stop_val_testing_round= stop_val_testing_round.reshape(-1, inner)   # Reshape to correct shape from vector form.

        C_bar[:, time]= np.mean(stop_val_testing_round, axis=1) 
        payoff_tp1= (payoff_option(S3[:, time+1::-1, :])*discount_f**(time+1)).flatten()
        c_value_no_ex_p1= C_bar[:, time+1] 
        Y_comp[:, time]=  np.where(bool_n_r[:, time+1], payoff_tp1, c_value_no_ex_p1)
        M_incr[:, time]= Y_comp[:, time] - C_bar[:, time] 
    t1_upperbound = datetime.now()
    time_ub.append(t1_upperbound-t0_upperbound)
    
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
        
    #CONSTRUCT MARTINGALE
    lowerbound= np.mean(stop_val_testing)
    lowerbound_std = np.std(stop_val_testing)/(traj_test_lb)**.5
 

    stopping_process= np.max(S3, -1)*discount_f**np.arange(steps+1) #np.vstack([payoff_option(S3[:,t::-1,0])*discount_f**t for t in range(steps+1)]).T #= S3[:,:,0]
   # M_incr*=discount_f_time**time_[1:]
    M=np.hstack((np.zeros((traj_test_ub,1)), np.cumsum(M_incr, axis=1)))
    print(np.mean(M[:,-3:], axis=0))
    max_traj=np.max((stopping_process -M), axis=1)
    upperbound = np.mean(max_traj)
    upperbound_std= np.std(max_traj)/(traj_test_ub)**.5

    exp_max = np.maximum.accumulate((stopping_process-M)[:, ::-1], axis=1)[:, ::-1]
    U = exp_max +M

    U_w1 = np.maximum(np.max(S3[:,-1,:],-1)*discount_f**(steps), U[:,-1] )-M[:,-1]
    for t_rough in range(steps)[::-1]:
        underlying_upperbound_test = S3[:,t_rough::-1,:]
        cur_payoff_ub_test = payoff_option(underlying_upperbound_test)*discount_f**(t_rough)
        reg_m_=underlying_upperbound_test.reshape(traj_test_ub, (t_rough+1)*d)
        con_val = model_.prediction_conval_model1(reg_m_, traj_test_ub, t_rough)
        ind_payoffnow=(con_val<=cur_payoff_ub_test)
        U_w1= np.where(ind_payoffnow, np.maximum(cur_payoff_ub_test, U[:, t_rough])-M[:,t_rough], U_w1)

    upperbound2 = np.mean(U_w1)
    upperbound_std2= np.std(U_w1)/(traj_test_ub)**.5

    # Control variate Martingale
    stop_val_testingcv = np.max(S3[:,-1,:], -1)*discount_f**steps-M[:,-1] #payoff_option()*discount_f**steps-M[:,-1]
    for time in range(steps)[::-1]:
        ## TESTING (Lowerbound CV)
        underlying_testCV = S3[:,time::-1,:]
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
    for d,H in [ (2,0.2), (1,0.3), (1, 0.7)]:
        for grid in [1300]:
            print(''.join(['*' for j in range(10)]), grid ,''.join(['*' for j in range(10)]))
            for i in range(1):                
                print(''.join(['-' for j in range(10)]), i, ''.join(['-' for j in range(10)]))
                list_inf=main(d, True, grid=grid, K_low=300,K_up=50, traj_est=10000, traj_test_ub=1000, traj_test_lb=50000, hurst=H, seed=i+8, steps=9)
                label_='AB LS'
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