"""
File which executes Glasserman (2004) Upper bound approach to a fractional brownian motion. Name HK comes from Haugh and Kaugen approach, which is a more or less similar idea but applied to supermartingales.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import utils

from tensorflow import keras, constant_initializer, compat, random as random_tf
import numpy as np
from modelRrobust2fBM import model_glasserman_general
compat.v1.logging.set_verbosity(compat.v1.logging.ERROR)
import pickle as pic
from tabulate import tabulate
from datetime import datetime


def main(d=1,print_progress=True, steps= 100, T=1, traj_est=80000, grid=100, step_inner=True, traj_test_lb=150000, traj_test_ub=10000, K_low=200, hurst=0.7, seed=0):
    """
    Main function, which executes algorithm by Glasserman (2004).

    Input:
        d: dimension of the fractional brownian motion. Stopping maximum out of d. Only d=1 is considered in report.
        print_progress: If True: printing results at the end. If False: Only printing times during loops execution algoritm 
        steps: N_T in report, number of possible future stopping dates.
        T: Time of the final possible stopping date.
        traj_est: Trajectories used for estimation of primal LSMC.
        grid: Number of inner simulations.
        step_inner: Determines size of the loop to evaluate basis functions in inner simulations.  If True: set to  2500*100 // traj_test_ub. Otherwise, no lo loops are used.
        traj_test_lb: Number of testing trajectories for primal LSMC.
        traj_test_ub: Number of testring trajectories to evaluate upper bound.
        K_low: Number of nodes,=#basis functions -1, in randomised neural network primal LSMC.
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
    # sim_s = 1*dt
    discount_f= np.exp(-r*dt)
    # generator = FBM(n=steps, hurst=hurst, length=T, method='daviesharte')
    hurst2=2*hurst
    time_s=time_[:,None]
    var_ = (time_s[1:]**hurst2 + time_[1:]**hurst2 - np.abs(time_[1:]-time_s[1:])**hurst2)/2
    B=np.linalg.cholesky(var_)
    Z= train_rng.normal(size=(steps,traj, d))
    # S_0 = np.random.randn(traj)*sim_s
    S = np.swapaxes(np.vstack((np.zeros((1, traj, d)), np.einsum('sj,jkd->skd', B, Z))), 0,1) #+ S_0[:,None, None]
    
    S2 = np.swapaxes(np.vstack((np.zeros((1, traj_test_lb, d)), np.einsum('sj,jkd->skd', B, test_rng.standard_normal((steps,traj_test_lb, d))))), 0,1)
    Z3 = test_rng.normal(size=(steps,traj_test_ub, d))
    S3 = np.swapaxes(np.vstack((np.zeros((1, traj_test_ub, d)), np.einsum('sj,jkd->skd', B, Z3))), 0,1)

    discount_f= np.exp(-r*dt)

    payoff_fracBM = lambda x: np.max(x) if x.ndim==1 else np.max(x[:,0], -1) if x.ndim==2 else  np.max(x[:,0] ,-1)
    payoff_option =payoff_fracBM


    K_lower= K_low


    input_size_lb= (1)*(d)
    input_size_ub=(1)*(d)

  
    model_= model_glasserman_general(model_nn_rng, seed, input_size_lb, K_lower, steps,d, None, input_size_ub, L=1)
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
  
    t0_fineincrement =datetime.now()
    for time in range(steps)[::-1]:    
       ## Training (Upperbound)
        underlying_upperbound = S3[:,time::-1,:]
        underlying_upperbound_next = S3[:,(time+1)::-1,:]
        
        z_inner = np.random.randn(traj_test_ub, d,inner)
        traj_inner = (np.einsum('t, tid->id', B[time, :time],  Z3[:time, :, :])[:,:, None] + B[time,time] * z_inner) # (traj_ub, d, inner) vector with realisations of next period underlying
        
        cur_payoff_inner = np.vstack(([payoff_option((traj_inner[:,:,i])[:,None,:]) for i in range(inner)])).T *discount_f**(time+1) # payoff time t+1 inner simulation
        reg_m_inner= [np.hstack(((traj_inner[:,:,i])[:,None,:], underlying_upperbound)) for i in range(inner)] # stack underlying and previous underlying trajectory -> time t+1 : 0 trajectory used to predict continuation value.
        reg_m_inner=np.vstack(reg_m_inner).reshape(traj_test_ub*inner, (time+2)*d)
        if time==steps-1:
            con_val_i=-np.inf
            actCValue=-np.inf
        else:
            con_val_i = np.hstack([model_.prediction_conval_model2(reg_m_inner[i*traj_test_ub:j*traj_test_ub], time+1, traj_test_ub).T for i,j in inner_])
            reg_m_inner2=underlying_upperbound_next.reshape(traj_test_ub, (time+2)*d)  #np.hstack((S3[:,time+1,:],actPayoff[:,None] ))
            actCValue= model_.prediction_conval_model1(reg_m_inner2, traj_test_ub, time+1)

        stop_val_i = np.mean(np.maximum(con_val_i, cur_payoff_inner), axis=1)
        actPayoff= payoff_option(underlying_upperbound_next)*discount_f**(time+1)
            
        actValue = np.maximum(actCValue, actPayoff)
        M_incr[:,time]= actValue-stop_val_i 

    t1_fineincrement =datetime.now()
    time_ub.append(t1_fineincrement - t0_fineincrement)
      
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
    val_= np.mean(stop_val)
    print(val_)
    std_ = np.std(stop_val)/(traj)**.5
    discount_f_time=np.exp(-r)
    lowerbound= np.mean(stop_val_testing)
    lowerbound_std = np.std(stop_val_testing)/(traj_test_lb)**.5
 

    stopping_process=np.max(S3,-1)*discount_f**steps
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
    return lowerbound, lowerbound_std, upperbound, upperbound_std, np.mean(np.array(time_training)), np.mean(np.array(time_ub)), CV_lowerbound, CV_lowerbound_std, upperbound2, upperbound_std2



information=[]
if __name__=='__main__':
    for d,H in [ (2,0.2)]: #(1,0.3), (1, 0.7)
        for grid in [600]:
            print(''.join(['*' for j in range(10)]), grid ,''.join(['*' for j in range(10)]))
            for i in range(1):                
                print(''.join(['-' for j in range(10)]), i , ''.join(['-' for j in range(10)]))
                list_inf=main(d, True, grid=grid, K_low=300,K_up=None, traj_est=100000, traj_test_ub=10000, traj_test_lb=50000, hurst=H, seed=i+8, steps=9)
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