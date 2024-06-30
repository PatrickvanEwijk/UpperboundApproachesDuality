"""
File which executes Andersen Broadie (2004) Upper bound approach to pricing a Bermudan Max Call option, with possibly multiple exercise rights.
        This variant of algorithm is implemented according to Tsisikilis & Van Roy LSMC.
"""


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import utils

from tensorflow import keras, constant_initializer, compat, random as random_tf
import numpy as np
from  BermudanMaxCall_model import model_glasserman_general
compat.v1.logging.set_verbosity(compat.v1.logging.ERROR)
import pickle as pic
from tabulate import tabulate
from datetime import datetime

def main(d=3, L=3, print_progress=True, steps=9, T=3,r=0.05, delta_dividend=0.1, traj_est=80000, grid=100, mode_kaggle=False, traj_test_lb=150000, traj_test_ub=10000, K_low=200, K_noise=None, S_0=110, strike=100, seed=0, payoff_=lambda x, strike: utils.payoff_maxcal(x, strike), seed_traj_testingALL_add=0):
    """
    Main function, which executes algorithm by Andersen Broadie (2004). 
        Slightly adjusted, as upper biased estimator is estimated directly rather than gap between lower- and upper-biased estimator.

    Input:
        d: dimension of the fractional brownian motion. Stopping maximum out of d. Only d=1 is considered in report.
        L: Number of exercise rights of the Bermudan Max Call option.
        print_progress: If True: printing results at the end. If False: Only printing times during loops execution algoritm 
        steps: N_T in report, number of possible future exercise dates.
        T: Time of the final possible exercise date.
        r: Interest rate.
        delta_dividend: The dividend rate on each stock in the underlying. 
            Interest rate r is fixed to 0.05; volatility is fixed to 0.2.
        mode_kaggle: Boolean. If ran on cloud, code could basically be ran with much more RAM. (Google cloud: machine 64 GB, Kaggle: TPU machine>200 GB).
        traj_est: Trajectories used for estimation of primal LSMC.
        grid: Number of inner simulations.
        traj_test_lb: Number of testing trajectories for primal LSMC.
        traj_test_ub: Number of testring trajectories to evaluate upper bound.
        K_low: Number of nodes,=#basis functions -1, in randomised neural network primal LSMC.
        K_noise: Number of nodes on noise terms in regression. Default (and in whole report) set to None->No noise terms in primal LSMC.
        S_0: Time-0 stock price of all stocks in the underlying.
        strike: Strike price of the Bermudan Max Call option.
        seed: Seed for randomised neural network and simulating trajectories.
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
    test_rng =  np.random.default_rng(seed+1000+seed_traj_testingALL_add)
    model_nn_rng = np.random.default_rng(seed+4000)
    sim_s = .5*dt
    S_0_train=S_0 #*np.exp( train_rng.normal(size=(traj, 1, d))*sigma*(sim_s)**.5 - .5*sigma**2*sim_s)
    discount_f= np.exp(-r*dt)
    dWS= train_rng.normal(size=(traj, steps, d)).astype(np.float32)*((dt)**0.5)
    S = S_0_train*np.exp( np.cumsum(np.hstack((np.zeros((traj,1,d)), dWS*sigma)), axis=1) + np.repeat( (r - delta_dividend - sigma**2/2)*time_, d).reshape(steps+1,d))
    S2 = S_0*np.exp( np.cumsum(np.hstack((np.zeros((traj_test_lb,1,d)), test_rng.normal(size=(traj_test_lb, steps, d)).astype(np.float32)*((dt)**0.5)*sigma)), axis=1) + np.repeat( (r- delta_dividend - sigma**2/2)*time_, d).reshape(steps+1,d))
    S3 = S_0*np.exp( np.cumsum(np.hstack((np.zeros((traj_test_ub,1,d)),  test_rng.normal(size=(traj_test_ub, steps, d)).astype(np.float32)*((dt)**0.5)*sigma)), axis=1) + np.repeat( (r- delta_dividend - sigma**2/2)*time_, d).reshape(steps+1,d))


    payoff_option = lambda x: payoff_(x, strike)

    K_lower= K_low

    input_size_lb= (1)*(d+1)
    input_size_ub=(1)*(d+1)

    model_ = model_glasserman_general(model_nn_rng, seed, input_size_lb, K_lower, steps, d, 0, input_size_ub, L=L, mode_kaggle=mode_kaggle, K_noise=K_noise)

    inner=grid
    
    M_incr=np.zeros((traj_test_ub, L, steps))
    if mode_kaggle:
        step_size=25000000
    else:
        step_size = 500000
    inner_ = np.arange(0, inner*traj_test_ub+step_size, step_size, dtype=int)
    inner_[inner_>=inner*traj_test_ub] = inner*traj_test_ub
    inner_=np.unique(inner_)
    inner_ = [inner_[i: i+2] for i in range(len(inner_)-1)]
    #### TRAINING #########
    t0_training=datetime.now()
    for ex_right in range(L):
        print('right=',ex_right)
        stop_val = payoff_option(S[:,-1,:])*discount_f**steps
        for time in range(steps)[::-1]:
            print('t=',time)
            ### TRAINING
            underlying = S[:,time,:]
            payoff_underlying = payoff_option(underlying)*discount_f**time
            reg_m=np.hstack((underlying, payoff_underlying[:,None]))
            prev_right_c = model_.prediction_conval_model1(reg_m, traj_est, time, ex_right-1) if ex_right>0 else 0

            y = stop_val#payoff_underlying+ prev_right_c#stop_val+ prev_right_c
            con_val = model_.train_finallayer_continuationvalue(reg_m, y, time, traj, dWS[:,time,:], ex_right)
            
            ex_ind = (payoff_underlying+prev_right_c>con_val)|(steps-time<=ex_right) # (steps-time<=ex_right): due to nonnegative payoff
            stop_val = np.where(ex_ind, payoff_underlying+prev_right_c, con_val)#np.where(ex_ind, payoff_underlying, stop_val) 
    t1_training=datetime.now()
    time_training.append(t1_training-t0_training)        
  
    ###### UPPERBOUND ####
    t0_upperbound = datetime.now()
    bool_n_r = np.zeros((traj_test_ub, steps+1, L))
    C_bar=np.zeros((traj_test_ub, steps+1, L))
    # C_bar[:,-1,:]=0 # Unnecesary as initialised as 0.
    Y_comp = np.zeros((traj_test_ub, steps, L))
    np.random.seed(seed+seed_traj_testingALL_add) # inner trajectories
    for time in range(steps-1, -1,-1):
        underlying_upperbound= S3[:,time,:]  
        ## Bool nrl
        cur_payoff_testing_ub_nrl = payoff_option(underlying_upperbound)*discount_f**time
        regm_boolnrl= np.hstack((underlying_upperbound, cur_payoff_testing_ub_nrl[:,None]))
        con_val_boolnrl=[np.zeros(traj_test_ub), *[model_.prediction_conval_model1(regm_boolnrl, traj_test_ub, time, ex_r) for ex_r in range(L)]]
        for l in range(L):
            bool_n_r[:,time,l]=np.where(con_val_boolnrl[l]+cur_payoff_testing_ub_nrl>=con_val_boolnrl[l+1], 1,0)
        for ex_right_left in range(L):
            bool_n_r[:, -(ex_right_left+1):, ex_right_left] =1 # Always exercise in final step
        # inner trajectories y^(l)_n,r
        dW_inner = np.random.randn(traj_test_ub, steps-time, d,inner).astype(np.float32)*(dt**0.5)*sigma
        dW_inner = np.cumsum(np.hstack(( np.zeros((traj_test_ub, 1, d, inner), dtype=np.float32), dW_inner)), axis=1)
        traj_inner = np.exp( dW_inner  + np.tile((r-delta_dividend-sigma**2/2)*np.arange(0*dt, (steps+1-time)*dt, dt)[:,None,None], [1, d, inner]))*np.tile(underlying_upperbound[:,:,None,None], [1, 1, 1+steps-time, inner]).transpose((0, 2, 1, 3)) 

        con_val_ex_general=dict()
        for ex_right_l_ in [right for right in list(range(1, L+1)) if right< steps+1-time]:
            prev_stop_time=np.array([-1])  
            stop_val_testing_tau_r=0
            # stop_val_testing_tau_r_p_1=np.zeros_like(np.where(bool_n_r[:,time,ex_right_l_-1]==1), dtype=np.float32).T
            # mask_bool_nrl= np.where(bool_n_r[:,time,ex_right_l_-1]==1)
            ex_right_l_0_startcount= ex_right_l_-1
            for ex_right_inner in range(ex_right_l_):
                print(f'right= {ex_right_inner}, time= {time}, ex rights all= {ex_right_l_}')
                stop_time=steps-(ex_right_l_0_startcount-ex_right_inner)
          
                if stop_time-time>=0: # Able to exercise all rights still
                    stop_val_testing_round=payoff_option(traj_inner[:,stop_time-time,:].transpose(0, 2, 1))*discount_f**stop_time
                    
                    for time_inner in range(stop_time-1,time,-1): # loop backwards to (time+1) to compute continuation value.
                        underlying_test = traj_inner[:,time_inner-time,:].transpose(0, 2, 1)
                        cur_payoff_testing_ub = payoff_option(underlying_test)*discount_f**time_inner
                        reg_m_testing=np.dstack((underlying_test, cur_payoff_testing_ub[:,:,None])).reshape(-1, d+1)
                        # Computing continuation value current level and previous level. 
                        # Current level
                        if (time_inner, ex_right_l_0_startcount-ex_right_inner) in con_val_ex_general.keys(): #  Continuous level of normal level next iteration (ex_right_inner+=1) := Continuation value previous level current iteration.
                            con_val_no_ex = con_val_ex_general[(time_inner, ex_right_l_0_startcount-ex_right_inner)]
                        else:
                            con_val_no_ex=[]
                            for slice_start, slice_end in inner_:
                                slice_length = slice_end-slice_start
                                con_val_no_ex.append(model_.prediction_conval_model1(reg_m_testing[slice_start:slice_end], slice_length, time_inner,  ex_right_l_0_startcount-ex_right_inner))  # ex_right_l_0_startcount-ex_right_inner: remaining rights-> backward over ex_right_inner 
                            con_val_no_ex=np.hstack(con_val_no_ex)
                            con_val_ex_general[(time_inner, ex_right_l_0_startcount-ex_right_inner)]=con_val_no_ex
                        # Previous level
                        if (time_inner, ex_right_l_0_startcount-1-ex_right_inner) in con_val_ex_general.keys(): #  Continuous level of normal level next iteration (ex_right_inner+=1) := Continuation value previous level current iteration.
                            con_val_ex = con_val_ex_general[(time_inner, ex_right_l_0_startcount-1-ex_right_inner)]
                        elif ex_right_l_0_startcount==ex_right_inner:
                            con_val_ex=np.zeros(traj_test_ub*inner) # Final right(ex_right_inner= ex_right_l_-1): No continuation value once exercised.
                        else:
                            con_val_ex= []
                            for slice_start, slice_end in inner_:
                                slice_length = slice_end-slice_start
                                con_val_ex.append(model_.prediction_conval_model1(reg_m_testing[slice_start:slice_end], slice_length, time_inner, ex_right_l_0_startcount-1-ex_right_inner)) 
                            con_val_ex=np.hstack(con_val_ex)
                            con_val_ex_general[(time_inner, ex_right_l_0_startcount-1-ex_right_inner)]=con_val_ex
        
                        ex_ind = ((cur_payoff_testing_ub.flatten()+con_val_ex>=con_val_no_ex) & (time_inner> prev_stop_time.flatten()))
                        stop_val_testing_round = np.where(ex_ind.reshape(-1, inner), cur_payoff_testing_ub, stop_val_testing_round)
                        stop_time = np.where(ex_ind.reshape(-1, inner), time_inner, stop_time)

                    prev_stop_time = np.copy(stop_time)
                    stop_val_testing_tau_r+=stop_val_testing_round

            C_bar[:, time, ex_right_l_-1]= np.mean(stop_val_testing_tau_r, axis=1) 
            payoff_tp1= payoff_option(S3[:, time+1, :])*discount_f**(time+1)
            c_value_no_ex_p1= C_bar[:, time+1, ex_right_l_-1] 
            c_value_ex_p1 = C_bar[:, time+1, ex_right_l_-2] if ex_right_l_>1 else 0 
            Y_comp[:, time, ex_right_l_-1]=  np.where(bool_n_r[:, time+1, ex_right_l_-1], payoff_tp1 + c_value_ex_p1, c_value_no_ex_p1)
            M_incr[:, ex_right_l_-1, time]= Y_comp[:, time, ex_right_l_-1] - C_bar[:, time, ex_right_l_-1] if time<=steps-ex_right_l_ else -1*np.inf # Only consider the non trivial region: time<steps-#ex rights
    t1_upperbound = datetime.now()
    time_ub.append(t1_upperbound-t0_upperbound)

    print('MEAN Martingale INCREMENTS:\n', np.mean(M_incr, axis=0))
    M=np.dstack((np.zeros(M_incr.shape[:2])[:,:,None], M_incr ))
    M= np.cumsum(M, axis=-1)

    #### TESTING - LB ####
    stop_val_testing=0
    prev_stop_time=-1    
    mean_stop_times=np.zeros(L)
    std_stop_times= np.zeros(L)
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

    #### TESTING - UB ####
    discount_f_time=np.exp(-r)
    terminal_payoff=payoff_option(S3[:,-1,:])*np.exp(-r*T)
    theta_upperbound= np.zeros((traj_test_ub, L, steps+1))
    theta_upperbound[:,:,-1]=terminal_payoff[:,None]
    for ex_right in range(L):
        print('right=',ex_right)
        theta_next_samelevel= payoff_option(S3[:,(-1),:])*discount_f**(steps)
        #payoff_option(S3[:,(-ex_right),:])*discount_f**(steps -ex_right+1) + theta_upperbound[:,ex_right-1, steps -ex_right+1] - M_incr[:,ex_right-1,steps -ex_right] if ex_right>0 else payoff_option(S3[:,(-ex_right),:])*discount_f**(steps -ex_right+1)
        for time in range(steps)[::-1]:
            underlying_test = S3[:,time,:]
            cur_payoff_testing = payoff_option(underlying_test)*discount_f**time
            theta_next_prevlevel= theta_upperbound[:,ex_right-1, time+1] if ex_right>0 else 0
            M_incr_prevlevel= M_incr[:,ex_right-1,time] if ex_right>0 else 0
            M_incr_samelevel= M_incr[:,ex_right,time]
            theta_next_samelevel= np.maximum(cur_payoff_testing - M_incr_prevlevel + theta_next_prevlevel, -M_incr_samelevel +theta_next_samelevel) if steps-ex_right>time else cur_payoff_testing - M_incr_prevlevel + theta_next_prevlevel
            theta_upperbound[:, ex_right, time] = np.copy(theta_next_samelevel)
     
    ### LOWERBOUND
    lowerbound= np.mean(stop_val_testing)
    lowerbound_std = np.std(stop_val_testing)/(traj_test_lb)**.5

    ### UPPERBOUND
    Dual_max_traj=theta_upperbound[:, -1,0]
    upperbound = np.mean(Dual_max_traj)
    upperbound_std= np.std(Dual_max_traj)/(traj_test_ub**0.5)    


    # # Control variate Martingale
    stop_val_testingcv=0
    prev_stop_timeCV=np.repeat(-1, traj_test_ub)  
    for ex_right in range(L):
        print('right=',ex_right)
        stop_timeCV= steps+1-(L-ex_right)
        stop_val_testingCV_round=payoff_option(S3[:,stop_timeCV,:])*discount_f**stop_timeCV 
        for time in range(steps)[-(L-ex_right)::-1]:
            underlying_test = S3[:,time,:]
            cur_payoff_testing = payoff_option(underlying_test)*discount_f**time
            reg_m_testing=np.hstack((underlying_test, cur_payoff_testing[:,None]))
            con_val_no_ex = model_.prediction_conval_model1(reg_m_testing, traj_test_ub, time, L-1-ex_right)      
            con_val_ex= model_.prediction_conval_model1(reg_m_testing, traj_test_ub, time, L-2-ex_right) if L-1-ex_right>0 else 0
            ex_ind = ((cur_payoff_testing+con_val_ex>=con_val_no_ex) & (time> prev_stop_timeCV))
            stop_val_testingCV_round = np.where(ex_ind, cur_payoff_testing, stop_val_testingCV_round)
            stop_timeCV = np.where(ex_ind, time, stop_timeCV)
        samples_array=np.arange(traj_test_ub)
        m_incr = M[samples_array, L-1-ex_right,stop_timeCV] - M[samples_array, L-1-ex_right,prev_stop_timeCV] if ex_right>0 else M[samples_array, L-1-ex_right,stop_timeCV]
        stop_val_testingcv+=stop_val_testingCV_round -m_incr
        prev_stop_timeCV = np.copy(stop_timeCV) # Update previous stopping time


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
        # print('up2', upperbound2)
        # print('std2', upperbound_std2)

        print('CV est',CV_lowerbound )
        print('CV std', CV_lowerbound_std)
        # print('time avg testing', np.mean(np.array(time_testing)))
        print('time avg training', np.mean(np.array(time_training)))
        print('time avg ub', np.mean(np.array(time_ub)))

    return lowerbound, lowerbound_std, upperbound, upperbound_std, np.mean(np.array(time_training)), np.mean(np.array(time_ub)), CV_lowerbound, CV_lowerbound_std


information=[]
if __name__=='__main__':
    np.set_printoptions(edgeitems=20, linewidth=200)
    for d,s0,n_stopping_rights in [(2,90,3), (2,90, 2), (2, 90, 1)]:
        for grid in [500]:
            print(''.join(['*' for j in range(10)]), grid ,''.join(['*' for j in range(10)]))
            for i in range(1):                
                print(''.join(['-' for j in range(10)]), i, ''.join(['-' for j in range(10)]))
                list_inf=main(d, n_stopping_rights, True, grid=grid, K_low=400,K_noise=None, traj_est=450000, traj_test_ub=4000, traj_test_lb=500000, S_0=s0, seed=i+8)
                label_='AB fullyTvR'
                inf_cols = [d, s0, n_stopping_rights, '', '', '', '', '']
                inf_list=utils.process_function_output(*list_inf, label_ = label_, grid= grid, info_cols=inf_cols)
                information.append(inf_list)

    # with open(f'run{datetime.now().strftime("%Y%m%d%H%m%S")}.pic', 'wb') as fh:
    #     pic.dump(information, fh)
   
    table_ = tabulate(utils.information_format(information), headers=utils.header_, tablefmt="latex_raw", floatfmt=".4f")

    print(table_)
    # folder_txt_log = '/content/drive/MyDrive/'#Tilburg/msc/Thesis/Log'
    # fh = open(f'logresults.txt', 'a')
    # fh.write(f'{datetime.now()}\n ')
    # fh.writelines(table_)
    # line="".join(np.repeat('*',75))
    # fh.write(f'\n {line} \n')
    # fh.close()