import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import utils

from tensorflow import keras, constant_initializer, compat, random as random_tf
import numpy as np
from modelRrobust2MS import model_HaughKaugen
compat.v1.logging.set_verbosity(compat.v1.logging.ERROR)
import pickle as pic
from tabulate import tabulate
from datetime import datetime

def main(d=3, L=3, print_progress=True, traj_est=80000, grid=100, mode_kaggle=False, traj_test_lb=150000, traj_test_ub=10000, K_low=200, K_up=10, S_0=110, strike=100, seed=0):
    time_training=[]
    time_testing=[]
    time_ub=[]
    utils.set_seeds(seed)
    steps= 9
    T=3
    dt = T/steps
    traj=traj_est
    sigma = 0.2
    time_ = np.arange(0,T+0.5*dt, dt)
    r=0.05
    delta_dividend= 0.1

    train_rng= np.random.default_rng(seed)
    test_rng =  np.random.default_rng(seed+1000)
    model_nn_rng = np.random.default_rng(seed+4000)
    sim_s = 3.5*dt
    S_0_train=S_0 *np.exp( train_rng.normal(size=(traj, 1, d))*sigma*(sim_s)**.5 - .5*sigma**2*sim_s)
    discount_f= np.exp(-r*dt)
    dWS= train_rng.normal(size=(traj, steps, d)).astype(np.float32)*((dt)**0.5)
    S = S_0_train*np.exp( np.cumsum(np.hstack((np.zeros((traj,1,d)), dWS*sigma)), axis=1) + np.repeat( (r - delta_dividend - sigma**2/2)*time_, d).reshape(steps+1,d))
    S2 = S_0*np.exp( np.cumsum(np.hstack((np.zeros((traj_test_lb,1,d)), test_rng.normal(size=(traj_test_lb, steps, d)).astype(np.float32)*((dt)**0.5)*sigma)), axis=1) + np.repeat( (r- delta_dividend - sigma**2/2)*time_, d).reshape(steps+1,d))
    S3 = S_0*np.exp( np.cumsum(np.hstack((np.zeros((traj_test_ub,1,d)),  test_rng.normal(size=(traj_test_ub, steps, d)).astype(np.float32)*((dt)**0.5)*sigma)), axis=1) + np.repeat( (r- delta_dividend - sigma**2/2)*time_, d).reshape(steps+1,d))

    payoff_maxcal=  lambda x: np.maximum(np.max(x, axis=-1) - strike,0)
    payoff_basketcall = lambda x: np.maximum(np.mean(x, axis=-1) - strike,0)
    payoff_option =payoff_maxcal

    K_lower= K_low
    K_upper = K_up

    input_size_lb= (1)*(d+1)
    input_size_ub=(1)*(d+1)

    model_ = model_HaughKaugen(model_nn_rng, seed, input_size_lb, K_lower, steps, d, K_upper, input_size_ub, L=L, mode_kaggle=mode_kaggle, K_noise=None)

    inner=grid
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
  
    t0_upperbound=datetime.now()
    bool_n_r = np.zeros((traj_test_ub, steps, L))
    Y_bar=np.zeros((traj_test_ub, steps, L))
    var_theta_n_r= np.zeros((traj_test_ub, steps, L), dtype=int)
    for time in range(steps):
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
        dW_inner = np.random.randn(traj_test_ub, steps-time, d,inner).astype(np.float32)*(dt)**.5
        dW_inner = np.cumsum(np.hstack(( np.zeros((traj_test_ub, 1, d, inner), dtype=np.float32), dW_inner)), axis=1)
        traj_inner = np.exp( sigma*dW_inner  + np.tile((r-delta_dividend-sigma**2/2)*np.arange(0*dt, (steps+1-time)*dt, dt)[:,None,None], [1, d, inner]))*np.tile(underlying_upperbound[:,:,None,None], [1, 1, 1+steps-time, inner]).transpose((0, 2, 1, 3)) 

        con_val_ex_general=dict()
        for ex_right_l_ in range(1, L+1):
            prev_stop_time=np.array([-1])  
            stop_val_testing_tau_r=0
            stop_val_testing_tau_r_p_1=np.zeros_like(np.where(bool_n_r[:,time,ex_right_l_-1]==1), dtype=np.float32).T
            mask_bool_nrl= np.where(bool_n_r[:,time,ex_right_l_-1]==1)
            ex_right_l_0_startcount= ex_right_l_-1

            for ex_right_inner in range(ex_right_l_):
                print(f'right= {ex_right_inner}, time= {time}, ex rights all= {ex_right_l_}')
                stop_time=steps-(ex_right_l_0_startcount-ex_right_inner)
                if stop_time-time<0:
                    print('less rights than dates left')    
                if stop_time-time>=0: # Able to exercise all rights still
                    stop_val_testing_round=payoff_option(traj_inner[:,stop_time-time,:].transpose(0, 2, 1))*discount_f**stop_time
                    
                    if steps-(ex_right_l_-ex_right_inner)==time: # Loop activated, but \tau_r+1 not captured in loop as computed outside loop for final step.
                        if len(stop_val_testing_tau_r_p_1)>0:
                                stop_val_testing_tau_r_p_1=stop_val_testing_tau_r_p_1+np.copy(stop_val_testing_round[mask_bool_nrl])
                    
                    for time_inner in range(steps-(ex_right_l_-ex_right_inner),time-1,-1):
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
                                con_val_ex.append(model_.prediction_conval_model1(reg_m_testing[slice_start:slice_end],  slice_length, time_inner, ex_right_l_0_startcount-1-ex_right_inner)) 
                            con_val_ex=np.hstack(con_val_ex)
                            con_val_ex_general[(time_inner, ex_right_l_0_startcount-1-ex_right_inner)]=con_val_ex
        
                        ex_ind = ((cur_payoff_testing_ub.flatten()+con_val_ex>=con_val_no_ex) & (time_inner> prev_stop_time.flatten()))
                        stop_val_testing_round = np.where(ex_ind.reshape(-1, inner), cur_payoff_testing_ub, stop_val_testing_round)
                        stop_time = np.where(ex_ind.reshape(-1, inner), time_inner, stop_time)
                        if time_inner==time+1:
                            if len(stop_val_testing_tau_r_p_1)>0:
                                stop_val_testing_tau_r_p_1=stop_val_testing_tau_r_p_1+np.copy(stop_val_testing_round[mask_bool_nrl])
                    prev_stop_time = np.copy(stop_time)
                    stop_val_testing_tau_r+=stop_val_testing_round
            Y_bar[:, time, ex_right_l_-1]= np.mean(stop_val_testing_tau_r, axis=1)    
            if len(stop_val_testing_tau_r_p_1)>0:
                var_theta_n_r[mask_bool_nrl, time, ex_right_l_-1]=np.mean(stop_val_testing_tau_r_p_1, axis=1)-Y_bar[mask_bool_nrl, time, ex_right_l_-1]
    t1_upperbound = datetime.now()
    time_ub.append(t1_upperbound-t0_upperbound)
    
    #### TESTING - UB  Brute Force####
    discount_f_time=np.exp(-r)
    terminal_payoff=payoff_option(S3[:,-1,:])*np.exp(-r*T)
    theta_upperbound= np.zeros((traj_test_ub, L, steps+1))
    theta_upperbound[:,:,-1]=terminal_payoff[:,None]
    from itertools import product
    comb_=[]
    Y_bar_extended= np.hstack((Y_bar, np.tile( (payoff_option(S3[:,-1,:])*np.exp(-r*T))[:,None,None], [1, 1, L])))
    for ex_rights in [ex_rights  for ex_rights in product(*[range(steps+1) for i in range(L)]) if np.all(np.diff(ex_rights)>0)]:
            Z_part= np.hstack([(payoff_option(S3[:,time_ex_right,:])*discount_f**time_ex_right)[:,None] for time_ex_right in ex_rights])
            Z_part=np.sum(Z_part, axis=1)
            diff_y = np.hstack(([ (Y_bar_extended[:, time_ex_right, L-num_-2]- Y_bar_extended[:, time_ex_right, L-num_-1])[:,None] if num_<L-1 else (-Y_bar_extended[:,time_ex_right, L-num_-1])[:,None] for num_, time_ex_right in enumerate(ex_rights)]))
            diff_y = np.sum(diff_y, axis=1)
            ex_rights_extended = [0, *ex_rights]
            ex_rights_extended= [(ex_rights_extended[i:i+2]) for i in range(len(ex_rights))]
            var_theta_sum = np.hstack([np.sum(var_theta_n_r[:,time_right_prev: time_right_next, L-ex_right_num-1], axis=1)[:,None] for ex_right_num, [time_right_prev,time_right_next]  in enumerate(ex_rights_extended)])
            var_theta_sum = np.sum(var_theta_sum, axis=1)
            comb_.append((Z_part+diff_y+var_theta_sum)[:,None])
    comb_= np.hstack((comb_))
    max_ = np.max(comb_, axis=1)
    gap = np.mean(max_)
    gap_std= np.std(max_)/(len(max_)**.5)
    print('Delta GAP', gap)
    print('Delta GAP std',gap_std)

    #### TESTING - UB2  Brute Force####
    terminal_payoff=payoff_option(S3[:,-1,:])*np.exp(-r*T)
    theta_upperbound= np.zeros((traj_test_ub, L, steps+1))
    theta_upperbound[:,:,-1]=terminal_payoff[:,None]

    comb_2=[]
    Y_bar_extended= np.hstack((Y_bar, np.tile( (payoff_option(S3[:,-1,:])*np.exp(-r*T))[:,None,None], [1, 1, L])))
    for ex_rights in [ex_rights  for ex_rights in product(*[range(steps+1) for i in range(L)]) if np.all(np.diff(ex_rights)>0)]:
            Z_part= np.hstack([(payoff_option(S3[:,time_ex_right,:])*discount_f**time_ex_right)[:,None] for time_ex_right in ex_rights])
            Z_part=np.sum(Z_part, axis=1)
            ex_rights_extended = [0, *ex_rights]
            ex_rights_extended= [(ex_rights_extended[i:i+2]) for i in range(len(ex_rights))]
            diff_y = np.hstack(([ (Y_bar_extended[:, time_right_prev, L-ex_right_num-1]- Y_bar_extended[:, time_right_next, L-ex_right_num-1])[:,None] for ex_right_num, [time_right_prev,time_right_next] in enumerate(ex_rights_extended)]))
            diff_y = np.sum(diff_y, axis=1)

            var_theta_sum = np.hstack([np.sum(var_theta_n_r[:,time_right_prev: time_right_next, L-ex_right_num-1], axis=1)[:,None] for ex_right_num, [time_right_prev,time_right_next]  in enumerate(ex_rights_extended)])
            var_theta_sum = np.sum(var_theta_sum, axis=1)
            comb_2.append((Z_part+diff_y+var_theta_sum)[:,None])
    comb_2= np.hstack((comb_2))
    max_2 = np.max(comb_2, axis=1)
    ub2 = np.mean(max_2)
    ub2_std= np.std(max_2)/(traj_test_ub**.5)
    print('UB type2', ub2)
    print('UB type2 std',ub2_std)


    #### TESTING - LB ####
    stop_val_testing=0
    prev_stop_time=-1    
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
        prev_stop_time = np.copy(stop_time)
        stop_val_testing+=stop_val_testing_round
    
     
    ### LOWERBOUND
    lowerbound= np.mean(stop_val_testing)
    lowerbound_std = np.std(stop_val_testing)/(traj_test_lb)**.5

    ### UPPERBOUND
    upperbound = lowerbound+ gap
    upperbound_std= (np.var(max_)/traj_test_ub + np.var(stop_val_testing)/traj_test_lb)**.5  


    if print_progress:
        print('Lowerbound')
        print('Value', lowerbound)
        print('Std',lowerbound_std)

        print('Upperbound')
        print('up', upperbound)
        print('std',upperbound_std)
        # print('up2', upperbound2)
        # print('std2', upperbound_std2)

        print('CV est',np.nan )
        print('CV std', np.nan)
        # print('time avg testing', np.mean(np.array(time_testing)))
        print('time avg training', np.mean(np.array(time_training)))
        print('time avg ub', np.mean(np.array(time_ub)))
    return lowerbound, lowerbound_std, upperbound, upperbound_std,  np.mean(np.array(time_training)), np.mean(np.array(time_ub)), 0, 0


information=[]
if __name__=='__main__':
    for d,s0,n_stopping_rights in [(2,90,3), (2,90, 2), (2, 90, 1)]:
        for grid in [120]:
            print(''.join(['*' for j in range(10)]), grid ,''.join(['*' for j in range(10)]))
            for i in range(1):                
                print(''.join(['-' for j in range(10)]), i ,  ''.join(['-' for j in range(10)]))
                list_inf=main(d, n_stopping_rights, True, grid=grid, K_low=400,K_up=50, traj_est=250000, traj_test_ub=10000, traj_test_lb=500000, S_0=s0, seed=i+8)
                label_='AB GAP'
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