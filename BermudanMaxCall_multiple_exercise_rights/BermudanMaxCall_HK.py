"""
File which executes  Glasserman (2004) Upper bound approach to pricing a Bermudan Max Call option, with possibly multiple exercise rights. Name HK comes from Haugh and Kaugen approach, which is a more or less similar idea but applied to supermartingales.
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


def main(d=3, L=3, print_progress=True, steps=9,T=3, r=0.05, delta_dividend=0.1, traj_est=80000, grid=100, mode_kaggle=False, traj_test_lb=150000, traj_test_ub=10000, K_low=200, K_noise=None, S_0=110, strike=100, seed=0, payoff_=lambda x, strike: utils.payoff_maxcal(x, strike)):
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
    test_rng =  np.random.default_rng(seed+2000)
    model_nn_rng = np.random.default_rng(seed+4000)
    sim_s = .5*dt
    S_0_train=S_0 *np.exp( train_rng.normal(size=(traj, 1, d))*sigma*(sim_s)**.5 - .5*sigma**2*sim_s)
    discount_f= np.exp(-r*dt)
    dWS= train_rng.normal(size=(traj, steps, d)).astype(np.float32)*((dt)**0.5)
    S = S_0_train*np.exp( np.cumsum(np.hstack((np.zeros((traj,1,d)), dWS*sigma)), axis=1) + np.repeat( (r - delta_dividend - sigma**2/2)*time_, d).reshape(steps+1,d))
    S2 = S_0*np.exp( np.cumsum(np.hstack((np.zeros((traj_test_lb,1,d)), test_rng.normal(size=(traj_test_lb, steps, d)).astype(np.float32)*((dt)**0.5)*sigma)), axis=1) + np.repeat( (r- delta_dividend - sigma**2/2)*time_, d).reshape(steps+1,d))
    S3 = S_0*np.exp( np.cumsum(np.hstack((np.zeros((traj_test_ub,1,d)),  test_rng.normal(size=(traj_test_ub, steps, d)).astype(np.float32)*((dt)**0.5)*sigma)), axis=1) + np.repeat( (r- delta_dividend - sigma**2/2)*time_, d).reshape(steps+1,d))


    discount_f= np.exp(-r*dt)

    payoff_option = lambda x: payoff_(x, strike)

    K_lower= K_low

    input_size_lb= (1)*(d+1)
    input_size_ub=(1)*(d+1)

    model_ = model_glasserman_general(model_nn_rng, seed, input_size_lb, K_lower, steps, d, 0, input_size_ub, L=L, K_noise=K_noise, mode_kaggle=mode_kaggle)


    inner=grid
    
    M_incr=np.zeros((traj_test_ub, L, steps))
    if mode_kaggle:
        step_size=25000000
    else:
        step_size = 300000
    inner_ = np.arange(0, inner*traj_test_ub+step_size, step_size, dtype=int)
    inner_[inner_>=inner*traj_test_ub] = inner*traj_test_ub
    inner_=np.unique(inner_)
    inner_ = [inner_[i: i+2] for i in range(len(inner_)-1)]
    stop_val_archive= np.zeros((traj_est, steps+1,  2))
    #### TRAINING #########
    t0_training=datetime.now()
    for ex_right in range(L):
        stop_val_archive=stop_val_archive[:,:,::-1] # Update stop_val_archive to set previous ex_right level to position stop_val_archive[:,:,0] and write new level at stop_val_archive[:,:,1]
        if print_progress:
            print('right=',ex_right)
        # final_time_exercising= steps-ex_right
        stop_val = payoff_option(S[:,-1,:])*discount_f**steps
        stop_val_archive[:, -1, 1] = np.copy(stop_val)
        for time in range(steps)[::-1]:
            if print_progress:
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


    ## Training (Upperbound)
    #con_val_i_archive= np.zeros((traj_test_ub, inner, steps))
    t0_upperbound = datetime.now()
    for time in range(steps)[::-1]:   
        if print_progress:     
            print('time=',time)
        underlying_upperbound = S3[:,time,:]   
        ## Inner trajectories
        traj_inner = np.exp( sigma*np.random.randn(traj_test_ub, d,inner)*(dt)**.5  + (r-delta_dividend-sigma**2/2)*dt)*underlying_upperbound[:,:,None]
        cur_payoff_inner = np.vstack(([payoff_option(traj_inner[:,:,i]) for i in range(inner)])).T *discount_f**(time+1)
        actPayoff= payoff_option(S3[:,time+1,:])*discount_f**(time+1)
        reg_m_t2 =np.hstack((S3[:,time+1,:], actPayoff[:,None]))
        for ex_right in range(L):
            if time<=steps-1-ex_right: # Only times before or equal to N_T- \ell +1 are relevant, as N_T-\ell +1 is a terminal state. Two rights, 9 periods, must exercise at time 8 if still holds two rights (ex_right=1).
                if time==steps-1:
                    reg_m_inner=None
                    con_val_i=0
                    prev_right_c_t = 0
                else:
                    reg_m_testing=np.dstack((traj_inner.transpose(0,2,1), cur_payoff_inner[:,:,None])).reshape(-1, d+1)
                    con_val_i=[]
                    for slice_start, slice_end in inner_:
                        slice_length = slice_end-slice_start
                        con_val_i.append(model_.prediction_conval_model1(reg_m_testing[slice_start:slice_end], slice_length, time+1,  ex_right))
                    con_val_i=np.hstack(con_val_i)
                    prev_right_c_t= con_val_i_archive_prev if ex_right>0 else 0
                    con_val_i_archive_prev = np.copy(con_val_i) # Update archive to set prev_right_c_t as con_val_i for next exercise right.

                value_ex = cur_payoff_inner.flatten() +prev_right_c_t
                nE_rL =np.mean(np.maximum(value_ex,con_val_i).reshape(traj_test_ub, inner), axis=1) #np.mean(np.maximum(value_ex,con_val_i), axis=1)

                if time<steps-1:
                    prev_right_c_tp1 = model_.prediction_conval_model1(reg_m_t2, traj_test_ub, time+1, ex_right-1) if ex_right>0 else 0         
                    actCValue= model_.prediction_conval_model1(reg_m_t2, traj_test_ub, time+1, ex_right)
                else:
                    actCValue=0
                    prev_right_c_tp1= 0 # np.zeros((traj_test_ub,))
                actValue = np.maximum(actCValue, actPayoff+prev_right_c_tp1)
                M_incr[:, ex_right, time]= actValue-nE_rL
    t1_upperbound = datetime.now()
    time_ub.append(t1_upperbound-t0_upperbound)
    

    #### TESTING - LB ####
    stop_val_testing=0
    prev_stop_time=-1    
    mean_stop_times=np.zeros(L)
    std_stop_times=np.zeros(L)
    for ex_right in range(L):
        if print_progress:
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
    terminal_payoff=payoff_option(S3[:,-1,:])*np.exp(-r*T)
    theta_upperbound= np.zeros((traj_test_ub, L, steps+1))
    theta_upperbound[:,:,-1]=terminal_payoff[:,None]
    for ex_right in range(L):
        if print_progress:
            print('right=',ex_right)
        theta_next_samelevel=terminal_payoff
        for time in range(steps)[::-1]:
            underlying_test = S3[:,time,:]
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
    M=np.dstack((np.zeros(M_incr.shape[:2])[:,:,None], M_incr ))
    M= np.cumsum(M, axis=-1)
    Dual_max_traj=theta_upperbound[:, -1,0]
    upperbound = np.mean(Dual_max_traj)
    upperbound_std= np.std(Dual_max_traj)/(traj_test_ub)**.5

    #### CONTROL VARIATE#####
    stop_val_testingcv=0
    prev_stop_timeCV=np.repeat(-1, traj_test_ub)  
    for ex_right in range(L):
        if print_progress:
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
        # print('up2', upperbound2)
        # print('std2', upperbound_std2)

        print('CV est',CV_lowerbound )
        print('CV std', CV_lowerbound_std)
        # print('time avg testing', np.mean(np.array(time_testing)))
        print('time avg training', np.mean(np.array(time_training)))
        print('time avg ub', np.mean(np.array(time_ub)))
    return lowerbound, lowerbound_std, upperbound, upperbound_std,  np.mean(np.array(time_training)), np.mean(np.array(time_ub)), CV_lowerbound, CV_lowerbound_std


information=[]
# if __name__=='__main__':
    # for d,s0,n_stopping_rights in [ (2,90, 3), (2,90,2), (2, 90, 1)]:
    #     for grid in [120]:
    #         print(''.join(['*' for j in range(10)]), grid ,''.join(['*' for j in range(10)]))
    #         for i in range(1):                
    #             print(''.join(['-' for j in range(10)]), i , ''.join(['-' for j in range(10)]))
    #             list_inf = main(d, n_stopping_rights, True, grid=grid, K_low=400,K_noise=50, traj_est=200000, traj_test_ub=10000, traj_test_lb=500000, S_0=s0, seed=i+8, mode_kaggle=False)
    #             label_= 'HK'
    #             inf_cols = [d, s0, n_stopping_rights, '', '', '', '']
    #             inf_list=utils.process_function_output(*list_inf, label_ = label_, grid= grid, info_cols=inf_cols)
    #             information.append(inf_list)

    # # with open(f'run{datetime.now().strftime("%Y%m%d%H%m%S")}.pic', 'wb') as fh:
    # #     pic.dump(information, fh)
   
    # table_ = tabulate(utils.information_format(information), headers=utils.header_, tablefmt="latex_raw", floatfmt=".4f")


    # print(table_)
    # # folder_txt_log = '/content/drive/MyDrive/'#Tilburg/msc/Thesis/Log'
    # # fh = open(f'logresults.txt', 'a')
    # # fh.write(f'{datetime.now()}\n ')
    # # fh.writelines(table_)
    # # line="".join(np.repeat('*',75))
    # # fh.write(f'\n {line} \n')
    # # fh.close()


traj_test_ub= 10000 
traj_est_primal_dual = 100000
traj_est_MM = 6000
traj_est_MM_belomestny2013=15000
K_lower=400
traj_lb_testing = 200000
information=[]
steps=20



basis_f_K_U_MM=150
basis_f_K_U_MM_belomestny2013=250
K_L_basic= 100
K_U_belomestny2009=200
K_U_SZH2013=150
K_L_AndersenBroadie2004=300
K_L_Glasserman2004=350
basis_f_K_U_MM_BHS=int(basis_f_K_U_MM*1.5)
grid_Glasserman2004=600
inner_sim_SZH=300

if __name__=='__main__':
    for d,s0,n_stopping_rights in [(4,100, 3)]:#[(2,90,9), (2,90, 8), (2,90, 7), (2, 90, 6), (2, 90, 5), (2,90,4), (2,90,3), (2,90,2), (2, 90, 1)]:
        for inner_sim in [500]:
            print(''.join(np.repeat('*', 10)), inner_sim ,''.join(np.repeat('*', 10)))
            for i in range(1):                
                # calibrations_= [[keras.activations.softplus, 30, 'HeNormal', 1]]
                # for activation_f, vs_factor, distribution_vs, w_c in calibrations_:
                inf_cols= [d, s0, n_stopping_rights, '', '', '', '']

                # label_ = f'Belom. et al. (2023)-{steps}'
                # print(''.join(np.repeat('-', 10)),label_, inner_sim, d, s0, n_stopping_rights, ''.join(np.repeat('-', 10)))
                # list_inf=SAA_LP(d, n_stopping_rights, True, grid=inner_sim, K_low=K_L_basic, K_up=basis_f_K_U_MM, traj_est=traj_est_primal_dual, traj_train_ub=traj_est_MM, traj_test_ub=traj_test_ub, traj_test_lb=traj_lb_testing, S_0=s0, seed=i+8, mode_desai_BBS_BHS='bbs')
                # inf_list=utils.process_function_output(*list_inf, label_ = label_, grid= inner_sim, info_cols=inf_cols)
                # information.append(inf_list)  
            
                # label_ = f'Belom. Hilb. Schoenmakers (2019)-{steps}'
                # print(''.join(np.repeat('-', 10)),label_, inner_sim, d, s0, n_stopping_rights, ''.join(np.repeat('-', 10)))
                # list_inf=SAA_LP(d, n_stopping_rights, True, grid=inner_sim, K_low=K_L_basic, K_up=basis_f_K_U_MM_BHS,  traj_est=traj_est_primal_dual, traj_train_ub=traj_est_MM, traj_test_ub=traj_test_ub, traj_test_lb=traj_lb_testing, S_0=s0, seed=i+8,  mode_desai_BBS_BHS='bhs')
                # inf_list=utils.process_function_output(*list_inf, label_ = label_, grid= inner_sim, info_cols=inf_cols)
                # information.append(inf_list)  

                # label_ =f'Desai et al. (2012)-{steps}'
                # print(''.join(np.repeat('-', 10)),label_, inner_sim, d, s0, n_stopping_rights, ''.join(np.repeat('-', 10)))                             
                # list_inf=SAA_LP(d, n_stopping_rights, True, grid=inner_sim, K_low=K_L_basic,K_up=basis_f_K_U_MM, traj_est=traj_est_primal_dual, traj_train_ub=traj_est_MM,traj_test_ub=traj_test_ub, traj_test_lb=traj_lb_testing, S_0=s0, seed=i+8, mode_desai_BBS_BHS='desai')
                # inf_list=utils.process_function_output(*list_inf, label_ = label_, grid= inner_sim, info_cols=inf_cols)
                # information.append(inf_list)  

                # for lambda_ in [1/20, 1/2, 1]:
                #     label_ = f'Belom. Emp. Dual; lambda={lambda_}'
                #     print(''.join(np.repeat('-', 10)),label_, inner_sim, d, s0, n_stopping_rights, ''.join(np.repeat('-', 10)))
                #     list_inf=BelomestnySAA2013(d, n_stopping_rights, True, grid=inner_sim, K_low=K_L_basic, K_up=basis_f_K_U_MM_belomestny2013, traj_est=traj_est_primal_dual, traj_est_ub=traj_est_MM_belomestny2013, traj_test_ub=traj_test_ub, traj_test_lb=traj_lb_testing, S_0=s0,  seed=i+8, lambda_=lambda_, p=100)
                #     inf_list=utils.process_function_output(*list_inf, label_ = label_, grid= inner_sim, info_cols=inf_cols)
                #     information.append(inf_list) 
                    

                # label_=f'SZH (2013) KU100 -{steps}'
                # print(''.join(np.repeat('-', 10)),label_, inner_sim, d, s0, n_stopping_rights, ''.join(np.repeat('-', 10)))
                # list_inf=SZH2013(d, n_stopping_rights, True, grid=inner_sim_SZH, K_low=K_L_basic,K_up=K_U_SZH2013, traj_est=traj_est_primal_dual, traj_test_ub=traj_test_ub, traj_test_lb=traj_lb_testing, S_0=s0,  seed=i+8)
                # inf_list=utils.process_function_output(*list_inf, label_ = label_, grid= inner_sim, info_cols=inf_cols)
                # information.append(inf_list)    

                # label_=f'Belom. et al. (2009) KU100-{steps}'
                # print(''.join(np.repeat('-', 10)),label_, inner_sim, d, s0, n_stopping_rights, ''.join(np.repeat('-', 10)))
                # list_inf=mainBelomestnyetal2009(d, n_stopping_rights, True, grid=inner_sim, K_low=K_L_basic,K_up=K_U_belomestny2009, traj_est=traj_est_primal_dual, traj_test_ub=traj_test_ub, traj_test_lb=traj_lb_testing, S_0=s0,  seed=i+8) 
                # inf_list=utils.process_function_output(*list_inf, label_ = label_, grid= inner_sim, info_cols=inf_cols)
                # information.append(inf_list)   


                # label_ = f'AB (2004)-{steps}'
                # print(''.join(np.repeat('-', 10)), label_, inner_sim, d, s0, n_stopping_rights, ''.join(np.repeat('-', 10)))
                # list_inf=AndersonBroadie2004(d, n_stopping_rights, True, grid=inner_sim, K_low=K_L_AndersenBroadie2004, K_noise=None, traj_est=traj_est_primal_dual, traj_test_ub=traj_test_ub//4, traj_test_lb=traj_lb_testing, S_0=s0,  seed=i+8) 
                # inf_list=utils.process_function_output(*list_inf, label_ = label_, grid= inner_sim, info_cols=inf_cols)
                # information.append(inf_list)

                # label_ = f'AB (2004)-{steps}'
                # print(''.join(np.repeat('-', 10)), label_, inner_sim, d, s0, n_stopping_rights, ''.join(np.repeat('-', 10)))
                # list_inf=AndersonBroadie2004(d, n_stopping_rights, True, grid=inner_sim*3//4, K_low=K_L_AndersenBroadie2004, K_noise=None, traj_est=traj_est_primal_dual, traj_test_ub=traj_test_ub//4, traj_test_lb=traj_lb_testing, S_0=s0,  seed=i+8) 
                # inf_list=utils.process_function_output(*list_inf, label_ = label_, grid= inner_sim, info_cols=inf_cols)
                # information.append(inf_list)

                # label_ = f'GM (2004)-{steps}'
                # print(''.join(np.repeat('-', 10)),label_, inner_sim, d, s0, n_stopping_rights, ''.join(np.repeat('-', 10)))
                # list_inf=Glassermann(d, n_stopping_rights, True, grid=K_L_Glasserman2004, K_low=K_L_Glasserman2004, K_noise=None, traj_est=traj_est_primal_dual, traj_test_ub=traj_test_ub, traj_test_lb=traj_lb_testing, S_0=s0,  seed=i+8) 
                # inf_list=utils.process_function_output(*list_inf, label_ = label_, grid= inner_sim, info_cols=inf_cols)
                # information.append(inf_list)    

                label_ = f'GM (2004)-{steps}'
                print(''.join(np.repeat('-', 10)),label_, inner_sim, d, s0, n_stopping_rights, ''.join(np.repeat('-', 10)))
                list_inf=main(d, n_stopping_rights, True, steps=steps, grid=inner_sim, K_low=K_L_basic,K_noise=None, traj_est=traj_est_primal_dual, traj_test_ub=traj_test_ub, traj_test_lb=traj_lb_testing, S_0=s0,  seed=i+8) 
                inf_list=utils.process_function_output(*list_inf, label_ = label_, grid= inner_sim, info_cols=inf_cols)
                information.append(inf_list)   
                

                # with open(f'run{datetime.now().strftime("%Y%m%d%H%m%S")}.pic', 'wb') as fh:
                #     pic.dump(information, fh)
                # repo.create_file(f'resultfiles/run{datetime.now().strftime("%Y%m%d%H%m%S")}.txt', f'run{datetime.now().strftime("%Y%m%d%H%m%S")}.txt', str(information), branch='main')

    table_ = tabulate(utils.information_format(information), headers=utils.header_, tablefmt="latex_raw", floatfmt=".4f")

    print(table_)


