import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras, constant_initializer, compat, random as random_tf
import numpy as np
from modelRrobust2MS import model_Belomestny_etal
compat.v1.logging.set_verbosity(compat.v1.logging.ERROR)
import pickle as pic
from tabulate import tabulate
from datetime import datetime





def main(d=3, L=3, print_progress=True, steps=9,T=3,delta_dividend=0.1, traj_est=80000, grid=100, mode_kaggle=False, traj_test_lb=150000, traj_test_ub=10000, K_low=200, K_up=100, K_noise=None, S_0=110, strike=100, seed=0, step_size=None, payoff_=lambda x, strike: utils.payoff_maxcal(x, strike)):
    time_training=[]
    time_testing=[]
    utils.set_seeds(seed)
    time_ub=[]
   
    dt = T/steps
    traj=traj_est
    sigma = 0.2
    time_ = np.arange(0,T+0.5*dt, dt)
    r=0.05

    train_rng= np.random.default_rng(seed)
    test_rng =  np.random.default_rng(seed+2000)
    model_nn_rng = np.random.default_rng(seed+4000)
   
    
    sim_s = .5*dt
    S_0_train=S_0 *np.exp(train_rng.normal(loc=0.0, scale=1.0, size=(traj, 1, d))*sigma*(sim_s)**.5 - .5*sigma**2*sim_s)
    dWs= train_rng.normal(loc=0.0, scale=1.0, size=(traj, steps, d)).astype(np.float32)*((dt)**0.5)
    S = S_0_train*np.exp( np.cumsum(np.hstack((np.zeros((traj,1,d)), dWs*sigma)), axis=1) + np.repeat( (r - delta_dividend - sigma**2/2)*time_, d).reshape(steps+1,d))
    S2 = S_0*np.exp( np.cumsum(np.hstack((np.zeros((traj_test_lb,1,d)), test_rng.normal(size=(traj_test_lb, steps, d))*((dt)**0.5)*sigma)), axis=1) + np.repeat( (r- delta_dividend - sigma**2/2)*time_, d).reshape(steps+1,d))
    dW_ub= train_rng.normal(size=(traj_est, steps, d)).astype(np.float32)*((dt)**0.5)
    S3 = S_0_train*np.exp( np.cumsum(np.hstack((np.zeros((traj_est,1,d)),dW_ub*sigma)), axis=1) + np.repeat( (r- delta_dividend - sigma**2/2)*time_, d).reshape(steps+1,d))

    discount_f= np.exp(-r*dt)
    payoff_option = lambda x: payoff_(x, strike)

    K_lower= K_low
    K_upper = K_up

    input_size_lb= d+1
    input_size_ub=d+1


    model_ = model_Belomestny_etal(model_nn_rng, seed, input_size_lb, K_lower, steps, d, K_upper, input_size_ub, L=L, K_noise=K_noise, mode_kaggle=mode_kaggle)

    
    stop_val_archive= np.zeros((traj_est, steps+1, 2))
    stop_val_archive_ub= np.zeros((traj_est, steps+1, 2))
    #### TRAINING ########
    t0_training=datetime.now()
    for ex_right in range(L):
        stop_val_archive=stop_val_archive[:,:,::-1] # Update stop_val_archive to set previous ex_right level to position stop_val_archive[:,:,0] and write new level at stop_val_archive[:,:,1]
        stop_val_archive_ub=stop_val_archive_ub[:,:,::-1] # Update stop_val_archive to set previous ex_right level to position stop_val_archive[:,:,0] and write new level at stop_val_archive[:,:,1]
        print('right=',ex_right)
        stop_val = payoff_option(S[:,-1,:])*discount_f**steps
        stop_val_ub= payoff_option(S[:,-1,:])*discount_f**steps
        stop_val_archive[:, -1, 1] = np.copy(stop_val)
        stop_val_archive_ub[:,-1, 1] = np.copy(stop_val_ub)
        for time in range(steps)[::-1]:
            print('t=',time)
            ### TRAINING
            ### LB
            underlying = S[:,time,:]
            payoff_underlying = payoff_option(underlying)*discount_f**time
            reg_m=np.hstack((underlying, payoff_underlying[:,None]))
            prev_right_c= stop_val_archive[:,time+1, 0]
            con_val_ex = model_.prediction_conval_model1(reg_m, traj_est, time, ex_right-1) if ex_right>0 else 0

            y = stop_val#payoff_underlying+ prev_right_c#stop_val+ prev_right_c
            con_val_no_ex = model_.train_finallayer_continuationvalue(reg_m, y, time, traj, dWs[:,time,:], ex_right) #  reg_m,y, time, traj, jump, ex_right=0
            ex_ind = (payoff_underlying+con_val_ex>con_val_no_ex)|(steps-time<=ex_right) # (steps-time<=ex_right): due to nonnegative payoff
            stop_val = np.where(ex_ind, payoff_underlying+prev_right_c, stop_val)#np.where(ex_ind, payoff_underlying, stop_val) 
            stop_val_archive[:,time, 1]=np.copy(stop_val)
            ### UB
            underlying_upperbound = S3[:,time,:]
            cur_payoff_ub = payoff_option(underlying_upperbound)*discount_f**(time)
            reg_m_ub=np.hstack((underlying_upperbound, cur_payoff_ub[:,None]))
            prev_right_c_ub = stop_val_archive_ub[:, time+1, 0]
            con_val_no_ex_ub =  model_.prediction_conval_model1(reg_m_ub, traj_est, time, ex_right)
            con_val_ex_ub= model_.prediction_conval_model1(reg_m_ub, traj_est, time, ex_right-1) if ex_right>0 else 0
            h_minus_c= (np.copy(stop_val_ub) - con_val_no_ex_ub)
            jumps = (dW_ub[:, time, :])/(dt)
            y_combined = jumps*h_minus_c[:,None]
            _ = model_.train_finallayer_continuationvalue_upper(reg_m_ub,y_combined, time, traj_est, ex_right, jumps)

            ex_ind_ub = (cur_payoff_ub+con_val_ex_ub>con_val_no_ex_ub)|(steps-time<=ex_right) # (steps-time<=ex_right): due to nonnegative payoff
            stop_val_ub =  np.where(ex_ind_ub, cur_payoff_ub+prev_right_c_ub, stop_val_ub) 
            stop_val_archive_ub[:,time, 1]=np.copy(stop_val_ub)

    t1_training=datetime.now()
    time_training.append(t1_training-t0_training)

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

    #CONSTRUCT MARTINGALE ON FINER GRID
    steps_fine = steps*grid
    dt_fine = T/(steps_fine)
    time_fine = np.arange(0,T+0.5*dt_fine, dt_fine)
    dW_testingub= test_rng.normal(size=(traj_test_ub, steps_fine, d)).astype(np.float32)*((dt_fine)**0.5)
    S4 = S_0*np.exp( np.cumsum(np.hstack((np.zeros((traj_test_ub,1,d)),dW_testingub*sigma)), axis=1) + np.repeat( (r- delta_dividend - sigma**2/2)*time_fine, d).reshape(steps_fine+1,d))
    discount_f_fine = np.exp(-r*dt_fine)

    t0_fineincrement =datetime.now()
    M_incr=np.zeros((traj_test_ub, L, steps_fine))
    ## Loop to construct martingale increments. Loop takes place over trajectories of UB if mode_kaggle=True
    if mode_kaggle:
        if step_size is None:
            step_size= int(np.floor(250*300*100000*8/K_up/grid/traj_test_ub))
        steps_ub = np.arange(0,traj_test_ub+step_size, step_size)
        steps_ub[steps_ub>=traj_test_ub]=traj_test_ub
        steps_ub=list(np.unique(steps_ub))
        steps_ub = [(steps_ub[i],steps_ub[i+1]) for i in range(len(steps_ub)-1)]

        for ex_right in range(L):
            M_incr_round= []    
            for step_i, step_i_p in steps_ub:
                underlying_upperbound_test = S4[step_i:step_i_p,:-1,:]
                # print(underlying_upperbound_test.shape)
                cur_payoff_ub_test = payoff_option(underlying_upperbound_test)*discount_f_fine**np.arange(steps_fine)
                reg_m_=np.dstack((underlying_upperbound_test, cur_payoff_ub_test[:,:,None]))
                M_incr_round.append(model_.prediction_Z_model_upper2(reg_m_, traj_test_ub, grid, dW_testingub[step_i:step_i_p], ex_right))
            M_incr_round=np.vstack(M_incr_round)
            M_incr[:,ex_right,:]=np.copy(M_incr_round)
    else:
        for ex_right in range(L):
            for t_fine in range(steps_fine):
                underlying_upperbound_test = S4[:,t_fine,:] #if (t_fine+1)//grid<steps_fine else S4[:,t_fine,:]
                cur_payoff_ub_test = payoff_option(underlying_upperbound_test)*discount_f_fine**t_fine #if t_fine//grid>-1 else payoff_option(underlying_upperbound_test)
                reg_m_=np.hstack((underlying_upperbound_test, cur_payoff_ub_test[:,None]))
                M_incr[:,ex_right, t_fine]= model_.prediction_Z_model_upper(reg_m_, traj_test_ub, t_fine//grid, dW_testingub[:,t_fine,:], ex_right)
    t1_fineincrement =datetime.now()
    time_ub.append(t1_fineincrement - t0_fineincrement)


    #### Reshape S4 and M at exercise points only (intermediate points deleted).
    grid_consideration = np.arange(0, steps_fine+.5*grid, grid, dtype=int)
    M=np.dstack((np.zeros((traj_test_ub,L,1)), np.cumsum(M_incr, axis=-1)))[:,:, grid_consideration]
    M_incr= np.diff(M, axis=-1)
    S4=S4[:,grid_consideration, :]

    #### TESTING - UB ####
    discount_f_time=np.exp(-r)
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
            theta_next_samelevel= np.maximum(cur_payoff_testing - M_incr_prevlevel + theta_next_prevlevel, -M_incr_samelevel +theta_next_samelevel)
            theta_upperbound[:, ex_right, time] = np.copy(theta_next_samelevel)
     
    ##### Lowerbound
    lowerbound= np.mean(stop_val_testing)
    lowerbound_std = np.std(stop_val_testing)/(traj_test_lb)**.5

    ##### UPPERBOUND
    Dual_max_traj=theta_upperbound[:, -1,0]
    upperbound = np.mean(Dual_max_traj)
    upperbound_std= np.std(Dual_max_traj)/(traj_test_ub)**.5

    # # Control variate Martingale
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
        m_incr = M[samples_array, L-1-ex_right,stop_timeCV] - M[samples_array, L-1-ex_right,prev_stop_timeCV] if ex_right>0 else M[samples_array, L-1-ex_right,stop_timeCV]
        stop_val_testingcv+=stop_val_testingCV_round -m_incr
        prev_stop_timeCV = np.copy(stop_timeCV) # Update previous stopping time

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

        print('CV est',CV_lowerbound)
        print('CV std',CV_lowerbound_std)
        print('time avg training', np.mean(np.array(time_training)))
        print('time avg ub', np.mean(np.array(time_ub)))
    return lowerbound, lowerbound_std, upperbound, upperbound_std, np.mean(np.array(time_training)), np.mean(np.array(time_ub)), CV_lowerbound, CV_lowerbound_std

# information=[]
# if __name__=='__main__':
#     for d,s0,n_stopping_rights in [ (2,90, 3), (2, 90, 2), (2,90,1)]:
#         for grid in [100]:
#             print(''.join(['*' for j in range(10)]), grid ,''.join(['*' for j in range(10)]))
#             for i in range(1):                
#                 print(''.join(['-' for j in range(10)]), i ,  ''.join(['-' for j in range(10)]))
#                 list_inf=main(d, n_stopping_rights, True, grid=grid, K_low=100,K_up=120, K_noise=None, traj_est=500000, traj_test_ub=1000, traj_test_lb=50000, S_0=s0, seed=i+8)
#                 label_='Belom. et al. LS'
#                 inf_cols = [d, s0, n_stopping_rights, '', '', '', '']
#                 inf_list=utils.process_function_output(*list_inf, label_ = label_, grid= grid, info_cols=inf_cols)
#                 information.append(inf_list)

#     # with open(f'run{datetime.now().strftime("%Y%m%d%H%m%S")}.pic', 'wb') as fh:
#     #     pic.dump(information, fh)
   
#     table_ = tabulate(utils.information_format(information), headers=utils.header_, tablefmt="latex_raw", floatfmt=".4f")

#     print(table_)
#     # folder_txt_log = '/content/drive/MyDrive/'#Tilburg/msc/Thesis/Log'
#     # fh = open(f'logresults.txt', 'a')
#     # fh.write(f'{datetime.now()}\n ')
#     # fh.writelines(table_)
#     # line="".join(np.repeat('*',75))
#     # fh.write(f'\n {line} \n')
#     # fh.close()


traj_test_ub= 10000 
traj_est_primal_dual = 100000
traj_est_MM = 6000
traj_est_MM_belomestny2013=15000
K_lower=400
traj_lb_testing = 200000
information=[]
steps=9



basis_f_K_U_MM=150
basis_f_K_U_MM_belomestny2013=250
K_L_basic= 200
K_U_belomestny2009=300
K_U_SZH2013=150
K_L_AndersenBroadie2004=300
K_L_Glasserman2004=350
basis_f_K_U_MM_BHS=int(basis_f_K_U_MM*1.5)
grid_Glasserman2004=600
inner_sim_SZH=300

if __name__=='__main__':
    for d,s0,n_stopping_rights in [(4,100, 2)]:#[(2,90,9), (2,90, 8), (2,90, 7), (2, 90, 6), (2, 90, 5), (2,90,4), (2,90,3), (2,90,2), (2, 90, 1)]:
        for inner_sim in [100]:
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

                label_=f'Belom. et al. (2009) KU100-{steps}'
                print(''.join(np.repeat('-', 10)),label_, inner_sim, d, s0, n_stopping_rights, ''.join(np.repeat('-', 10)))
                list_inf=main(d, n_stopping_rights, True, grid=inner_sim, K_low=K_L_basic,K_up=K_U_belomestny2009, K_noise=50, traj_est=traj_est_primal_dual, traj_test_ub=traj_test_ub, traj_test_lb=traj_lb_testing, S_0=s0,  seed=i+8) 
                inf_list=utils.process_function_output(*list_inf, label_ = label_, grid= inner_sim, info_cols=inf_cols)
                information.append(inf_list)   


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
                

                # with open(f'run{datetime.now().strftime("%Y%m%d%H%m%S")}.pic', 'wb') as fh:
                #     pic.dump(information, fh)
                # repo.create_file(f'resultfiles/run{datetime.now().strftime("%Y%m%d%H%m%S")}.txt', f'run{datetime.now().strftime("%Y%m%d%H%m%S")}.txt', str(information), branch='main')

    table_ = tabulate(utils.information_format(information), headers=utils.header_, tablefmt="latex_raw", floatfmt=".4f")

    print(table_)



