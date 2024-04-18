from tensorflow import keras, random, constant_initializer, compat, Variable, multiply
import os,sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import utils
import numpy as np
from sklearn import linear_model
import gurobipy as gb
from datetime import datetime
import scipy
import scipy.optimize

options = {
    "WLSACCESSID": "83a2a6cf-c6f9-4ac7-b149-f10663900b16",
    "WLSSECRET": "2dac01c6-2cbb-4e13-9a26-4a1df64a3e2b",
    "LICENSEID": 2418399,
}

class model_fBm():
    def __init__(self, model_seed, seed_keras,input_size_lb, K_lower, steps,d, K_upper, input_size_ub, K_upper_s=5, L=1, mode_kaggle=False):
        utils.set_seeds(seed_keras)

        connect_bias_lb=1 
        connect_bias_ub=1
        connect_w_lb=1
        connect_w_ub=1
        scale_=4
        initializer_w_lb=  [keras.initializers.HeNormal()]
        initializer_b_lb=  [constant_initializer(model_seed.normal( 0, scale_, size=(K_lower,)))] 
        layers_lb=[K_lower]
        activation_f_lower =keras.activations.softplus

        initializer_w_ub = [keras.initializers.HeNormal()]
        initializer_b_ub=  [constant_initializer(model_seed.normal( 0, scale_, size=(K_upper,)))]
        initializer_w_s= [keras.initializers.HeNormal()]
        initializer_b_s=  [constant_initializer(model_seed.normal( 0, scale_, size=(K_upper_s,)))]
        layers_ub=[K_upper]
        activation_f_upper = keras.activations.softplus 

        
        ## Lowerbound
        self.models=dict()
        self.models_ub=dict()
        self.models_ub_noise=dict()
        self.K=K_lower
        self.K_upper=K_upper
        self.d=d
        self.mode_kaggle=mode_kaggle
        self.steps=steps
        self.ex_rights=L
        for time in range(steps+1): #+1 needed for SAA
            mask_b_lb=[(np.random.rand(layer)<=connect_bias_lb) for layer in layers_lb]
            input_ly_size = input_size_lb*(time+1)
            layers_all_lb = [input_ly_size,*layers_lb]
            mask_w_lb=[(np.random.rand(layers_all_lb[i],layers_all_lb[i+1])<=connect_w_lb) for i in range(len(layers_all_lb)-1) ]
            model = keras.Sequential()
            model.add(keras.Input(shape=(input_ly_size,)))
            for layer_number, nodes in enumerate(layers_lb):
                model.add(keras.layers.Dense(nodes, activation=activation_f_lower, kernel_initializer= initializer_w_lb[layer_number], bias_initializer= initializer_b_lb[layer_number],  name=f'layer{layer_number}'))
                model.layers[-1].kernel.assign(multiply(model.layers[-1].kernel, mask_w_lb[layer_number]))
                model.layers[-1].bias.assign(multiply(model.layers[-1].bias, mask_b_lb[layer_number]))
            self.models[time]=model
        if L>1:
            self.theta_coeff = np.zeros((K_lower+1,L, steps))
        else:
            self.theta_coeff = np.zeros((K_lower+1, steps))

        ## Upperbound
        for time in range(steps):
            mask_b_ub=[(np.random.rand(layer)<=connect_bias_ub) for layer in layers_ub]
            input_ly_size = input_size_ub*(time+1)
            layers_all_ub = [input_ly_size,*layers_ub]
            mask_w_ub=[(np.random.rand(layers_all_ub[i],layers_all_ub[i+1])<=connect_w_ub) for i in range(len(layers_all_ub)-1) ]
            model_ub=keras.Sequential()
            model_ub.add(keras.Input(shape=(input_ly_size,)))
            for layer_number, nodes in enumerate(layers_ub):
                model_ub.add(keras.layers.Dense(nodes, activation=activation_f_upper, kernel_initializer= initializer_w_ub[layer_number], bias_initializer= initializer_b_ub[layer_number],  name=f'layer{layer_number}'))
                model_ub.layers[-1].kernel.assign(multiply(model_ub.layers[-1].kernel, mask_w_ub[layer_number]))
                model_ub.layers[-1].bias.assign(multiply(model_ub.layers[-1].bias, mask_b_ub[layer_number]))
            self.models_ub[time]=model_ub
        if L>1:
            self.beta_coeff_upper = np.zeros((K_upper, L, steps, d))
        else:
            self.beta_coeff_upper = np.zeros((K_upper, steps, d))
        
        ## Noise fitting
        for time in range(steps):
            model_ub_s = keras.Sequential()
            model_ub_s.add(keras.Input(shape=(input_size_ub*(time+1),)))
            for layer_number, nodes in enumerate([K_upper_s]):
                model_ub_s.add(keras.layers.Dense(nodes, activation=activation_f_upper, kernel_initializer= initializer_w_s[layer_number], bias_initializer= initializer_b_s[layer_number],  name=f'layer{layer_number}'))
            self.models_ub_noise[time] = model_ub_s
    
    random_basis_LS= lambda self, x, time: self.models[time](x, training=False).numpy() 
    random_basis_LS_upper= lambda self, x, time: self.models_ub[time](x, training=False).numpy()
    random_basis_LS_upper2= lambda self, x, time:  self.models_ub_noise[time](x, training=False).numpy() 

    def prediction_conval_model1(self, reg_m, traj, time, ex_right=0):
        if self.theta_coeff.ndim>2:
            theta=self.theta_coeff[:, ex_right, time]
        else:
            theta=self.theta_coeff[:, time]
        return np.hstack((np.ones((traj,1)), self.random_basis_LS(reg_m, time))) @ theta

    def prediction_conval_model2(self, reg_m, time, trajub, ex_right=0):
        if self.theta_coeff.ndim>2:
            theta=self.theta_coeff[:, ex_right, time]
        else:
            theta=self.theta_coeff[:, time]
        return  np.einsum('ijk, k-> ij',np.hstack((np.ones((reg_m.shape[0],1)), self.random_basis_LS(reg_m, time))).reshape(-1, trajub, self.K+1) , theta)


 
class model_HaughKaugen(model_fBm):
     def train_finallayer_continuationvalue(self, reg_m,y, time, traj, jump, ex_right=0):
        reg_mbasis_f= self.random_basis_LS(reg_m, time)
        reg_mbasis_f_W= np.hstack([ self.random_basis_LS_upper(reg_m, time)* (jump[:, d])[:, None] for d in range(jump.shape[1])])
        reg_functions_all = np.hstack((reg_mbasis_f,reg_mbasis_f_W))
        linear_regression_output= linear_model.LinearRegression().fit(reg_functions_all,y)       
    
        theta= np.hstack((linear_regression_output.intercept_, linear_regression_output.coef_[: reg_mbasis_f.shape[1]]))
        if self.theta_coeff.ndim>2:
            self.theta_coeff[:,ex_right, time]=theta
            self.beta_coeff_upper[:,ex_right, time, :]=linear_regression_output.coef_[reg_mbasis_f.shape[1]:].reshape(-1,jump.shape[1], order='F')
        else:
            self.theta_coeff[:,time]=theta
            self.beta_coeff_upper[:,time,:]=linear_regression_output.coef_[reg_mbasis_f.shape[1]:].reshape(-1,jump.shape[1], order='F')
        return np.hstack((np.ones((traj,1)), reg_mbasis_f)) @ theta
   

class model_Belomestny_etal(model_fBm):
    def train_finallayer_continuationvalue(self, reg_m,y, time, traj, jump, ex_right=0):
        reg_mbasis_f= self.random_basis_LS(reg_m, time)
        reg_mbasis_f_W= np.hstack([ self.random_basis_LS_upper2(reg_m, time)* (jump[:, d])[:, None] for d in range(jump.shape[1])])
        reg_functions_all = np.hstack((reg_mbasis_f,reg_mbasis_f_W))
        linear_regression_output= linear_model.LinearRegression().fit(reg_functions_all,y)       
    
        theta= np.hstack((linear_regression_output.intercept_, linear_regression_output.coef_[: reg_mbasis_f.shape[1]]))
        if self.theta_coeff.ndim>2:
            self.theta_coeff[:,ex_right, time]=theta
        else:
            self.theta_coeff[:,time]=theta
        return np.hstack((np.ones((traj,1)), reg_mbasis_f)) @ theta
    
    def train_finallayer_continuationvalue_upper(self, reg_m, y, time, traj, ex_right=0):
        reg_mbasis_f= self.random_basis_LS_upper(reg_m, time).astype(np.float32)
        linear_regression_output= linear_model.LinearRegression(fit_intercept=False).fit(reg_mbasis_f,y)
        beta= linear_regression_output.coef_# np.hstack((linear_regression_output.intercept_[:,None], linear_regression_output.coef_))
        if self.beta_coeff_upper.ndim>3:
            self.beta_coeff_upper[:,ex_right, time,:]=beta.T
        else:
            self.beta_coeff_upper[:,time,:]=beta.T
        return reg_mbasis_f @ beta.T # np.hstack((np.ones((traj,1)), reg_mbasis_f)) @ beta
     
       
    def prediction_Z_model_upper(self, reg_m, traj, time, jump, ex_right=0):
        if self.beta_coeff_upper.ndim>3:
            beta=self.beta_coeff_upper[:, ex_right, time]
        else:
            beta=self.beta_coeff_upper[:, time]
        trans_bf_coeff= self.random_basis_LS_upper(reg_m, time) @ beta
        return  np.einsum('sd, sd-> s',trans_bf_coeff, jump)
      
        
    def prediction_Z_model_upper2(self, reg_m, traj, fine_grid, jump, ex_right=0):
        if self.beta_coeff_upper.ndim>3:
            beta=self.beta_coeff_upper[:, ex_right, :]
        else:
            beta=self.beta_coeff_upper
        s_dim = reg_m.shape[0]
        t_dim = reg_m.shape[1]
        # test_123 = self.prediction_Z_model_upper(reg_m[:,400,:], traj, 4, jump[:, 400, :],0)
        reg_m= reg_m.reshape(-1, reg_m.shape[-1], order='F')
        
        t_ = np.repeat(np.arange(t_dim//fine_grid), fine_grid)
        trans_bf= self.random_basis_LS_upper(reg_m, t_).reshape(t_dim, s_dim, self.K_upper)
        beta_coeff= np.repeat(beta, fine_grid, axis=1).T
        trans_bf_coeff =  np.einsum('tsk,dtk->std', trans_bf, beta_coeff)
        Z_W= np.einsum('std, std-> st', trans_bf_coeff, jump)
        return Z_W



class model_Schoenmakers(model_fBm):
    def __init__(self,model_seed, seed_keras, input_size_lb, K_lower, steps,d, K_upper, input_size_ub, layers_ub_s=5, L=1, mode_kaggle=False):
        super().__init__(model_seed, seed_keras, input_size_lb, K_lower, steps,d, K_upper, input_size_ub, layers_ub_s, L, mode_kaggle)
        if L>1:
            self.beta_coeff_upper = np.zeros((K_upper//self.d, L, steps))
        else:
            self.beta_coeff_upper = np.zeros((K_upper//self.d, steps))

    def train_finallayer_continuationvalue(self, reg_m,y, time, traj, jump,dt, mode=0, ex_right=0):
        reg_mbasis_f= self.random_basis_LS(reg_m, time)
        if mode ==0: # Mode=0: no denoise terms
            trans_basisf=self.random_basis_LS_upper(reg_m, time).reshape(-1, self.d, self.K_upper//self.d)
            reg_mbasis_f_W = np.einsum('ijk,ij->ik', trans_basisf, jump)
            reg_functions_all = np.hstack((reg_mbasis_f,reg_mbasis_f_W))
            linear_regression_output= linear_model.LinearRegression().fit(reg_functions_all,y)
        else:
            trans_basisf=self.random_basis_LS_upper(reg_m, time).reshape(-1, self.d, self.K_upper//self.d)
            reg_mbasis_f_W = np.einsum('ijk,ij->ik', trans_basisf, jump)

            trans_basisf_denoise= np.hstack([ self.random_basis_LS_upper(reg_m, time)* (jump[:, d]**2-dt)[:, None] for d in range(jump.shape[1])])
            reg_functions_all = np.hstack((reg_mbasis_f,reg_mbasis_f_W,trans_basisf_denoise))
            linear_regression_output= linear_model.LinearRegression().fit(reg_functions_all,y)
        theta=  np.hstack((linear_regression_output.intercept_, linear_regression_output.coef_[: self.K]))
        if self.theta_coeff.ndim>2:
            self.theta_coeff[:,ex_right, time]=theta
            self.beta_coeff_upper[:,ex_right, time]=linear_regression_output.coef_[self.K: self.K_upper//self.d+ self.K]
        else:
            self.theta_coeff[:,time]=theta
            self.beta_coeff_upper[:,time]=linear_regression_output.coef_[self.K: self.K_upper//self.d+ self.K]

        return np.hstack((np.ones((traj,1)), reg_mbasis_f)) @ theta
    
    def prediction_Z_model_upper(self, reg_m, traj, time, jump, ex_right=0):
        if self.beta_coeff_upper.ndim>2:
            beta=self.beta_coeff_upper[:, ex_right, time]
        else:
            beta=self.beta_coeff_upper[:, time]
        trans_bf= self.random_basis_LS_upper(reg_m, time).reshape(-1, self.d, self.K_upper//self.d)
        reg_mbasis_f_W = np.einsum('ijk,ij->ik', trans_bf, jump)
        return reg_mbasis_f_W@ beta
      
        
    def prediction_Z_model_upper2(self, reg_m, traj, fine_grid, jump, ex_right=0):
        if self.beta_coeff_upper.ndim>2:
            beta=self.beta_coeff_upper[:, ex_right, :]
        else:
            beta=self.beta_coeff_upper
        s_dim = reg_m.shape[0]
        t_dim = reg_m.shape[1]
        reg_m= reg_m.reshape(-1, reg_m.shape[-1], order='F')
        t_ = np.repeat(np.arange(t_dim//fine_grid), fine_grid)
        trans_bf= self.random_basis_LS_upper(reg_m, t_).reshape(-1, self.d, self.K_upper//self.d)
        reg_mbasis_f_W = np.einsum('ijk,ij->ik', trans_bf, jump.reshape(-1, self.d, order='F')).reshape(s_dim, t_dim, self.K_upper//self.d, order='F' )
        return np.einsum('stk, tk-> st', reg_mbasis_f_W , np.repeat(beta, fine_grid, axis=1).T)


class model_SAA(model_HaughKaugen):
    """
    Class which applies to problems of 4 papers (Desai et al., 2012; Belomestny, Bender, Schoenmakers, 2024; Belomestny, Hildebrand, Schoenmakers, 2017; Belomestny 2013).
        * Desai et al. 2012: min_psi mean_i \max_t (\G_t^i-M_t^i(psi))
        * Belomestny, Bender, Schoenmakers, 2024:  min_psi mean_i \max_t (\G_t^{i,A}-M_t^i(psi)); in which \G_t^{i,A}=\G_t^{i} if t>=1 and A^i otherwise, in which A^i is randomly drawn.
        * Belomestny, Hildebrand, Schoenmakers, 2017: min_psi max_i \max_t (\G_t^i-M_t^i(psi)).
        * Belomestny, 2013:  z_p^i= 1/p log(p*exp(\sum_t G_t^i-M_t^i)) and min mean(z_p^i) + lambda * \sqrt{1/(N-1) * [\sum z_p^i - mean(z_p^i) ]}.
    In which G_t^i denotes the discounted payoff at time t for trajectory i and M_t^i(psi) the constructed martingale respectively.
    """

    def LP_multiple(self, payoff_paths, created_martingale_incrs, print_progress=True, mode_BHS=False, lasso_penalty = 1/100):
        """
        Function which applies Linear problems of 3 papers (Desai et al., 2012; Belomestny, Bender, Schoenmakers, 2024; Belomestny, Hildebrand, Schoenmakers, 2017).
        *(mode_BHS=False) Desai et al. 2012: min_psi mean_i \max_t (\G_t^i-M_t^i(psi))
        *(mode_BHS=False) Belomestny, Bender, Schoenmakers, 2024:  min_psi mean_i \max_t (\G_t^{i,A}-M_t^i(psi)); in which \G_t^{i,A}=\G_t^{i} if t>=1 and A^i otherwise, in which A^i is randomly drawn.
        *(mode_BHS=True) Belomestny, Hildebrand, Schoenmakers, 2017: min_psi max_i \max_t (\G_t^i-M_t^i(psi)).

        Due to overfitting, in all above formulations a lasso penalty is added based on \psi. 
        Additionally, to penaltise in a fair way, the martingale increments are scaled such that they have all a standard error of 1.
        Due to numerical instabilities in the optimisation software, increments with a standard error below 0.0001 are considered as 0 (so not considered at all), just ass increments which are after scaling smaller than 0.0001

        Input:
            payoff_paths: np.array() of shape (N, T+1) in which N denotes the number of sample trajectories and T the number of time steps; (T+1) due to time 0 included.
            created_martingale_incrs: np.array() of shape (N, T+1, K) in which N denotes the number of sample trajectories and T the number of time steps; (T+1) due to time 0 included. K denotes the number of martingale increments from the basis functions.
            mode_BHS: See above. True for algorithm  Belomestny, Hildebrand, Schoenmakers, 2017 and False otherwise.

        Output:
           r_res: np.array() of shape (K, self.ex_rights, T+1), with optimal coefficients for martingale increments (to create best martingale).
           u_opt.mean(): ptimal function value minimising objective final right training set.
           compile_time: Compile time to compile Gurobi model.
           solver_time: Time required by the solver to solve the problem.
        """
        created_martingale_incrs_adj= np.copy(created_martingale_incrs)
        n= int(np.shape(payoff_paths)[0])
        t = self.steps
        p= self.K # Use randomised NN as basis functions
        env_ = gb.Env(params=options) if self.mode_kaggle==True else gb.Env()
        with env_ as env, gb.Model(env=env) as LP:
            if print_progress==False:
                LP.Params.OutputFlag = 0 
            LP.Params.BarHomogeneous=0
            LP.Params.NumericFocus=2
            LP.Params.OptimalityTol=10**-5
            LP.Params.BarConvTol=10**-5
            LP.Params.Crossover=0
            LP.Params.Method=2
            if n>20000 and self.K>150 and self.mode_kaggle==False: # Set Gurobi settings for large model: Reduce Threads to 1 and set Presolve to 2.
                LP.Params.Presolve=2
                LP.Params.Threads=1
            r = LP.addMVar((p, t), vtype=gb.GRB.CONTINUOUS, lb=-1*gb.GRB.INFINITY) #gb.MVar.fromlist([[LP.addVar(vtype=gb.GRB.CONTINUOUS, lb=-1*gb.GRB.INFINITY) for time in range(t)] for p_i in range(p)])
            g = LP.addMVar((p, t), vtype=gb.GRB.CONTINUOUS, lb=-1*gb.GRB.INFINITY) #LASSO penalty term 
            LP.addConstr(g>=r)#LASSO penalty term 
            LP.addConstr(g>=-1*r)#LASSO penalty term 
            if mode_BHS==False:
                u = LP.addMVar((n,), vtype=gb.GRB.CONTINUOUS, lb=payoff_paths[:,0], name='u_VAR')
                LP.setObjective(1/n*gb.quicksum(u[i] for i in range(n))+lasso_penalty*g.sum(), gb.GRB.MINIMIZE) #LASSO penalty term 
            else: # Type Belomestny Hildebrand Schoenmakers maximum penaltisation-> focus on worst case maximum difference rather than average
                u = LP.addVar(vtype=gb.GRB.CONTINUOUS, lb=np.max(payoff_paths[:,0]))
                LP.setObjective(u+lasso_penalty*g.sum(), gb.GRB.MINIMIZE) #LASSO penalty term 
            
            scaler_Mart_incr = np.std(created_martingale_incrs, axis=0) # np.max(np.abs(created_martingale_incrs_adj), axis=0) #
            mask_instably_close0= scaler_Mart_incr<0.0001
            created_martingale_incrs_adj=np.where(np.tile(mask_instably_close0[None,:,:], [n, 1, 1]), 0, created_martingale_incrs_adj/np.tile(scaler_Mart_incr[None,:,:], [n, 1, 1]))
            mask_instably_close0_solo = np.abs(created_martingale_incrs_adj)<0.00001
            created_martingale_incrs_adj=np.where(mask_instably_close0_solo, 0 , created_martingale_incrs_adj)
            t0 = datetime.now()
            step_loop=5000
            innerloop = np.arange(0,n+step_loop, step_loop)
            innerloop[innerloop>=n]=n
            innerloop=np.unique(innerloop)
            
            innerloop= [(innerloop[i], innerloop[i+1]) for i in range(len(innerloop)-1)]
            if mode_BHS==False:
                for time_s in range(1,t+1):
                    for inner_prev, inner_next in innerloop:
                        LP.addConstr(u[inner_prev:inner_next]+(created_martingale_incrs_adj[inner_prev:inner_next, :time_s, :].T* (r[:,:time_s])[:,:,None]).sum((0,1)) >=payoff_paths[inner_prev:inner_next, time_s],  name=f"constraintMAX_{inner_prev}_{time_s}")
            else: # Type Belomestny Hildebrand Schoenmakers maximum penaltisation-> focus on worst case maximum difference rather than average
                for time_s in range(1,t+1):
                    for inner_prev, inner_next in innerloop:
                        LP.addConstr(u+(created_martingale_incrs_adj[inner_prev:inner_next, :time_s, :].T* (r[:,:time_s])[:,:,None]).sum((0,1)) >=payoff_paths[inner_prev:inner_next, time_s], name=f"constraintMAX_{inner_prev}_{time_s}")
            t1= datetime.now()
            compile_time = t1-t0
            solver_time=0
            LP.optimize()
            solver_time+= LP.Runtime
            r_opt=np.array([[r[p_i][time].X  for time in range(t)] for p_i in range(p)]) 
            r_opt=np.where(mask_instably_close0.T, 0, r_opt/scaler_Mart_incr.T)
            u_opt= np.array([u[i].X for i in range(n) ]) if mode_BHS==False else np.array([u.X])

            theta_upperbound2=np.zeros((n, self.ex_rights, t+1))
            r_coeff=[r_opt]
            M_opt= np.hstack((np.zeros((n,1)), np.cumsum(np.sum(created_martingale_incrs*r_opt.T, axis=-1),axis=1) ))
            upperbound_recursive_series=np.maximum.accumulate((payoff_paths-M_opt)[:, self.steps::-1], axis=1)[:, ::-1] + M_opt[:, :self.steps+1]
            theta_upperbound2[:,0, :self.steps+1] =  np.copy(upperbound_recursive_series)
            # If another level of exercise right is added, only the right hand side of the constraints changes and the lowerbound of u. Select these.
            names_cons=[c.ConstrName for c in LP.getConstrs() if 'constraintMAX' in c.ConstrName] 
            name_u_vars= [v.VarName for v in LP.getVars() if 'u_VAR' in v.VarName]
            for ex_right in range(1, self.ex_rights):
                M_prevlevel = np.zeros((n, self.steps-ex_right+2))
                #:self.steps-ex_right+1 as last coefficients 0. Can only exercise up to self.steps-ex_right+1 
                M_prevlevel[:,1:]= np.cumsum(np.sum(created_martingale_incrs[:, :self.steps-ex_right+1,:]*(r_coeff[ex_right-1].T[:self.steps-ex_right+1,:]), axis=-1),axis=1)
                M_incr_prevlevel= np.diff(M_prevlevel, axis=1) 
                new_RHS =  payoff_paths[:,:self.steps-ex_right+1]- M_incr_prevlevel + theta_upperbound2[:,ex_right-1, 1:self.steps-ex_right+2] 
                new_RHS=np.hstack((new_RHS, np.tile(np.min(payoff_paths)-1 , [n, ex_right]) ))# Make sure that max not due to final values-> exercise before or equal to #rights left == times left. If #times left<#rights left: set to constraint to become unbinding

                #new_RHS[:,self.steps-ex_right+1:]=np.min(payoff_paths)-1 # Make sure that max not due to final values-> exercise before or equal to #rights left == times left. If #times left<#rights left: set to constraint to become unbinding
                constraints = [LP.getConstrByName(name) for name in names_cons]
                u_var = [LP.getVarByName(name) for name in name_u_vars]
                t0=datetime.now()
                LP.setAttr('RHS', constraints, new_RHS[:,1:].flatten('F'))
                if mode_BHS==False:
                    LP.setAttr('LB', u_var, new_RHS[:,0])
                else:
                    LP.setAttr('LB', u_var, np.max(new_RHS[:,0])) # Type Belomestny Hildebrand Schoenmakers maximum penaltisation-> focus on worst case maximum difference rather than average
                t1=datetime.now()
                compile_time+=t1-t0
                LP.optimize()
                solver_time+= LP.Runtime

                r_opt=np.array([[r[p_i][time].X  for time in range(t)] for p_i in range(p)]) 
                r_opt=np.where(mask_instably_close0.T, 0, r_opt/scaler_Mart_incr.T)
                r_opt[:,self.steps-ex_right:]=0 # Make sure that max not due to final values-> exercise before or equal to #rights left == times left. If #times left<#rights left: set to constraint to become unbinding. Therefore final M_increment not relevant and set to 0.
                u_opt= np.array([u[i].X for i in range(n)]) if mode_BHS==False else np.array([u.X])

                r_coeff.append(r_opt)
                M_opt= np.hstack((np.zeros((n,1)), np.cumsum(np.sum(created_martingale_incrs*r_opt.T, axis=-1),axis=1) ))
                upperbound_recursive_series=np.maximum.accumulate((new_RHS-M_opt)[:, self.steps-ex_right::-1], axis=1)[:, ::-1] + M_opt[:, :self.steps+1-ex_right]
            # upperbound_recursive_series[:, self.steps+1-ex_right:]=new_RHS[:, self.steps+1-ex_right:] # Less rights left than ex rights: exercise right
                upperbound_recursive_series=np.hstack((upperbound_recursive_series, new_RHS[:, self.steps+1-ex_right:])) # Less rights left than ex rights: exercise right
            
                theta_upperbound2[:,ex_right,:] =  np.copy(upperbound_recursive_series)
            r_res = np.zeros((self.K,self.ex_rights, self.steps))
            for right_num,r_right in enumerate(r_coeff):
                r_res[:, right_num,:]= r_right

        return r_res, u_opt.mean(), compile_time, solver_time
    
    


    def LP_BELOM_BFGS_multiple(self, payoff_paths, created_martingale_incrs, lambda_=3, p=50, print_progress=True, calc_nonsmoothenedTrainingUB= False):
        """
        Function which applies BFGS to problem in belomestny (2013) with z_p^i= 1/p log(p*exp(\sum_t G_t^i-M_t^i)) and min mean(z_p^i) + lambda * \sqrt{1/(N-1) * [\sum z_p^i - mean(z_p^i) ]}.
            Stopping criterion in BFGS based on maximum absolute value in Jacobian (below  0.005).

        Input:
            payoff_paths: np.array() of shape (N, T+1) in which N denotes the number of sample trajectories and T the number of time steps; (T+1) due to time 0 included.
            created_martingale_incrs: np.array() of shape (N, T+1, K) in which N denotes the number of sample trajectories and T the number of time steps; (T+1) due to time 0 included. K denotes the number of martingale increments from the basis functions.
            lambda_: lambda parameter in objective.
            p: smoothen parameter (default 50).
            print_progress: printing the progress of the BFGS optimisation (default True).
            calc_nonsmoothenedTrainingUB: calculating upperbound non-smoothened training trajectories (default False).

        Output:
           r_res: np.array() of shape (K, self.ex_rights, T+1), with optimal coefficients for martingale increments (to create best martingale).
           theta_upperbound2: theta in Balder, Mahayni and Schoenmakers (2013), corresponding to non-smoothened training trajectories.
                    Only calculated if calc_nonsmoothenedTrainingUB=True, else return -1
           res.fun: optimal function value minimising objective final right on smoothened maxima training set.
        """

        n= int(np.shape(payoff_paths)[0])
        K = np.shape(created_martingale_incrs)[-1]
        sqrt_term_objective_var=lambda_*(1/(n-1))**0.5        
        def objective_BFGS(params,created_martingale_incrs, series_prev_level=payoff_paths, p=p):
            M_ = np.zeros((n, self.steps-ex_right+1))
            M_[:,1:]= np.cumsum(np.sum(created_martingale_incrs*params.reshape(self.steps-ex_right,K), axis=-1),axis=1)
            time_0_recursive_ub=(series_prev_level-M_)[:,:1+self.steps-ex_right]
            p_G_minus_M= p*(time_0_recursive_ub)
            # M_= np.hstack((np.zeros((n,1)), np.cumsum(np.sum(created_martingale_incrs*params.reshape(t,K), axis=-1),axis=1)))
            ## See (4.5) in Belomestny-> smoothen Z=max_t(g(S_t)-M_t)
            ## Alternative calculation of Z_p to avoid overflow error-> substract maximum 
            max_val = np.max(p_G_minus_M, axis=1)
            int_inner_term = np.exp(p_G_minus_M - max_val[:,None])
            int_inner_term_sum =  np.sum(int_inner_term, axis=-1)
            u = (np.log(int_inner_term_sum) + max_val)/p
            avg_term= np.mean(u)
            diff_term_var = u-avg_term
            std_term = np.sqrt(diff_term_var@diff_term_var)
            fun = avg_term + sqrt_term_objective_var*std_term

            d_expterm_d_psi_div_p= -np.cumsum(int_inner_term[:,:0:-1], axis=1)[:,::-1,None]  * created_martingale_incrs
            jac=np.tensordot((1/n + sqrt_term_objective_var*(diff_term_var/std_term))/int_inner_term_sum, d_expterm_d_psi_div_p, axes=(0,0)).flatten()
            return fun, jac
        

        def callback(x,rel_mart_incr, series_prev_level, p):
            """
            Callback to print progress BFGS optimisation.
            """
            if callback.iteration%10==0:
                val, jac = objective_BFGS(x,rel_mart_incr, series_prev_level, p)
                print(f"Iteration: {callback.iteration}, Objective Value: {val}   - Norm Jacobian {np.linalg.norm(jac)} - Max abs Jacobian {np.max(np.abs(jac))}")
            callback.iteration += 1 
        callback.iteration = 0

        r_coeff=[]
        theta_upperbound2=np.zeros((n, self.ex_rights, self.steps+1))
        for ex_right in range(self.ex_rights):
            if ex_right==0:
                series_prev_level = payoff_paths
            else:
                M_prevlevel = np.zeros((n, self.steps-ex_right+2))
                M_prevlevel[:,1:]= np.cumsum(np.sum(created_martingale_incrs[:, :self.steps-ex_right+1,:]*r_coeff[ex_right-1].reshape(self.steps-ex_right+1,K), axis=-1),axis=1)
                M_incr_prevlevel= np.diff(M_prevlevel, axis=1) 
                #series_prev_level=np.zeros((n, self.steps-ex_right))
                series_prev_level =  payoff_paths[:,:self.steps-ex_right+1]- M_incr_prevlevel + theta_upperbound2[:,ex_right-1, 1:2+self.steps-ex_right]
                series_prev_level=np.hstack((series_prev_level, (payoff_paths[:,-1])[:,None] ))
                series_prev_level=series_prev_level[:,:self.steps-ex_right+1]
            x_0 = np.zeros(K*(self.steps-ex_right))
            rel_mart_incr= created_martingale_incrs[:, :self.steps-ex_right, :]
            # for p_ in [1, 5, p]:
            p_=p
            if print_progress:
                res=scipy.optimize.minimize(objective_BFGS,x_0, jac=True, method='BFGS', args=(rel_mart_incr, series_prev_level,p_), options={'disp': True, 'gtol': 0.005, 'maxiter': 2000, 'norm': 2 }, callback= lambda x: callback(x,rel_mart_incr, series_prev_level, p_))
            else:
                res=scipy.optimize.minimize(objective_BFGS,x_0, jac=True, method='BFGS', args=(rel_mart_incr, series_prev_level, p_), options={'disp': False, 'gtol': 0.005, 'maxiter': 2000, 'norm': 2})
                # x_0=res.x
            r_opt = res.x
            r_coeff.append(r_opt)
            M_opt= np.hstack((np.zeros((n,1)), np.cumsum(np.sum(rel_mart_incr*r_opt.reshape(self.steps-ex_right,K), axis=-1),axis=1) ))
            upperbound_recursive_series = np.zeros_like(payoff_paths)
            upperbound_recursive_series=np.maximum.accumulate((series_prev_level-M_opt)[:, self.steps-ex_right::-1], axis=1)[:, ::-1] + M_opt[:, :self.steps+1-ex_right]
            upperbound_recursive_series[:, self.steps+1-ex_right:]=series_prev_level[:, self.steps+1-ex_right:] # Less rights left than ex rights: exercise right
            theta_upperbound2[:,ex_right, :self.steps-ex_right+1] =  np.copy(upperbound_recursive_series)

            print(res.success)
            print(upperbound_recursive_series[:,0].mean())
            if res.success==False:
                raise Exception('Did not converge!')
      
        r_res = np.zeros((K,self.ex_rights, self.steps))
        for right_num,r in enumerate(r_coeff):
            r_res[:, right_num,:self.steps - right_num]= r.reshape(self.steps- right_num, -1).T
        if calc_nonsmoothenedTrainingUB:
            ### NON SMOOTHENED UB
            terminal_payoff2=payoff_paths[:,-1]
            theta_upperbound2= np.zeros((n, self.ex_rights, self.steps+1))
            theta_upperbound2[:,:,-1]=terminal_payoff2[:,None]
            M_incr=np.zeros((n, self.ex_rights, self.steps))
            for t_fine in range(self.steps):
                M2_train_bf_t = created_martingale_incrs[:,t_fine//1,:]
                for ex_right in range(self.ex_rights):
                    M_incr[:,ex_right, t_fine]= M2_train_bf_t@ r_res[:,ex_right, t_fine]
            for ex_right in range(self.ex_rights):
                series_prev_level = payoff_paths[:,:-1]- M_incr[:, ex_right-1, :] + theta_upperbound2[:,ex_right-1, 1:] if ex_right>0 else payoff_paths[:,:-1]
                series_prev_level=np.hstack((series_prev_level, terminal_payoff2[:,None]))[:, :self.steps-ex_right+1]
                series_cur_level = np.zeros_like(payoff_paths)
                series_cur_level[:,-1] = payoff_paths[:,-1]
                M_= np.hstack((np.zeros((n, 1)), np.cumsum(M_incr[:,ex_right,:], axis=-1)))
                M_ = M_[:, :self.steps-ex_right+1]
                theta_upperbound2[:, ex_right, :self.steps+1-ex_right]= np.maximum.accumulate((series_prev_level-M_)[:, self.steps-ex_right::-1], axis=1)[:, ::-1] + M_[:, :self.steps+1-ex_right]
        else:
            theta_upperbound2=-1
        return r_res, theta_upperbound2, res.fun
    
    