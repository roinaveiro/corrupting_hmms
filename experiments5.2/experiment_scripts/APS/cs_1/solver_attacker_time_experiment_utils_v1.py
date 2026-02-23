import numpy as np
from hmm_utils import HMM
from aps_gibbs_class import aps_gibbs
from smoothing_state_attacker import ss_attacker
from smoothing_distribution_attacker import sd_attacker
from decoding_attacker import dec_attacker
#from nn_RS.nn_RS import *
from joblib import Parallel, delayed
import pandas as pd
import json
import time


    
def problem_mat(n_obs, t_len, n_states, seed_id = 10):
    
    np.random.seed(seed_id)
    trans_mat = np.random.dirichlet(np.ones(n_states), n_states)
    emiss_mat = np.random.dirichlet(np.ones(n_obs), n_states)
    prior_mat = np.random.dirichlet(np.ones(n_states))
    X = np.random.randint(0,n_obs,t_len).reshape(-1,1)
    
    return {'trans': trans_mat, 'emiss': emiss_mat, 'prior': prior_mat, 'X':X}

def compute_prob_decoding_setting(mat_d):
    
    hmm = HMM(n_components=mat_d['trans'].shape[0], n_obs=mat_d['emiss'].shape[0])
    hmm.emissionprob_ = mat_d['emiss']
    hmm.transmat_     = mat_d['trans']
    hmm.startprob_    = mat_d['prior']
    
    chain_l = mat_d['X'].shape[0]
    d = {}
    max_val = -1
    max_state = 0
    min_val = 1.1
    min_state = 0
    t_min = 0
    t_max= 0
    
    for i in range(1,chain_l+1):
        vec = hmm.smoothing(mat_d['X'], i)
        if max_val < max(vec):
            max_val = max(vec) 
            max_state = np.argmax(vec)
            t_max = i
        if min_val > min(vec):
            min_val = min(vec)  
            min_state = np.argmin(vec) 
            t_min = i
            
            
    d['max_t'] = t_max
    d['max_state'] = max_state
    d['max_val']  = max_val
    
    d['min_t'] = t_min
    d['state_min'] = min_state
    d['val_min']  = min_val

    d['decode'] = hmm.decode(mat_d['X'])

    return d





def attacker2str(attacker):
    if type(attacker) ==  ss_attacker:
        if attacker.c == 1:
                atck_str = 'att'
        elif attacker.c == -1:
                atck_str = 'rep'
    elif type(attacker) == sd_attacker:
        atck_str = 'dd'
    elif type(attacker) == dec_attacker:
        atck_str = 'pd'

    return atck_str




def solve_attacker_problem(attacker, solver_str, sim_sec, idx = 0, 
                      params_solver = {'APS': {'cooling_schedule':np.arange(500, 1000000, 10)},
                                       'RS': {'RS_iters': 5000, 'mcts_iters':100 ,
                                                    'eps': 0.05, 'lr':0.005}}, 
                        save = False, save_dir = '', exp_str = '', flag_dir = ''):
    
    if solver_str == 'APS':
        solver = aps_gibbs(attacker, **params_solver[solver_str])

    elif solver_str == 'RS':
        solver = nn_RS(attacker, **params_solver[solver_str])
    
    
    start = time.time()
    
    z_star, quality = solver.iterate(simulation_seconds = sim_sec)
    
    end = time.time()

    res_dict = {'ite': idx,'time':sim_sec,'q':quality, 
            'z': np.argmax(z_star, axis = 1), 
            'solver': solver_str, 
            'attacker': attacker2str(attacker), 
            '|Q|': attacker.n_components, 
            '|X|': ((attacker.X).shape)[0], 
            '|T|': attacker.n_obs,
             'rho': attacker.rho_probs[0],
              'k': attacker.k_value,
               'utility_time': end-start}
    
    
    
    if save == True:        
        res_df = pd.DataFrame([res_dict])
        set_up_str =  exp_str +'_'+ solver_str +'_'+attacker2str(attacker)+'_time_' + str(sim_sec) + '_idx_'+ str(idx)
        hmm_str = '_Q_' + str(attacker.n_components) + '_X_'+ str(((attacker.X).shape)[0]) + '_T_' + str(attacker.n_obs)
        unc_str = '_rho_' + str(attacker.rho_probs[0]) + '_k_' + str(attacker.k_value)
        file_str = set_up_str + hmm_str + unc_str + '_one_iteration_'
        res_df.to_json(save_dir  + file_str +'.json', orient = 'split')
        with open(flag_dir + file_str + ".txt", "w") as text_file:
            text_file.write(solver_str+' --simulation with sec: ' + str(sim_sec) +' done!' + ' time to complete: '+ str(end-start))

    return res_dict




def parallel_iter_attacker_problem(attacker, solver_str, sim_sec, N= 10, 
                                   params_solver = {'APS': {'cooling_schedule':np.arange(500, 1000000, 10)},
                                   'RS': {'RS_iters': 5000, 'mcts_iters':100 ,
                                    'eps': 0.05, 'lr':0.005}},
                                     num_jobs = -2, save_un = False, save_dir_un = False, exp_dir_un = '',
                                   save = False, save_dir = '', exp_str = '',flag_dir = ''):
    
    
    
    
    
    res =  Parallel(n_jobs=num_jobs)(delayed(solve_attacker_problem)(attacker = attacker, 
                                                       solver_str =solver_str, 
                                                       sim_sec = sim_sec, 
                                                       idx = i ,
                                                       params_solver = params_solver, 
                                                       save = save_un, 
                                                       save_dir = save_dir_un, 
                                                       exp_str = exp_dir_un,
                                                       flag_dir = flag_dir) for i in range(N))
    
    res_df = pd.DataFrame(res)
    
    if save == True:
        
        set_up_str =  exp_str +'_'+ solver_str +'_'+attacker2str(attacker)+'_time_' + str(sim_sec) 
        hmm_str = '_Q_' + str(attacker.n_components) + '_X_'+ str(((attacker.X).shape)[0]) + '_T_' + str(attacker.n_obs)
        unc_str = '_rho_' + str(attacker.rho_probs[0]) + '_k_' + str(attacker.k_value)
        file_str = set_up_str + hmm_str + unc_str + '_N_iteration_.json'
        res_df.to_json(save_dir  + file_str, orient = 'split')
    
    return res_df



def parallel_time_attacker_problem(attacker, solver_str, t_list, N = 10, 
                                   params_solver = {'APS': {'cooling_schedule':np.arange(500, 1000000, 10)},
                                   'RS': {'RS_iters': 5000, 'mcts_iters':100 ,
                                    'eps': 0.05, 'lr':0.005}},
                                      num_jobs_time = 1, 
                                     num_jobs_ite =  1,
                                      save_un_ = False,
                                       save_dir_un_ = False,
                                       exp_dir_str = '',
                                      save_t = False,
                                       save_t_dir = '',
                                       exp_t_str = '',
                                        save = False,
                                         save_dir = '',
                                          exp_str = '',
                                          flag_dir = ''):
        
        
    
    
    res =  Parallel(n_jobs=num_jobs_time)(delayed(parallel_iter_attacker_problem)(attacker = attacker,
                                                                                 solver_str = solver_str,
                                                                                  sim_sec = t, 
                                                                                  N = N,
                                                                                  params_solver = params_solver, 
                                                                                  num_jobs = num_jobs_ite,
                                                                                   save_un = save_un_,
                                                                                   save_dir_un = save_dir_un_,
                                                                                  exp_dir_un = exp_dir_str, 
                                                                                   save = save_t,
                                                                                   save_dir = save_t_dir,
                                                                                    exp_str = exp_t_str,
                                                                                    flag_dir = flag_dir)
                                                                                    for t in t_list)

    
    res_df = pd.concat(res).reset_index(drop = True)
    
    if save == True:
        set_up_str =  exp_str +'_'+ solver_str +'_'+attacker2str(attacker)
        hmm_str = '_Q_' + str(attacker.n_components) + '_X_'+ str(((attacker.X).shape)[0]) + '_T_' + str(attacker.n_obs)
        unc_str = '_rho_' + str(attacker.rho_probs[0]) + '_k_' + str(attacker.k_value)
        file_str = set_up_str + hmm_str + unc_str + '_all_.json'
        res_df.to_json(save_dir + file_str, orient =  'split')
        
    return res_df






def parallel_ver_hor_solve_attck_problem(attacker, solver_str, time_list, N, 
                      params_solver = {'APS': {'cooling_schedule':np.arange(500, 1000000, 10)},
                                       'RS': {'RS_iters': 5000, 'mcts_iters':100 ,
                                                    'eps': 0.05, 'lr':0.005}}, num_jobs = -2, 
                                            save_ite = False, save_dir_ite = '', 
                                           save_glb = False , save_dir_glb = '', exp_str = '',
                                            flag_dir = ''):
    
    
    
    
    res = Parallel(n_jobs=num_jobs)(delayed(solve_attacker_problem)(attacker, solver_str, 
                                                              t, i, params_solver, save_ite, save_dir_ite, exp_str, flag_dir) for i in range(N) for t in time_list)
    
    res_df = pd.DataFrame(res)
    
    if save_glb == True:
        res_df = pd.DataFrame(res)
        set_up_str =  exp_str +'_'+ solver_str +'_'+attacker2str(attacker)
        hmm_str = '_Q_' + str(attacker.n_components) + '_X_'+ str(attacker.n_obs) + '_T_' + str(((attacker.X).shape)[0])
        unc_str = '_rho_' + str(attacker.rho_probs[0]) + '_k_' + str(attacker.k_value)
        file_str = set_up_str + hmm_str + unc_str + '_all.json'
        res_df.to_json(save_dir_glb  + file_str, orient = 'split')

    return res_df





