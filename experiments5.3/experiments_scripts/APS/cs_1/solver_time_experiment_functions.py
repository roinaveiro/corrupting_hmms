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


def problem_mat(n_states, n_obs, t_len, dich_params = np.array([0.95, 0.8, 1.06]), seed_id = 10):
    
    np.random.seed(seed_id)
    trans_mat = np.random.dirichlet(np.ones(n_states)*dich_params[0], n_states)
    emiss_mat = np.random.dirichlet(np.ones(n_obs)*dich_params[1], n_states)
    prior_mat = np.random.dirichlet(np.ones(n_states)*dich_params[2])
    X = np.random.randint(0,n_states,t_len).reshape(-1,1)
    
    return {'trans': trans_mat, 'emiss': emiss_mat, 'prior': prior_mat, 'X':X}

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
                                                    'eps': 0.05, 'lr':0.005}}):
    
    if solver_str == 'APS':
        solver = aps_gibbs(attacker, **params_solver[solver_str])

    elif solver_str == 'RS':
        solver = nn_RS(attacker, **params_solver[solver_str])
    
    z_star, quality = solver.iterate(simulation_seconds = sim_sec)

    return {'ite': idx,'time':sim_sec,'q':quality, 
            'z': np.argmax(z_star, axis = 1), 
            'solver': solver_str, 
            'attacker': attacker2str(attacker), 
            '|Q|': attacker.n_components, 
            '|X|': ((attacker.X).shape)[0], 
            '|T|': attacker.n_obs,
             'rho': attacker.rho_probs[0],
              'k': attacker.k_value}


def parallel_iter_attacker_problem(attacker, solver_str, sim_sec, N= 10, 
                                   params_solver = {'APS': {'cooling_schedule':np.arange(500, 1000000, 10)},
                                   'RS': {'RS_iters': 5000, 'mcts_iters':100 ,
                                    'eps': 0.05, 'lr':0.005}},
                                     num_jobs = -2, save = False, 
                                     save_dir = '', exp_str = ''):
    
    
    
    res =  Parallel(n_jobs=num_jobs)(delayed(solve_attacker_problem)(attacker = attacker, 
                                                        solver_str = solver_str, 
                                                         sim_sec = sim_sec, 
                                                          idx = i , params_solver = params_solver) for i in range(N))
    
    res_df = pd.DataFrame(res)
    
    if save == True:
        
        set_up_str =  exp_str +'_'+ solver_str +'_'+attacker2str(attacker)+'_time_' + str(sim_sec) 
        hmm_str = '_Q_' + str(attacker.n_components) + '_X_'+ str(((attacker.X).shape)[0]) + '_T_' + str(attacker.n_obs)
        unc_str = '_rho_' + str(attacker.rho_probs[0]) + '_k_' + str(attacker.k_value)
        file_str = set_up_str + hmm_str + unc_str + '_one_iteration_.json'
        res_df.to_json(save_dir  + file_str, orient = 'split')
    
    return res_df
        


def parallel_time_attacker_problem(attacker, solver_str, t_list, N = 10, 
                                   params_solver = {'APS': {'cooling_schedule':np.arange(500, 1000000, 10)},
                                   'RS': {'RS_iters': 5000, 'mcts_iters':100 ,
                                    'eps': 0.05, 'lr':0.005}},
                                      num_jobs_time = 1, 
                                     num_jobs_ite =  1, 
                                      save_t = False,
                                       save_t_dir = '',
                                        save = False,
                                         save_dir = '',
                                          exp_str = ''):
        
    
    
    res =  Parallel(n_jobs=num_jobs_time)(delayed(parallel_iter_attacker_problem)(attacker,
                                                                                 solver_str,
                                                                                  sim_sec = t, 
                                                                                  N = N,
                                                                                  params_solver = params_solver, 
                                                                                  num_jobs = num_jobs_ite,
                                                                                   save = save_t,
                                                                                   save_dir = save_t_dir,
                                                                                    exp_str = exp_str)
                                                                                    for t in t_list)

    
    res_df = pd.concat(res).reset_index(drop = True)
    
    if save == True:
        set_up_str =  exp_str +'_'+ solver_str +'_'+attacker2str(attacker)
        hmm_str = '_Q_' + str(attacker.n_components) + '_X_'+ str(((attacker.X).shape)[0]) + '_T_' + str(attacker.n_obs)
        unc_str = '_rho_' + str(attacker.rho_probs[0]) + '_k_' + str(attacker.k_value)
        file_str = set_up_str + hmm_str + unc_str + '_all_.json'
        res_df.to_json(save_dir + file_str, orient =  'split')
        
    return res_df

