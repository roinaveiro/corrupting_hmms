import os
import numpy as np
import pandas as pd
from hmm_utils import HMM
from smoothing_state_attacker import ss_attacker
from smoothing_distribution_attacker import sd_attacker
from decoding_attacker import dec_attacker
from simulated_annealing import simulated_annealing
from aps_gibbs import aps_gibbs
from nn_RS.nn_RS import *
from scipy.special import softmax
from joblib import Parallel, delayed


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

def remove_scale_smoothing_state_attacker(utility, len_T, w1, w2):
    
    scale = w1 + w2 * len_T + 1
    
    return utility - scale

def remove_scale_smoothing_distribution_attacker(utility, len_T ,w1 ,w2):
    
    scale = w2 * len_T + 1
    
    return utility - scale

def remove_scale_decoding_attacker(utility, len_T, w1, w2):
    
    scale = w1* len_T + w2* len_T +1
    
    return utility - scale

def remove_scale_regardless_attacker(u, attacker):
    
    if type(attacker) ==  ss_attacker:
        scaled_eu = remove_scale_smoothing_state_attacker(u, attacker.T, attacker.w1, attacker.w2)
    elif type(attacker) == sd_attacker:
        scaled_eu = remove_scale_smoothing_distribution_attacker(u, attacker.T, attacker.w1, attacker.w2)
    elif type(attacker) == dec_attacker:
        scaled_eu = remove_scale_decoding_attacker(u, attacker.T, attacker.w1, attacker.w2)

    return scaled_eu 


def create_attacker_obj_exp2(att_st, comb_num, params_d_):
    
    att_param = params_d_[att_st]
    att_num_param = att_param[str(comb_num)]
    
    if att_st == 'atr':
        att = ss_attacker(params_d_['prior'],            
                     params_d_['trans'], 
                     params_d_['emiss'], 
                     att_num_param['rho'],            
                     params_d_['X'], 
                     w1 = att_param['w1'],
                     w2 = att_param['w2'] , 
                     t = att_param['t'],
                     state = att_param['state'],  
                     c = att_param['c'],   
                     k_value= att_num_param['k'])  
        
    elif att_st == 'rep':
        att = ss_attacker(params_d_['prior'],            
                     params_d_['trans'], 
                     params_d_['emiss'], 
                     att_num_param['rho'],             
                     params_d_['X'], 
                     w1 = att_param['w1'],
                     w2 = att_param['w2'] , 
                     t = att_param['t'],
                     state = att_param['state'],
                     c = att_param['c'],
                     k_value= att_num_param['k'])    
        
    elif att_st == 'dd':
        att = sd_attacker(params_d_['prior'],            
                         params_d_['trans'], 
                         params_d_['emiss'], 
                         att_num_param['rho'],              
                         params_d_['X'], 
                         w1 = att_param['w1'],
                         w2 = att_param['w2'] , 
                         t = att_param['t'], 
                         k_value= att_num_param['k'])  
    elif att_st == 'pd':
        att = dec_attacker(params_d_['prior'],            
                         params_d_['trans'], 
                         params_d_['emiss'], 
                         att_num_param['rho'],                      
                         params_d_['X'], 
                         w1 = att_param['w1'],
                         w2 = att_param['w2'] ,   
                          seq = att_param['seq'], 
                         k_value= att_num_param['k'])  
    
    return att



def solver_attacker_problem_cum_RS_single_iter_exp2(att_st, 
                                               comb_num,
                                               solver_str, 
                                               params_d_,
                                               idx):

    attacker = create_attacker_obj_exp2(att_st, comb_num, params_d_)

    minT = min(params_d_['time_range'])
    maxT = max(params_d_['time_range'])
    deltaT = params_d_['time_range'][1] - params_d_['time_range'][0]
    params_solver  = params_d_[att_st][str(comb_num)]['params_solver']
    
    if solver_str== 'SA':
        params_solver[solver_str]['mcts_iters'] = -1
    elif solver_str == 'MCTS':
        params_solver[solver_str]['sa_iters'] = -1

    rs = nn_RS(attacker, 
               flag= solver_str , 
               RS_iters = params_solver[solver_str]['RS_iters'], 
               mcts_iters = params_solver[solver_str]['mcts_iters'], 
               sa_iters = params_solver[solver_str]['sa_iters'], 
               eps = params_solver[solver_str]['eps'], 
               lr = params_solver[solver_str]['lr'])
    
    d_l = []
    sim_sec = 0
    for t in np.arange(minT, maxT + deltaT, deltaT):
        
        start = time.time()
        
        z_star, quality = rs.iterate(simulation_seconds = deltaT)
        
        sim_sec += deltaT 
        print(sim_sec)
        end = time.time()
        

        res_dict = {'ite': idx,
                    'time':sim_sec,
                    'q':quality,
                     'scaled_eu': remove_scale_regardless_attacker(quality,attacker),
                    'z': np.argmax(z_star, axis = 1), 
                    'solver': solver_str, 
                   'attacker': attacker2str(attacker), 
                   '|Q|': attacker.n_components, 
                   '|X|': attacker.n_obs, 
                   '|T|': attacker.T,
                    'rho': attacker.rho_probs[0],
                     'k': attacker.k_value,
                     'utility_time': end-start,
                     'RS_iters': params_solver[solver_str]['RS_iters'],
                     'mcts_iters': params_solver[solver_str]['mcts_iters'],
                      'sa_iters': params_solver[solver_str]['sa_iters'],
                       'eps': params_solver[solver_str]['eps'],
                        'lr': params_solver[solver_str]['lr']}
        
        d_l.append(res_dict)
    
    res_df = pd.DataFrame(d_l)
    
    save_dict = params_d_[att_st][str(comb_num)]
    
    if save_dict['save_ite_b'] == True:        
        set_up_str =  save_dict['id_str'] +'_'+ solver_str +'_'+attacker2str(attacker) + '_idx_'+ str(idx)
        hmm_str = '_Q_' + str(attacker.n_components) + '_X_'+ str(attacker.n_obs) + '_T_' + str(attacker.T)
        unc_str = '_rho_' + str(attacker.rho_probs[0]) + '_k_' + str(attacker.k_value)
        file_str = set_up_str + hmm_str + unc_str 
        res_df.to_json(save_dict[solver_str]['save_ite_d']  + file_str +'.json', orient = 'split')
        res_df.to_csv(save_dict[solver_str]['save_ite_d']  + file_str +'.csv', index = False)
        
    return res_df

def parallel_solver_attacker_problem_cum_RS_single_iter_exp2(att_st_, 
                                               comb_num_,
                                               solver_str_, 
                                               params_d__):
    
    N = params_d__['N']
    num_jobs = params_d__['n_jobs']
    
    
    res =  Parallel(n_jobs=num_jobs)(delayed(solver_attacker_problem_cum_RS_single_iter_exp2)(att_st = att_st_,
                                                       comb_num = comb_num_,
                                                       solver_str = solver_str_, 
                                                       params_d_ = params_d__,
                                                       idx = i) for i in range(N))
    
    res_df = pd.concat(res).reset_index(drop=True)
    save_dict = params_d__[att_st_][str(comb_num_)]
    
    if save_dict['save_gb_b'] == True:
        attacker = create_attacker_obj_exp2(att_st_, comb_num_, params_d__)
        set_up_str =  save_dict['id_str'] +'_'+ solver_str_ +'_'+att_st_
        hmm_str = '_Q_' + str(attacker.n_components) + '_X_'+ str(attacker.n_obs) + '_T_' + str(attacker.T)
        unc_str = '_rho_' + str(attacker.rho_probs[0]) + '_k_' + str(attacker.k_value)
        file_str = set_up_str + hmm_str + unc_str 
        res_df.to_json(save_dict[solver_str_]['save_gb_d']  + file_str +'.json', orient = 'split')
        res_df.to_csv(save_dict[solver_str_]['save_gb_d']  + file_str +'.csv', index = False)
        
    return res_df

