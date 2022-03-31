import numpy as np
from hmm_utils import HMM
from mc_ennumeration import MC_enumeration, MC_enumeration_parallel
from smoothing_distribution_attacker import sd_attacker
from params import *
from scipy.special import softmax
from collections import Counter
import time 
from joblib import Parallel, delayed
from scipy.special import rel_entr


def Z_2_vec(Z_set):
    z_str_list = []
    Z_set_vec = np.argmax(Z_set ,axis=2)
    for z in Z_set_vec:
        z_str_list.append("".join(list(map(str,z.reshape(-1)))))
    return z_str_list


def utl_w_sd_attacker(X ,priors, transition, emission, rho_probs, t_, k_, N_, num_jobs_=-2):

    f1_att = sd_attacker(priors, transition, emission, rho_probs,X = X, w1 = 1.0, w2 = 0.0 , t = t_, k_value= k_)
    f2_att =  sd_attacker(priors, transition, emission, rho_probs,X = X, w1 = 0.0, w2 = 1.0 , t = t_, k_value= k_)
    utl_l1 = MC_enumeration_parallel(f1_att, N= N_, verbose = False, num_jobs = num_jobs_)[1]   #joblib
    utl_l2 = MC_enumeration(f2_att, N = 1, verbose = False)[1]  #joblib
    
    return f1_att ,utl_l1, utl_l2


def exp_kl_div(hmm_D, attacker, z_star, N_):
    
    kl_div_arr = np.zeros(N_)
    for n in range(N_):
        y_vec = attacker.attack_X(rho_matrix = attacker.rho_probs, z_matrix = z_star)
        p_clean_data = hmm_D.smoothing((attacker.X).astype(int), attacker.t)
        p_tn_data = hmm_D.smoothing(y_vec.astype(int), attacker.t)
        # KL divergence
        kl_div = np.sum(rel_entr(p_clean_data, p_tn_data)) 
        kl_div_arr[n] = kl_div
    return np.mean(kl_div_arr)


def exp_kl_ratio_w1_w2(hmm_D, attacker, utl_l1, utl_l2, Z_set, fn_rt, init_rt, rt_st, N2, num_jobs =-2):
    
    rt_l = np.arange(init_rt,fn_rt, rt_st)
    utl_vec = rt_l.reshape(len(rt_l),1) * utl_l1 + utl_l2
    idx = np.argmax(utl_vec, axis = 1)
    z_star_arr = Z_set[idx]    
    res_l = Parallel(n_jobs=num_jobs)(delayed(exp_kl_div)(hmm_D, attacker, z_star, N2) for z_star in (z_star_arr))
      
    return rt_l, res_l, z_star_arr


def get_w1_w2_grid(start, stop, n_values):
    
    w1_vals = np.linspace(start, stop, n_values)
    w2_vals = np.linspace(start, stop, n_values)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)
    
    return W1, W2


def get_kl_div_grid_w1_w2(hmm_D,attacker,utl_1, utl_2, Z_set, W1_loop, W2_loop, N_):
    
    z_star_arr = []
    res_l = []
    for element in (range(len(W1_loop))):
        utl_vec = W1_loop[element] * utl_1 + W2_loop[element] * utl_2
        max_indx = np.argmax(utl_vec)
        z_star = Z_set[max_indx]
        z_star_arr.append(z_star)
        exp_kl = exp_kl_div(hmm_D, attacker, z_star, N_)
        res_l.append(exp_kl)
    return z_star_arr,res_l


        
def KL_div_w1_w2_ratio_contour_box_experiment(X, hmm_D, rho_probs, t_, k_, N1, N2, params_dict, num_jobs = -2):
    
    trans_mat = hmm_D.transmat_
    emiss_mat = hmm_D.emissionprob_
    prior_mat = hmm_D.startprob_
    
    att_obj ,utl_l1, utl_l2 = utl_w_sd_attacker(X, prior_mat, trans_mat, emiss_mat, rho_probs, t_, k_, N1, num_jobs)

    Z_set = att_obj.generate_attacks()
    
    res_d = {}
    
    if 'ratio' in params_dict:
        
        init_rt = params_dict['ratio']['init_rt']
        fn_rt = params_dict['ratio']['fn_rt']
        rt_st  = params_dict['ratio']['rt_st']
        rt_list, res_list, z_star_arr = exp_kl_ratio_w1_w2(hmm_D, att_obj,utl_l1, utl_l2, Z_set, fn_rt, init_rt, rt_st, N2, num_jobs)
        res_d['ratio'] = {'rt_list':rt_list, 
                          'res_list':res_list,
                           'z_star_arr': Z_2_vec(z_star_arr)}


    if 'contour' in params_dict:
        
        start = params_dict['contour']['start']
        stop = params_dict['contour']['stop']
        n_values = params_dict['contour']['n_values']
    
        W1, W2 = get_w1_w2_grid(start, stop, n_values)

        z_star_arr, res_l_contour = get_kl_div_grid_w1_w2(hmm_D,att_obj, utl_l1, utl_l2, Z_set, W1.reshape(-1), W2.reshape(-1), N2)
    
        kl_div_res = np.array(res_l_contour).reshape(W1.shape)
    

        res_d['contour'] = {'W1':W1, 
                            'W2':W2,
                            'kl_div':kl_div_res,
                             'z_star_arr':Z_2_vec(z_star_arr)}
     
    if 'box' in params_dict:
        
        diff_l = len(X) - (np.sum(X.reshape(-1)==np.argmax(Z_set ,axis=2), axis =1))    
        res_l = Parallel(n_jobs=num_jobs)(delayed(exp_kl_div)(hmm_D, att_obj, z, N2) for z in (Z_set))
        res_d['box']  = {'z': Z_2_vec(Z_set),
                         'diff_n_comp': diff_l,
                          'kl_div': res_l}
        

    
    res_d['info'] = {'z':Z_2_vec(Z_set),
                     't': t_,
                     'k': k_,
                      'N1': N1,
                        'N2':N2,
                         'utl1':utl_l1,
                           'utl2':utl_l2}
    
    
        
    return res_d



