import numpy as np
import pandas as pd
from hmm_utils import HMM
from mc_ennumeration import MC_enumeration, MC_enumeration_parallel
from smoothing_state_attacker import ss_attacker
from smoothing_distribution_attacker import sd_attacker
from decoding_attacker import dec_attacker
from scipy.special import softmax
from collections import Counter
import time 
from joblib import Parallel, delayed
from scipy.special import rel_entr
from scipy.spatial.distance import hamming


def Z_2_vec(Z_set):
    z_str_list = []
    Z_set_vec = np.argmax(Z_set ,axis=2)
    for z in Z_set_vec:
        z_str_list.append("".join(list(map(str,z.reshape(-1)))))
    return z_str_list

def get_w1_w2_grid(start_w1, stop_w1, n_values_w1, start_w2, stop_w2, n_values_w2):
    
    w1_vals = np.linspace(start_w1, stop_w1, n_values_w1)
    w2_vals = np.linspace(start_w2, stop_w2, n_values_w2)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)
    
    return W1, W2


def utl_w_ss_attacker(X ,priors, transition, emission, rho_probs, t_, state_, c_, k_, N_, num_jobs_=-2):

    f1_att = ss_attacker(priors, transition, emission, rho_probs,X = X, 
                              w1 = 1.0, w2 = 0.0 , t = t_, state= state_, c=c_, k_value= k_)
    f2_att =  ss_attacker(priors, transition, emission, rho_probs,X = X, 
                              w1 = 0.0, w2 = 1.0 , t = t_, state= state_, c=c_, k_value= k_)
    start = time.time()
    utl_l1 = MC_enumeration_parallel(f1_att, N= N_, verbose = False, num_jobs = num_jobs_)[1]   #joblib
    end = time.time()
    print('Time to complete MC enumeration', end - start)
    utl_l2 = MC_enumeration(f2_att, N = 1, verbose = False)[1]  #joblib

    return f1_att ,utl_l1, utl_l2


def utl_w_sd_attacker(X ,priors, transition, emission, rho_probs, t_, k_, N_, num_jobs_=-2):

    f1_att = sd_attacker(priors, transition, emission, rho_probs,X = X, w1 = 1.0, w2 = 0.0 , t = t_, k_value= k_)
    f2_att =  sd_attacker(priors, transition, emission, rho_probs,X = X, w1 = 0.0, w2 = 1.0 , t = t_, k_value= k_)
    start = time.time()
    utl_l1 = MC_enumeration_parallel(f1_att, N= N_, verbose = False, num_jobs = num_jobs_)[1]   #joblib
    end = time.time()
    print('Time to complete MC enumeration', end - start)
    utl_l2 = MC_enumeration(f2_att, N = 1, verbose = False)[1]  #joblib
    
    return f1_att ,utl_l1, utl_l2


def utl_w_dec_attacker(X ,priors, transition, emission, rho_probs, seq_, k_, N_, num_jobs_=-2):

    f1_att = dec_attacker(priors, transition, emission, rho_probs,X = X, w1 = 1.0, w2 = 0.0 , seq = seq_, k_value= k_)
    f2_att =  dec_attacker(priors, transition, emission, rho_probs,X = X, w1 = 0.0, w2 = 1.0 , seq = seq_, k_value= k_)
    start  = time.time()
    utl_l1 = MC_enumeration_parallel(f1_att, N= N_, verbose = False, num_jobs = num_jobs_)[1]   #joblib
    end = time.time()
    print('Time to complete MC enumeration', end - start)
    utl_l2 = MC_enumeration(f2_att, N = 1, verbose = False)[1]  #joblib
    
    return f1_att ,utl_l1, utl_l2


def exp_tainted_data_pr(hmm_D, attacker, z_star, N_):    
    
    p_tn_data_arr = np.zeros(N_)
    for n in range(N_):
        hmm_sample = attacker.sample_hmm()
        y_vec = attacker.attack_X(rho_matrix = hmm_sample.rho, z_matrix = z_star)
        p_tn_data = hmm_D.smoothing(y_vec.astype(int), attacker.t)[attacker.state]
        p_tn_data_arr[n] = p_tn_data
        
    return [np.mean(p_tn_data_arr), np.std(p_tn_data_arr)/np.sqrt(N_)]


def exp_kl_div(hmm_D, attacker, z_star, N_):
    
    kl_div_arr = np.zeros(N_)
    for n in range(N_):
        hmm_sample = attacker.sample_hmm()
        y_vec = attacker.attack_X(rho_matrix = hmm_sample.rho, z_matrix = z_star)
        p_clean_data = hmm_D.smoothing((attacker.X).astype(int), attacker.t)
        p_tn_data = hmm_D.smoothing(y_vec.astype(int), attacker.t)
        # KL divergence
        kl_div = np.sum(rel_entr(p_clean_data, p_tn_data)) 
        kl_div_arr[n] = kl_div
    return [np.mean(kl_div_arr), np.std(kl_div_arr)/np.sqrt(N_)]


def exp_hamm_d(hmm_D, attacker, z_star, N_):
    
    hamm_d_arr = np.zeros(N_)
    for n in range(N_):
        hmm_sample = attacker.sample_hmm()
        y_vec = attacker.attack_X(rho_matrix = hmm_sample.rho, z_matrix = z_star)
        dec_tn_data = hmm_D.decode(y_vec.astype(int))[1]
        # hamming distance
        hamm_d = hamming(attacker.seq, dec_tn_data)  
        hamm_d_arr[n] = hamm_d
    return [np.mean(hamm_d_arr), np.std(hamm_d_arr)/np.sqrt(N_)]


def exp_impact(hmm_D, attacker, z_star, N_):
    
    
    if type(attacker) == ss_attacker:
        res_list =  exp_tainted_data_pr(hmm_D, attacker, z_star, N_)
    
    elif type(attacker) == sd_attacker:
        res_list = exp_kl_div(hmm_D, attacker, z_star, N_)
    
    elif type(attacker) == dec_attacker:
        res_list = exp_hamm_d(hmm_D, attacker, z_star, N_)

    return res_list

        
    
def exp_ratio_w1_w2(hmm_D, attacker, utl_l1, utl_l2, Z_set, fn_rt, init_rt, rt_st, N2, num_jobs =-2):
    
    rt_l = np.arange(init_rt,fn_rt, rt_st)
    utl_vec = rt_l.reshape(len(rt_l),1) * utl_l1 + utl_l2
    idx = np.argmax(utl_vec, axis = 1)
    z_star_arr = Z_set[idx]    
    res_array = Parallel(n_jobs=num_jobs)(delayed(exp_impact)(hmm_D, attacker, z_star, N2) for z_star in (z_star_arr))
    res_l = np.array([i[0] for i in res_array])
    res_std = np.array([i[1] for i in res_array])
    return rt_l, res_l, res_std, z_star_arr



def get_grid_w1_w2(hmm_D,attacker,utl_1, utl_2, Z_set, W1_loop, W2_loop, N_):
     
    z_star_arr = []
    res_l = []
    goal = len(W1_loop)
    count = 0
    start= time.time()
    for element in (range(len(W1_loop))):
        utl_vec = W1_loop[element] * utl_1 + W2_loop[element] * utl_2
        max_indx = np.argmax(utl_vec)
        z_star = Z_set[max_indx]
        exp_imp = exp_impact(hmm_D, attacker, z_star, N_)[0]
        z_star_arr.append(z_star)
        res_l.append(exp_imp)
        count+=1
        if count%10==0:
            end = time.time()
            print(count/goal, 'pct', end-start, 'time(s)')
            
    return z_star_arr, res_l



def all_experiments_loop(X, hmm_D, problem_dict, unc_dict, params_dict, num_jobs_=-2):
    
    trans_mat = hmm_D.transmat_
    emiss_mat = hmm_D.emissionprob_
    prior_mat = hmm_D.startprob_
    
    if problem_dict['attacker'] == 'ss' :
        
        att_obj ,utl_l1, utl_l2 = utl_w_ss_attacker(X ,prior_mat, trans_mat, emiss_mat, unc_dict['rho'], problem_dict['t'],
                                              problem_dict['state'], problem_dict['c'], unc_dict['k'], unc_dict['N1'], num_jobs_)
        
    elif problem_dict['attacker'] == 'sd':
        
        att_obj ,utl_l1, utl_l2 = utl_w_sd_attacker(X, prior_mat, trans_mat, emiss_mat, unc_dict['rho'], 
                                                    problem_dict['t'], unc_dict['k'],unc_dict['N1'], num_jobs_)
        
    elif problem_dict['attacker'] == 'dec':
        
        att_obj ,utl_l1, utl_l2 = utl_w_dec_attacker(X, prior_mat, trans_mat, emiss_mat, unc_dict['rho'], 
                                                     problem_dict['seq'], unc_dict['k'], unc_dict['N1'], num_jobs_)
            

    Z_set = att_obj.generate_attacks()
    
    res_d = {}
    
    if 'ratio' in params_dict:
        
        init_rt = params_dict['ratio']['init_rt']
        fn_rt = params_dict['ratio']['fn_rt']
        rt_st  = params_dict['ratio']['rt_st']
        
        print('Ratio computations ...')
        start = time.time()
        rt_list, res_list, res_std, z_star_arr = exp_ratio_w1_w2(hmm_D, att_obj,utl_l1, utl_l2, Z_set, fn_rt, 
                                                                        init_rt, rt_st, unc_dict['N2'],num_jobs_)
        
    
        res_d['ratio'] = {'rt_list':rt_list, 
                          'res':res_list,
                           'res_std_plus': res_list +  2* res_std,
                            'res_std_minus': res_list -2*res_std,
                           'z_star_arr': Z_2_vec(z_star_arr)}
        
        end  = time.time()
        print('Ratio plot computations completed.', 'time (s)', end - start)

    if 'contour' in params_dict:
        
        start_w1 = params_dict['contour']['start_w1']
        stop_w1 = params_dict['contour']['stop_w1']
        n_values_w1 = params_dict['contour']['n_values_w1']
        start_w2 = params_dict['contour']['start_w2']
        stop_w2 = params_dict['contour']['stop_w2']
        n_values_w2 = params_dict['contour']['n_values_w2']
            
        print('Contour plot computations ...')
        
        start = time.time()
        W1, W2 = get_w1_w2_grid(start_w1, stop_w1, n_values_w1, start_w2 , stop_w2, n_values_w2)

        z_star_array,res_l_contour = get_grid_w1_w2(hmm_D,att_obj, utl_l1, utl_l2, Z_set, 
                                                    W1.reshape(-1), W2.reshape(-1), unc_dict['N2'])
    
        res_list = np.array(res_l_contour).reshape(W1.shape)
    

        res_d['contour'] = {'W1':W1, 
                            'W2':W2,
                            'res_list':res_list, 
                             'z_star_arr':Z_2_vec(z_star_array)}
        end = time.time()
        print('Contour plot computations completed.', 'time (s)' , end - start)
        
    if 'box' in params_dict:
        if params_dict['box']==True:
            
            print('Box plot computations ...')
            start = time.time()
            diff_l = len(X) - (np.sum(X.reshape(-1)==np.argmax(Z_set ,axis=2), axis =1))    
            res_l = Parallel(n_jobs=num_jobs_)(delayed(exp_impact)(hmm_D, att_obj, z, unc_dict['N2']) for z in (Z_set))
            res_l = [i[0] for i in res_l]
            res_d['box']= {'diff_n_comp': diff_l,
                            'res':res_l,
                             'z':Z_2_vec(Z_set)}
            end = time.time()
            print('Box plot computations completed.', 'time (s)', end - start )
    
    if problem_dict['attacker'] == 'ss' :

        res_d['info'] = {'z':Z_2_vec(Z_set),
                'c':problem_dict['c'],
                 'state':problem_dict['state'],
                 't': problem_dict['t'],
                'k': unc_dict['k'] ,
                'utl1':utl_l1 ,
                'utl2': utl_l2,
                 'N1':unc_dict['N1'],
                  'N2':unc_dict['N2']}
    
    elif problem_dict['attacker'] == 'sd':

        res_d['info'] = {'z':Z_2_vec(Z_set),
                 't': problem_dict['t'],
                 'k': unc_dict['k'],
                  'N1': unc_dict['N1'],
                    'N2':unc_dict['N2'],
                     'utl1':utl_l1,
                       'utl2':utl_l2}   
        
    elif problem_dict['attacker'] == 'dec':
        
        res_d['info'] = {'z':Z_2_vec(Z_set),
                 'seq':"".join(list(map(str, problem_dict['seq'].reshape(-1)))),
                  'k': unc_dict['k'],
                  'utl1':utl_l1 ,
                  'utl2': utl_l2,
                   'N1':unc_dict['N1'],
                    'N2':unc_dict['N2']}
    


    
    return res_d


def ratio_results_to_df(res_d):
    A = pd.DataFrame(res_d['ratio'])
    A = A.rename(columns={'rt_list': 'ratio'})
    l_1 =[[str(i)] for i in list(A['z_star_arr'])]
    A['z_star_arr'] = l_1
    return A

def contour_results_to_df(res_d):
    d = {}
    d['w1_values'] = list(res_d['contour']['W1'].reshape(-1))
    d['w2_values'] = list(res_d['contour']['W2'].reshape(-1))
    d['res'] = list(res_d['contour']['res_list'].reshape(-1))
    l_1 =[[str(i)] for i in res_d['contour']['z_star_arr']]
    d['z_star_arr'] = l_1

    return pd.DataFrame(d)


def box_results_to_df(res_d):
    
    l_1 =[[str(i)] for i in res_d['box']['z']]
    df = pd.DataFrame(res_d['box'])
    df['z'] = l_1
    
    return df


def info_results_to_df(res_d):
    
    l_1 =[[str(i)] for i in res_d['info']['z']]
    res_d['info']['z'] = l_1
    return pd.DataFrame(res_d['info'])
    





