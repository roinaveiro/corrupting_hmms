from solver_attacker_time_experiment_utils_v2 import *
from params_experiments_2 import params_dict as params2
from params_experiments_3 import params_dict as params3
from tqdm import tqdm_notebook as tqdm
from scipy.special import rel_entr
from scipy.spatial.distance import hamming


def generate_z_matrix_from_attack(z, attacker):
    
    z_matrix = np.zeros((attacker.T,attacker.n_obs))
    
    for i in range(z_matrix.shape[0]):
        z_matrix[i,z[i]] = 1
        
    return z_matrix

def exp_tainted_data_computation(hmm_D, attacker, z_star, N_):    
    
    p_tn_data_arr = np.zeros(N_)
    for n in range(N_):
        hmm_sample = attacker.sample_hmm()
        y_vec = attacker.attack_X(rho_matrix = hmm_sample.rho, z_matrix = z_star)
        p_tn_data = hmm_D.smoothing(y_vec.astype(int), attacker.t)[attacker.state]
        p_tn_data_arr[n] = p_tn_data
        
    return p_tn_data_arr

def exp_kl_div_computation(hmm_D, attacker, z_star, N_):
    
    kl_div_arr = np.zeros(N_)
    for n in range(N_):
        hmm_sample = attacker.sample_hmm()
        y_vec = attacker.attack_X(rho_matrix = hmm_sample.rho, z_matrix = z_star)
        p_clean_data = hmm_D.smoothing((attacker.X).astype(int), attacker.t)
        p_tn_data = hmm_D.smoothing(y_vec.astype(int), attacker.t)
        # KL divergence
        kl_div = np.sum(rel_entr(p_clean_data, p_tn_data)) 
        kl_div_arr[n] = kl_div
    return kl_div_arr 


def exp_hamm_d_computation(hmm_D, attacker, z_star, N_):
    
    hamm_d_arr = np.zeros(N_)
    for n in range(N_):
        hmm_sample = attacker.sample_hmm()
        y_vec = attacker.attack_X(rho_matrix = hmm_sample.rho, z_matrix = z_star)
        dec_tn_data = hmm_D.decode(y_vec.astype(int))[1]
        # hamming distance
        hamm_d = hamming(attacker.seq, dec_tn_data)  
        hamm_d_arr[n] = hamm_d
    return hamm_d_arr

def exp_impact_computation(hmm_D, attacker, z_star, N_):
    
    
    if type(attacker) == ss_attacker:
        res_list =  exp_tainted_data_computation(hmm_D, attacker, z_star, N_)
    
    elif type(attacker) == sd_attacker:
        res_list = exp_kl_div_computation(hmm_D, attacker, z_star, N_)
    
    elif type(attacker) == dec_attacker:
        res_list = exp_hamm_d_computation(hmm_D, attacker, z_star, N_)

    return res_list

def impact_enc_att_l(hmm_D, attacker, z_enc_list, n_iter = 100):
    exp_comp_l = []
    for z_enc in (z_enc_list):
        z_matrix = generate_z_matrix_from_attack(z_enc, attacker)
        res_l = exp_impact_computation(hmm_D, attacker, z_matrix, n_iter)
        exp_comp_l.append(res_l)
    all_res_l = np.concatenate(exp_comp_l)
    return [np.mean(all_res_l), np.std(all_res_l)/np.sqrt(len(all_res_l)), np.std(all_res_l)]



def obtain_impact_info_exp2(res_df, att_st, num, parameter_dict = params2 ,num_iters = 100):
    
    ##extracting results
    z_enconded_list = list(res_df[res_df['time']==1200]['z'])

    
    mat = {}
    mat['trans'] = parameter_dict['trans']
    mat['emiss'] = parameter_dict['emiss']
    mat['prior'] = parameter_dict['prior']
    mat['X'] = parameter_dict['X']
    par_d  = parameter_dict[att_st]
    
    ### defining hmm
    hmm = HMM(n_components=mat['trans'].shape[0], n_obs=mat['emiss'].shape[1])
    hmm.emissionprob_ = mat['emiss']
    hmm.transmat_     = mat['trans']
    hmm.startprob_    = mat['prior']
    
    ### defining attacker
    if att_st == 'atr':
        att = ss_attacker(mat['prior'],            #THING TO CHANGE
                         mat['trans'], 
                         mat['emiss'], 
                         par_d[num]['rho'],
                         mat['X'], 
                         w1 = par_d['w1'],
                         w2 = par_d['w2'] , 
                         t = par_d['t'],
                         state = par_d['state'],
                         c = par_d['c'],
                         k_value= par_d[num]['k'])
        
        unt_pr = (hmm.smoothing(mat['X'].astype(int), att.t)[att.state])
        tai_pr = (impact_enc_att_l(hmm, att, z_enconded_list, n_iter = num_iters))[0]
        infod = {'untainted data':str(round(unt_pr,4)),
         'tainted data': str(round(tai_pr,4)), 
         'att_str': att_st, 
         'solver': res_df['solver'][0],
         'comb_num': num, 
         'w1':par_d['w1'], 
         'w2': par_d['w2'], 
         't': par_d['t'], 
         'state': par_d['state'], 
         'k':par_d[num]['k'],
          'c':par_d['c']}

        
    elif att_st =='rep':
        att = ss_attacker(mat['prior'],            #THING TO CHANGE
                         mat['trans'], 
                         mat['emiss'], 
                         par_d[num]['rho'],
                         mat['X'], 
                         w1 = par_d['w1'],
                         w2 = par_d['w2'] , 
                         t = par_d['t'],
                         state = par_d['state'],
                         c = par_d['c'],
                         k_value= par_d[num]['k'])
        
        
        unt_pr = (hmm.smoothing(mat['X'].astype(int), att.t)[att.state])
        tai_pr = (impact_enc_att_l(hmm, att, z_enconded_list, n_iter = num_iters))[0]
        infod = {'untainted data':str(round(unt_pr,4)),
         'tainted data': str(round(tai_pr,4)), 
         'att_str': att_st, 
         'solver': res_df['solver'][0],
         'comb_num': num, 
         'w1':par_d['w1'], 
         'w2': par_d['w2'], 
         't': par_d['t'], 
         'state': par_d['state'], 
         'k':par_d[num]['k'],
          'c':par_d['c']}
        
    elif att_st == 'dd':
        att = sd_attacker(mat['prior'],            #THING TO CHANGE
                         mat['trans'], 
                         mat['emiss'], 
                         par_d[num]['rho'],
                         mat['X'], 
                         w1 = par_d['w1'],
                         w2 = par_d['w2'] , 
                         t = par_d['t'], 
                         k_value= par_d[num]['k'])
        
        
        kl_div = impact_enc_att_l(hmm, att, z_enconded_list, n_iter = num_iters)[0]
        infod  = { 'kl_div': kl_div,
         'att_str': att_st, 
         'solver': res_df['solver'][0],
         'comb_num': num, 
         'w1':par_d['w1'], 
         'w2': par_d['w2'], 
         't': par_d['t'],  
         'k':par_d[num]['k']}
        
    elif att_st == 'pd':
        att = dec_attacker(mat['prior'],            #THING TO CHANGE
                         mat['trans'], 
                         mat['emiss'], 
                         par_d[num]['rho'],
                         mat['X'], 
                         w1 = par_d['w1'],
                         w2 = par_d['w2'] , 
                          seq = par_d['seq'], 
                         k_value= par_d[num]['k'])
        
        nhd = impact_enc_att_l(hmm, att, z_enconded_list, n_iter = num_iters)[0]
        infod  = { 'nhd': nhd,
         'att_str': att_st, 
         'solver': res_df['solver'][0],
         'comb_num': num, 
         'w1':par_d['w1'], 
         'w2': par_d['w2'], 
         'seq': par_d['seq'], 
         'k':par_d[num]['k']}
        
    return infod
        



def obtain_impact_info_exp3(res_df, att_st, num, parameter_dict= params3 ,num_iters = 100):
    
    ##extracting results
    z_enconded_list = list(res_df[res_df['time']==1200]['z'])

    
    par_d  = parameter_dict[att_st]
    mat = parameter_dict['mat'][num]
    
    ### defining hmm
    hmm = HMM(n_components=mat['trans'].shape[0], n_obs=mat['emiss'].shape[1])
    hmm.emissionprob_ = mat['emiss']
    hmm.transmat_     = mat['trans']
    hmm.startprob_    = mat['prior']
    
    ### defining attacker
    if att_st == 'atr':
        att = ss_attacker(mat['prior'],            #THING TO CHANGE
                         mat['trans'], 
                         mat['emiss'], 
                         mat['rho'],
                         mat['X'], 
                         w1 = par_d['w1'],
                         w2 = par_d['w2'] , 
                         t = par_d[num]['t'],
                         state = par_d[num]['state'],
                         c = par_d['c'],
                         k_value= parameter_dict['k'])
        
        unt_pr = (hmm.smoothing(mat['X'].astype(int), att.t)[att.state])
        tai_pr = (impact_enc_att_l(hmm, att, z_enconded_list, n_iter = num_iters))[0]
        infod = {'untainted data':str(round(unt_pr,4)),
         'tainted data': str(round(tai_pr,4)), 
         'att_str': att_st, 
         'solver': res_df['solver'][0],
         'comb_num': num, 
         'w1':par_d['w1'], 
         'w2': par_d['w2'], 
         't': par_d[num]['t'], 
         'state': par_d[num]['state'], 
         'k':parameter_dict['k'],
          'c':par_d['c']}

        
    elif att_st =='rep':
        att = ss_attacker(mat['prior'],            #THING TO CHANGE
                         mat['trans'], 
                         mat['emiss'], 
                         mat['rho'],
                         mat['X'], 
                         w1 = par_d['w1'],
                         w2 = par_d['w2'] , 
                         t = par_d[num]['t'],
                         state = par_d[num]['state'],
                         c = par_d['c'],
                         k_value= parameter_dict['k'])
        
        unt_pr = (hmm.smoothing(mat['X'].astype(int), att.t)[att.state])
        tai_pr = (impact_enc_att_l(hmm, att, z_enconded_list, n_iter = num_iters))[0]
        infod = {'untainted data':str(round(unt_pr,4)),
         'tainted data': str(round(tai_pr,4)), 
         'att_str': att_st, 
         'solver': res_df['solver'][0],
         'comb_num': num, 
         'w1':par_d['w1'], 
         'w2': par_d['w2'], 
         't': par_d[num]['t'], 
         'state': par_d[num]['state'], 
         'k':parameter_dict['k'],
          'c':par_d['c']}
        
    elif att_st == 'dd':
        att = sd_attacker(mat['prior'],            #THING TO CHANGE
                         mat['trans'], 
                         mat['emiss'], 
                         mat['rho'],
                         mat['X'], 
                         w1 = par_d['w1'],
                         w2 = par_d['w2'] , 
                         t = par_d[num]['t'], 
                         k_value= parameter_dict['k'])
        
        kl_div = impact_enc_att_l(hmm, att, z_enconded_list, n_iter = num_iters)[0]
        infod  = { 'kl_div': kl_div,
         'att_str': att_st, 
         'solver': res_df['solver'][0],
         'comb_num': num, 
         'w1':par_d['w1'], 
         'w2': par_d['w2'], 
         't': par_d[num]['t'],  
         'k':parameter_dict['k']}
        
        
        
    elif att_st == 'pd':
        att = dec_attacker(mat['prior'],            #THING TO CHANGE
                         mat['trans'], 
                         mat['emiss'], 
                         mat['rho'],
                         mat['X'], 
                         w1 = par_d['w1'],
                         w2 = par_d['w2'] , 
                          seq = par_d[num]['seq'], 
                         k_value= parameter_dict['k'])
        
        nhd = impact_enc_att_l(hmm, att, z_enconded_list, n_iter = num_iters)[0]
        infod  = { 'nhd': nhd,
         'att_str': att_st, 
         'solver': res_df['solver'][0],
         'comb_num': num, 
         'w1':par_d['w1'], 
         'w2': par_d['w2'], 
         'seq': par_d[num]['seq'], 
         'k':parameter_dict['k']}
     
    return infod