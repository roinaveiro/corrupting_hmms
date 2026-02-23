import os
import numpy as np
import pandas as pd
from collections import Counter
from viz_func_APS_vs_RS_3 import * 

def create_dict_DF_APS_selected_v2(exp, att_str, hyp_comb_list, comb_num):
    dict_DF = OrderedDict()
    exp_dir = 'experiment_'+str(exp)+'_scaled_csv'
    for hyp_c in hyp_comb_list:
        for f in os.listdir(exp_dir + '/' +'APS'+ '/'+ att_str+'/' + hyp_c):
            if '.json' in f:
                if 'comb_'+str(comb_num) in f:
                    if hyp_c == 'cs_1':
                        dict_DF['APS_A'] = pd.read_json(exp_dir + '/' +'APS' + '/'+ att_str+'/' + hyp_c +'/'+f ,orient = 'split')
                    elif hyp_c == 'cs_500':
                        dict_DF['APS_B'] = pd.read_json(exp_dir + '/' +'APS' + '/'+ att_str+'/' + hyp_c +'/'+f ,orient = 'split')
    return dict_DF
    
def create_dict_DF_RS_SA_selected_v2(exp,  solver, att_str, hyp_comb_list, comb_num):
    
    dict_DF = OrderedDict()
    exp_dir = 'experiment_'+str(exp)+'_scaled_csv/'
    for hyp_c in hyp_comb_list:
        for f in os.listdir(exp_dir + '/' +solver + '/'+ att_str+'/' + hyp_c):
            if '.json' in f:
                if 'comb_'+str(comb_num) in f:
                    dict_DF[solver+'_'+hyp_c] = pd.read_json(exp_dir + '/' +solver + '/'+ att_str+'/' + hyp_c +'/'+f ,orient = 'split')

    return dict_DF

def create_dict_RME_selected_v2(exp, att_str, comb_num):
    
    dict_DF = OrderedDict()
    exp_dir = 'experiment_'+str(exp)+'_scaled_csv'
    for f in os.listdir(exp_dir + '/' +'RME'+ '/'+ att_str+'/'):
        if '.json' in f:
            if 'comb_'+str(comb_num) in f:
                dict_DF['RME'] = pd.read_json(exp_dir + '/' +'RME' + '/'+ att_str +'/'+f ,orient = 'split')
            
    return dict_DF

def create_dict_DF_solver_v2(exp_, solver_, att_str_, hyp_comb_list_ ,comb_num_):
    
    if solver_ == 'APS':
        dict_DF = create_dict_DF_APS_selected_v2(exp = exp_, 
                                              att_str = att_str_, 
                                              hyp_comb_list = hyp_comb_list_ ,
                                              comb_num = comb_num_)
    elif solver_ == 'MCTS':
        dict_DF = create_dict_DF_RS_SA_selected_v2(exp = exp_, 
                                                solver = solver_, 
                                                att_str = att_str_,
                                                hyp_comb_list = hyp_comb_list_,
                                                comb_num = comb_num_)
    elif solver_ == 'SA':
        dict_DF = create_dict_DF_RS_SA_selected_v2(exp = exp_, 
                                                solver = solver_, 
                                                att_str = att_str_,
                                                hyp_comb_list = hyp_comb_list_,
                                                comb_num = comb_num_)
    elif solver_== 'RME':
        dict_DF = create_dict_RME_selected_v2(exp = exp_, 
                                           att_str = att_str_, 
                                           comb_num = comb_num_)
        
        
    return dict_DF





def difference_with_max(res_df, setup_d):
    
    res_df = res_df[res_df['time']==1200].reset_index()
    z_attack = res_df.iloc[res_df['scaled_eu'].idxmax()]['z']
    obs_chain = list(setup_d['X'].reshape(-1))
    return Counter(np.array(obs_chain) == np.array(z_attack))[False]

def difference_with_mean(res_df, setup_d):
    
    diff_num = []
    res_df = res_df[res_df['time']==1200].reset_index()
    z_list = list(res_df['z'])
    obs_chain = list(setup_d['X'].reshape(-1))
    for z in z_list:
        diff_num.append(Counter(np.array(obs_chain) == np.array(z))[False])
    
    return np.mean(diff_num)

def compute_difference_btw_z_star_and_obs(s_d, exp, solver, att_str, hyp_l, comb_n, flag = 'max'):
    
    l_res = []
    dDF = create_dict_DF_solver_v2(exp_ = exp, 
                                solver_ = solver, 
                                att_str_ = att_str, 
                                hyp_comb_list_ = hyp_l ,
                                comb_num_= comb_n)
    
    for k in dDF:
        if flag == 'max':
            diff = difference_with_max(res_df = dDF[k], setup_d = s_d)
        elif flag == 'mean':
            diff = difference_with_mean(res_df = dDF[k], setup_d = s_d)
        
        rd = {'exp':exp,
          'att': att_str,
          'solver':solver,
          'hyp': k,
         'comb': comb_n,
          'flag': flag,
             'diff':diff}
        
        l_res.append(rd)
        
        
    return l_res

def compute_df_btw_z_star_and_obs_loop_exp_2(s_d, solver_l, att_l, hyp_l_d, comb_n_l, flag = 'max'):
    list_r  = []
    for s in solver_l:
        for a in att_l:
            for c in comb_n_l:
              
                aux_l = compute_difference_btw_z_star_and_obs(s_d = s_d, 
                                                      exp = 2,
                                                    solver = s,
                                                    att_str = a,
                                                     hyp_l = hyp_l_d[s],
                                                     comb_n = c,
                                                     flag = flag)
                list_r += aux_l

    return pd.DataFrame(list_r)


def compute_df_btw_z_star_and_obs_loop_exp_3(s_d_d, solver_l, att_l, hyp_l_d, comb_n_l, flag = 'max'):
    
    
    list_r  = []
    for s in solver_l:
        for a in att_l:
            for c in comb_n_l:
                aux_l = compute_difference_btw_z_star_and_obs(s_d = s_d_d[c], 
                                                      exp = 3,
                                                    solver = s,
                                                    att_str = a,
                                                     hyp_l = hyp_l_d[s],
                                                     comb_n = c,
                                                     flag = flag)
                list_r += aux_l

    return pd.DataFrame(list_r)



