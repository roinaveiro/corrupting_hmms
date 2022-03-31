import pandas as pd
import numpy as np


def ratio_results_to_df_P11(res_d):
    A = pd.DataFrame(res_d['ratio'])
    A = A.rename(columns={'rt_list': 'ratio', 
                        'res_list': 'prob_tainted_data',
                         'p_clean_data':'prob_clean_data'})
    
    l_1 =[[str(i)] for i in list(A['z_star_arr'])]
    A['z_star_arr'] = l_1
    
    return A

def ratio_df_to_results_P11(df):
    in_d = {}
    #df.columns = ['rt_list','res_list','p_clean_data']
    in_d['rt_list'] = list(df['ratio'])
    in_d['res_list'] = list(df['prob_tainted_data'])
    in_d['p_clean_data'] = list(df['prob_clean_data'])[0]
    
    return in_d

def contour_results_to_df_P11(res_d):
    d = {}
    d['w1_values'] = list(res_d['contour']['W1'].reshape(-1))
    d['w2_values'] = list(res_d['contour']['W2'].reshape(-1))
    d['prob_tainted_data'] = list(res_d['contour']['p_tainted_data'].reshape(-1))
    d['prob_clean_data']  = res_d['contour']['p_clean_data']
    return pd.DataFrame(d)

def contour_df_to_results_P11(df):
    in_d = {}
    dim = int(np.sqrt(len(df)))
    W1 = np.array(df['w1_values']).reshape(dim,dim)
    W2 = np.array(df['w2_values']).reshape(dim,dim)
    p_tainted_data = np.array(df['prob_tainted_data']).reshape(dim,dim)
    p_clean_data = list(df['prob_clean_data'])[0]
    in_d['W1'] = W1
    in_d['W2'] = W2
    in_d['p_tainted_data']  = p_tainted_data 
    in_d['p_clean_data'] = p_clean_data
    return in_d

def n_comp_to_box_df_P11(res_d):
    
    l_1 =[[str(i)] for i in res_d['box']['z']]
    res_d['box']['z'] = l_1
    
    return pd.DataFrame(res_d['box'])

def box_to_n_comp_P11(df):
        
    res_d = {}
    
    res_d['diff_n_comp'] = list(df['diff_n_comp'])
    res_d['prob_tainted_data'] = list(df['prob_tainted_data'])     
    
    return res_d

def results_to_info_df_P11(res_d):
    
    l_1 =[[str(i)] for i in res_d['info']['z']]
    res_d['info']['z'] = l_1
    return pd.DataFrame(res_d['info'])



def df_to_viz_d(df_d):
    
    res_d = {}
    
    if 'ratio' in df_d:
        res_d['ratio'] = ratio_df_to_results_P11(df_d['ratio'])
    if ('contour') and ('info') in df_d:
        res_d['contour'] = contour_df_to_results_P11(df_d['contour'])
        res_d['contour']['c'] = list(df_d['info']['c'])[0]
    if ('contour') in df_d:
        res_d['box'] = box_to_n_comp_P11(df_d['box'])
        
    res_d['c'] = list(df_d['info']['c'])[0]
    res_d['t'] = list(df_d['info']['t'])[0]
    res_d['state'] = list(df_d['info']['state'])[0]
    
    return res_d





