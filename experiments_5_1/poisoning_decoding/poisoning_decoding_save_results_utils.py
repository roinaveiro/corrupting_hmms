import pandas as pd
import numpy as np


def ratio_results_to_df_P2(res_d):
    A = pd.DataFrame(res_d['ratio'])
    A = A.rename(columns={'rt_list': 'ratio', 
                        'res_list': 'jac_d'})
    l_1 =[[str(i)] for i in list(A['z_star_arr'])]
    A['z_star_arr'] = l_1
    return A

def ratio_df_to_results_P2(df):
    in_d = {}
    in_d['rt_list'] = list(df['ratio'])
    in_d['res_list'] = list(df['jac_d'])
    return in_d

def contour_results_to_df_P2(res_d):
    d = {}
    d['w1_values'] = list(res_d['contour']['W1'].reshape(-1))
    d['w2_values'] = list(res_d['contour']['W2'].reshape(-1))
    d['jac_d'] = list(res_d['contour']['jac_d'].reshape(-1))
    l_1 =[[str(i)] for i in res_d['contour']['z_star_arr']]
    d['z_star_arr'] = l_1

    return pd.DataFrame(d)

def contour_df_to_results_P2(df):
    in_d = {}
    out_d = {}
    dim = int(np.sqrt(len(df)))
    W1 = np.array(df['w1_values']).reshape(dim,dim)
    W2 = np.array(df['w2_values']).reshape(dim,dim)
    jac_d = np.array(df['jac_d']).reshape(dim,dim)
    in_d['W1'] = W1
    in_d['W2'] = W2
    in_d['jac_d']  = jac_d
    return in_d

def n_comp_to_results_P2(res_d):
    
    l_1 =[[str(i)] for i in res_d['box']['z']]
    df = pd.DataFrame(res_d['box'])
    df['z'] = l_1
    return df

def box_df_to_n_comp_P2(df):
        
    res_d = {}

    res_d['diff_n_comp'] = list(df['diff_n_comp'])
    res_d['jac_d'] = list(df['jac_d']) 
    
    return res_d

def results_to_info_df_P2(res_d):
    l_1 =[[str(i)] for i in res_d['info']['z']]
    res_d['info']['z'] = l_1
    return pd.DataFrame(res_d['info'])
    

def df_to_viz_d(df_d):
    
    res_d = {}
    
    if 'ratio' in df_d:
        res_d['ratio'] = ratio_df_to_results_P2(df_d['ratio'])
    if 'contour' in df_d:
        res_d['contour'] = contour_df_to_results_P2(df_d['contour'])
    if 'box' in df_d:
        res_d['box'] = box_df_to_n_comp_P2(df_d['box'])
    
    
    return res_d






