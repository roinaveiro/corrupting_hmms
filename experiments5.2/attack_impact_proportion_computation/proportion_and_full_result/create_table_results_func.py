import os
import numpy as np
import pandas as pd
import json
from collections import Counter


with open('dict_hyp_13_09_22.json', 'r') as fp:
    hyp_D = json.load(fp)
with open('dict_T.json', 'r') as fp:
    dict_T = json.load(fp)

def load_imp_df(exp, attacker, comb_num):
    if attacker == 'attr':
        aux = 'atr'
    elif attacker in ['rep', 'dd', 'pd']:
        aux = attacker
        
    imp_df = pd.read_csv('attack_impact_results/experiment_'+str(exp)+'/experiment_'+str(exp)
                         +'_attacker_'+str(aux)+'_comb_'+str(comb_num)+'_.csv')
    return imp_df

def load_comp_df(exp):
    comp_df = pd.read_csv('n_components_csv/exp_'+str(exp)+'_diff_mean.csv')
    return comp_df
def choose_hyp(exp, attacker, comb_num, solver, hyp_D = hyp_D):
    hyp = hyp_D[exp][attacker][comb_num][solver]
    return hyp


def compute_mean_diff(exp, attacker, comb_num, solver, dict_T_len = dict_T, hypD = hyp_D ,n_decimals = 3):
    
    comp_df = load_comp_df(exp)
    if 'APS' in solver:
        hyp = solver
        solver = 'APS'
    elif solver =='RME':
        hyp = 'RME'
    else:
        hyp = solver +'_' + choose_hyp(exp, attacker, comb_num, solver, hyp_D = hypD)        
    
    mean_diff = comp_df[(comp_df['exp']==int(exp))&(comp_df['att']==attacker)&(comp_df['comb']==int(comb_num))
        &(comp_df['solver']==solver)&(comp_df['hyp']==hyp)]['diff'].values[0]
    pr_diff = round(mean_diff/dict_T_len[exp][comb_num],n_decimals)

    return format(pr_diff,"."+str(n_decimals)+"f")



def compute_metric(exp, attacker, comb_num, solver,  n_decimals = 3):
    
    imp_df = load_imp_df(exp, attacker, comb_num)
    if 'APS' in solver:
        col_name = 'solver_1'
    else:
        col_name = 'solver'
    if attacker == 'attr':
        unt_d = imp_df[imp_df[col_name]==solver]['untainted data'].values[0]
        tai_d = imp_df[imp_df[col_name]==solver]['tainted data'].values[0]
        metric = tai_d - unt_d
    elif attacker == 'rep':
        unt_d = imp_df[imp_df[col_name]==solver]['untainted data'].values[0]
        tai_d = imp_df[imp_df[col_name]==solver]['tainted data'].values[0]
        metric = unt_d - tai_d
    elif attacker == 'dd':
        metric = imp_df[imp_df[col_name]==solver]['kl_div'].values[0]
    elif attacker == 'pd':
        metric = imp_df[imp_df[col_name]==solver]['nhd'].values[0]
    
    return format(round(metric,n_decimals),"."+str(n_decimals)+"f")

def create_single_comb_result(exp, attacker, comb_num, solver, dTl = dict_T, hypD= hyp_D):
    
    pr_diff = compute_mean_diff(exp, attacker, comb_num, solver, dict_T_len = dTl)
    metric = compute_metric(exp,attacker,comb_num, solver)
    res_comb_d = { 'metric_'+str(comb_num):metric, 'pr_diff_'+str(comb_num):pr_diff}
    return res_comb_d


def create_row_results(exp, attacker, solver, dTl = dict_T, hypD= hyp_D):

    d = {}
    if solver =='MCTS':
        d['solver'] = 'MCTS'
    else:
        d['solver'] = solver

    for c in ['1','2','3','4']:
        d.update(create_single_comb_result(exp, attacker, c, solver, dTl, hyp_D))
    
    return pd.DataFrame([d])

def create_full_result(exp, attacker, dTl = dict_T, hypD= hyp_D):
    lr = []
    for s in ['APS_A', 'APS_B', 'MCTS', 'SA','RME']:
        df = create_row_results(exp, attacker, s, dTl, hypD)
        lr.append(df)
    return pd.concat(lr).reset_index(drop=True)
