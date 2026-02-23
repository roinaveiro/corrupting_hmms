import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")

def compute_q_mean(x):
    return np.mean(x)
def compute_std_plus(x):
    return np.mean(x)+np.std(x)
def compute_std_minus(x):
    return np.mean(x)-np.std(x)

def warning_call(dict_df):
    
    APS_df = dict_df['APS']
    RS_df = dict_df['RS']
    b_dict = {}
    
    str_l = ['|Q|','|X|','|T|','rho','k']
    
    for str_ in str_l:
        b_dict[str_] = (APS_df[str_].values[0] == RS_df[str_].values[0])
    
    return b_dict

def transform_df_to_plot_format(df):
    
    grouped_df = df.groupby("time")
    grouped_lists = grouped_df["q"].apply(list)
    grouped_lists = grouped_lists.reset_index()
    grouped_lists['std_minus'] = grouped_lists['q'].apply(compute_std_minus)
    grouped_lists['std_plus'] = grouped_lists['q'].apply(compute_std_plus)
    grouped_lists['mean'] = grouped_lists['q'].apply(compute_q_mean)
    grouped_lists['|Q|'] = df['|Q|'].values[0]
    grouped_lists['|X|'] = df['|X|'].values[0]
    grouped_lists['|T|'] = df['|T|'].values[0]
    grouped_lists['rho'] = df['rho'].values[0]
    grouped_lists['k'] = df['k'].values[0]
    grouped_lists['solver'] = df['solver'].values[0]
    grouped_lists['attacker'] = df['attacker'].values[0]
    return grouped_lists

def plot_quality_one_optimizer(df_1, color_params, single=False):
    df = transform_df_to_plot_format(df_1)
    plt.plot(df['time'],df['mean'], color = color_params['color_mean'],label=df['solver'].values[0])
    plt.plot(df['time'],df['std_plus'], '--', color = color_params['color_std'])
    plt.plot(df['time'],df['std_minus'], '--',color = color_params['color_std'])
    plt.fill_between(df["time"], df["mean"], df["std_plus"], color=color_params["color_sh"])
    plt.fill_between(df["time"], df["mean"], df["std_minus"], color=color_params["color_sh"])
    if single ==True:
        n_Q = df['|Q|'].values[0]
        n_X = df['|X|'].values[0]
        n_T = df['|T|'].values[0]
        rho = df['rho'].values[0]
        k = df['k'].values[0]
        attacker = (df['attacker'].values[0]).upper()
        title_str = '|Q| = '+ str(n_Q) + ',|X| = '+ str(n_X) +',|T| = '+ str(n_T) +',rho = '+ str(rho)[:4] +',k= '+ str(k)+' ('+ attacker+')'
        plt.title(title_str)
        plt.show()


def plot_quality_vs_time(dict_df, color_params_APS={"color_mean": "red", "color_std": "darkred", "color_sh": "lightsalmon"},
                              color_params_RS = {'color_mean':"blue", 'color_std':'darkblue',"color_sh":"lightblue"}):
    
    if list(set(warning_call(dict_df).values()))[0] == True:
        plot_quality_one_optimizer(dict_df['APS'], color_params_APS)
        plot_quality_one_optimizer(dict_df['RS'], color_params_RS)
        plt.ylabel('Quality')
        plt.xlabel('time (s)')
        plt.legend()
        n_Q = dict_df['APS']['|Q|'].values[0]
        n_X = dict_df['APS']['|X|'].values[0]
        n_T = dict_df['APS']['|T|'].values[0]
        rho = dict_df['APS']['rho'].values[0]
        k = dict_df['APS']['k'].values[0]
        attacker = (dict_df['APS']['attacker'].values[0]).upper()
        title_str = '|Q| = '+ str(n_Q) + ',|X| = '+ str(n_X) +',|T| = '+ str(n_T) +',rho = '+ str(rho)[:4] +',k= '+ str(k)+' ('+ attacker+')'
        plt.title(title_str)
        plt.show()
    else:
        print('WARNING:NOT COMPARING THE SAME PROBLEMS')