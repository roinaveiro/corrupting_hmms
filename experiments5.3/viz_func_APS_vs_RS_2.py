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

def warning_call_v1(dict_df):
    str_l = ['|Q|','|X|','|T|','rho','k']
    b_dict = {}
    for str_ in str_l:
        aux_l = []
        for key in dict_df:
            aux_l.append(dict_df[key][str_][0])
        b_dict[str_] = len(set(aux_l))
    return b_dict

def remove_scale_smoothing_state_attacker(utility, len_T, w1, w2):
    
    scale = w1 + w2 * len_T + 1
    
    return utility - scale

def remove_scale_smoothing_distribution_attacker(utility, len_T ,w1 ,w2):
    
    scale = w2 * len_T + 1
    
    return utility - scale

def remove_scale_decoding_attacker(utility, len_T, w1, w2):
    
    scale = w1* len_T + w2* len_T +1
    
    return utility - scale

def scale_files(exp_folder_l, w1_w2_dict):
    
    for exp_f in (exp_folder_l):
        for algo in os.listdir(exp_f + '/'):
            for att in os.listdir(exp_f + '/' + algo):
                if algo == 'RME':
                    for file in os.listdir(exp_f + '/' + algo+'/'+att):
                        df = pd.read_json(exp_f + '/' + algo + '/'+ att + '/'+ file, orient = 'split')
                        file_dir = exp_f + '/' + algo + '/'+ att + '/'+ file
                        if att == 'attr':
                            w1 = w1_w2_dict['attr']
                            df['scaled_eu'] = list(df.apply(lambda x: remove_scale_smoothing_state_attacker(df.q, df['|T|'], w1, 1), axis = 1)[0:1].values[0])
                        elif att =='rep':
                            w1 = w1_w2_dict['rep']
                            df['scaled_eu'] = list(df.apply(lambda x: remove_scale_smoothing_state_attacker(df.q, df['|T|'], w1, 1), axis = 1)[0:1].values[0])
                        elif att =='dd':
                            w1 = w1_w2_dict['dd']
                            df['scaled_eu'] = list(df.apply(lambda x: remove_scale_smoothing_distribution_attacker(df.q, df['|T|'], w1, 1), axis = 1)[0:1].values[0])
                        elif att =='pd':
                            w1 = w1_w2_dict['pd']
                            df['scaled_eu'] = list(df.apply(lambda x: remove_scale_decoding_attacker(df.q, df['|T|'], w1, 1), axis = 1)[0:1].values[0])
                        df.to_json(file_dir, orient="split")
                else:
                    for hyp_comb in os.listdir(exp_f + '/' + algo+'/'+att):
                        for file in os.listdir(exp_f + '/' + algo+'/'+att + '/' + hyp_comb):
                            df = pd.read_json(exp_f + '/' + algo + '/'+ att + '/'+ hyp_comb+'/'+ file, orient = 'split')
                            file_dir = exp_f + '/' + algo + '/'+ att + '/'+ hyp_comb+'/'+ file
                            if att == 'attr':
                                w1 = w1_w2_dict['attr']
                                df['scaled_eu'] = list(df.apply(lambda x: remove_scale_smoothing_state_attacker(df.q, df['|T|'], w1, 1), axis = 1)[0:1].values[0])
                            elif att =='rep':
                                w1 = w1_w2_dict['rep']
                                df['scaled_eu'] = list(df.apply(lambda x: remove_scale_smoothing_state_attacker(df.q, df['|T|'], w1, 1), axis = 1)[0:1].values[0])
                            elif att =='dd':
                                w1 = w1_w2_dict['dd']
                                df['scaled_eu'] = list(df.apply(lambda x: remove_scale_smoothing_distribution_attacker(df.q, df['|T|'], w1, 1), axis = 1)[0:1].values[0])
                            elif att =='pd':
                                w1 = w1_w2_dict['pd']
                                df['scaled_eu'] = list(df.apply(lambda x: remove_scale_decoding_attacker(df.q, df['|T|'], w1, 1), axis = 1)[0:1].values[0])
                    
                            df.to_json(file_dir, orient="split")


                        



def transform_df_to_plot_format(df):
    
    grouped_df = df.groupby("time")
    grouped_lists = grouped_df["scaled_eu"].apply(list)
    grouped_lists = grouped_lists.reset_index()
    grouped_lists['std_minus'] = grouped_lists['scaled_eu'].apply(compute_std_minus)
    grouped_lists['std_plus'] = grouped_lists['scaled_eu'].apply(compute_std_plus)
    grouped_lists['mean'] = grouped_lists['scaled_eu'].apply(compute_q_mean)
    grouped_lists['|Q|'] = df['|Q|'].values[0]
    grouped_lists['|X|'] = df['|X|'].values[0]
    grouped_lists['|T|'] = df['|T|'].values[0]
    grouped_lists['rho'] = df['rho'].values[0]
    grouped_lists['k'] = df['k'].values[0]
    grouped_lists['solver'] = df['solver'].values[0]
    grouped_lists['attacker'] = df['attacker'].values[0]
    return grouped_lists

def plot_quality_one_optimizer_v1(df_1, color_params,solver_label = '', single=False,grid = False, mean_flag = False,y_axis_boolean=False, y_axis_t= '',size  = (8,5.5)):
    df = transform_df_to_plot_format(df_1)
    #print('Mean last time',df.iloc[-1]['mean'])
    plt.plot(df['time'],df['mean'], color = color_params['color_mean'],label= solver_label)
    if mean_flag == False:
        plt.plot(df['time'],df['std_plus'], '--', color = color_params['color_std'])
        plt.plot(df['time'],df['std_minus'], '--',color = color_params['color_std'])
        plt.fill_between(df["time"], df["mean"], df["std_plus"], color=color_params["color_sh"])
        plt.fill_between(df["time"], df["mean"], df["std_minus"], color=color_params["color_sh"])
    if single ==True:
        plt.rcParams["figure.figsize"] = size            
        n_Q = df['|Q|'].values[0]
        n_X = df['|X|'].values[0]
        n_T = df['|T|'].values[0]
        rho = df['rho'].values[0]
        k = df['k'].values[0]
        attacker = (df['attacker'].values[0]).upper()
        title_str = ' |Q| = '+ str(n_Q) + ',|X| = '+ str(n_X) +',|T| = '+ str(n_T) +',rho = '+ str(rho)[:4] +',k= '+ str(k)+' ('+ attacker+')'
        plt.ylabel('Quality')
        plt.xlabel('time (s)')
        plt.legend(loc = 4)
        if y_axis_boolean == True:
            plt.yticks(y_axis_t)
        plt.title(title_str)
        plt.show()
    if grid ==True:
        n_Q = df['|Q|'].values[0]
        n_X = df['|X|'].values[0]
        n_T = df['|T|'].values[0]
        rho = df['rho'].values[0]
        k = df['k'].values[0]
        attacker = (df['attacker'].values[0]).upper()
        plt.ylabel('Expected utility')
        plt.xlabel('time (s)')
        plt.legend(loc = 4)
        if y_axis_boolean == True:
            plt.yticks(y_axis_t)

def choose_best_combination(dict_DF):
    max_dict = {}
    for k in dict_DF:
        max_dict[k] = max(transform_df_to_plot_format(dict_DF[k])['mean'])
    max_key = max(max_dict, key=max_dict.get)
    return max_key

def obtain_max_value(dict_DF):
    max_dict = {}
    for k in dict_DF:
        max_dict[k] = max(transform_df_to_plot_format(dict_DF[k])['mean'])
    max_value = max(max_dict.values())
    return max_value
    
def obtain_min_value(dict_DF):
    min_dict = {}
    for k in dict_DF:
        min_dict[k] = min(transform_df_to_plot_format(dict_DF[k])['mean'])
    min_value = min(min_dict.values())
    return min_value
            
            

def plot_quality_vs_time_1(dict_df, dict_color, y_axis_boolean=False, y_axis_t= '',size  = (8,5.5), only_mean = False,title_size = 24, save_boolean = False, save_string = '', dir_string = '', multigrid = False):
    if len(set(warning_call_v1(dict_df).values())) == 1:
        plt.rcParams["figure.figsize"] = size            
        for solver in dict_df:
            #print('-------------------',solver, '-----------------------------')
            plot_quality_one_optimizer_v1(dict_df[solver], dict_color[solver], solver, single = False, mean_flag = only_mean)
            key = solver
        plt.ylabel('Expected utility')
        plt.xlabel('time (s)')
        plt.legend(bbox_to_anchor=(1.04,1))
        n_Q = dict_df[solver]['|Q|'].values[0]
        n_X = dict_df[solver]['|X|'].values[0]
        n_T = dict_df[solver]['|T|'].values[0]
        rho = dict_df[solver]['rho'].values[0]
        k = dict_df[solver]['k'].values[0]
        attacker = (dict_df[solver]['attacker'].values[0]).upper()
        title_str = '|Q| = '+ str(n_Q) + ',|X| = '+ str(n_X) +',|T| = '+ str(n_T) +',rho = '+ str(rho)[:4] +',k= '+ str(k)+' ('+ attacker+')'
        if y_axis_boolean == True:
            plt.yticks(y_axis_t)
        plt.title(title_str)
        if save_boolean ==True:
            pr_str = 'Q_'+ str(n_Q) + 'X_'+ str(n_X) +'_T_'+ str(n_T) +'_rho_'+ str(rho)[:4] +'_k_'+ str(k)+'_'+ attacker+'_'
            plt.savefig(dir_string+pr_str +'_'+".jpg",bbox_inches='tight')
        if multigrid == False:
            plt.show()
    else:
        print('WARNING:NOT COMPARING THE SAME PROBLEMS')


def grid_solver_quality(dict_df, color_dict, fig_size = (22,12),col_elements = 3,mean_ind = False,y_axis_boolean_ = True, y_axis_t_ = np.arange(22.5,39.5,1.5),title_size = 24, save_boolean = False, save_string = '', dir_string = ''):
    
    if len(set(warning_call_v1(dict_df).values())) == 1:
        prod = int(len(dict_df))/col_elements
        if len(dict_df)%col_elements!=0:
            prod+=1
        fig = plt.figure(figsize = fig_size)
        grid_str = str(int(prod))+str(col_elements)
        if y_axis_boolean_ == False:
            min_val = obtain_min_value(dict_df) - 3.5
            max_val = obtain_max_value(dict_df) + 3.5
            y_axis_t_ = np.arange(min_val, max_val, 1.5)
            y_axis_boolean_ = True
        for i,j in enumerate(dict_df):
            ax1 = fig.add_subplot(int(grid_str+str(i+1)))
            plot_quality_one_optimizer_v1(dict_df[j], color_dict[j], solver_label = j, grid=True, mean_flag = mean_ind, 
                                          y_axis_boolean = y_axis_boolean_, y_axis_t = y_axis_t_)
     
        solver =  list(dict_df.keys())[0]
        n_Q = dict_df[solver]['|Q|'].values[0]
        n_X = dict_df[solver]['|X|'].values[0]
        n_T = dict_df[solver]['|T|'].values[0]
        rho = dict_df[solver]['rho'].values[0]
        k = dict_df[solver]['k'].values[0]
        attacker = (dict_df[solver]['attacker'].values[0]).upper()
        title_str = '|Q| = '+ str(n_Q) + ',|X| = '+ str(n_X) +',|T| = '+ str(n_T) +',rho = '+ str(rho)[:4] +',k= '+ str(k)+' ('+ attacker+')'
        fig.tight_layout()
        plt.suptitle(title_str,fontsize = title_size,y = 1.05)
        if save_boolean ==True:
            pr_str = 'Q_'+ str(n_Q) + 'X_'+ str(n_X) +'_T_'+ str(n_T) +'_rho_'+ str(rho)[:4] +'_k_'+ str(k)+'_'+ attacker+'_'
            plt.savefig(dir_string+save_string +'_'+pr_str+".jpg",bbox_inches='tight')
    else:
        print('ERROR')
        
def grid_plot_all_combinations(dDict, dict_COLOR, fig_size_ = (22,12), save_boolean = False, file_str = 'file' ,dir_string ='' ):
    
    fig = plt.figure(figsize = fig_size_)
    for d in dDict:
        ax1 = fig.add_subplot(2,2,int(d))
        plot_quality_vs_time_1(dDict[d], dict_COLOR ,save_boolean = False, only_mean = True, multigrid = True)
    if save_boolean ==True:
        plt.savefig(dir_string +file_str+".jpg",bbox_inches='tight')
    plt.show()
