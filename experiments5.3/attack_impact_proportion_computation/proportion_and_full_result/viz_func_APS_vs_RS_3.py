import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
plt.style.use("fivethirtyeight")

color_params_red ={"color_mean": "red", "color_std": "darkred", "color_sh": "lightsalmon"}
color_params_blue = {'color_mean':"blue", 'color_std':'darkblue',"color_sh":"lightblue"}
color_params_green = {'color_mean':'green', 'color_std': 'darkgreen', 'color_sh':'lightgreen'}
color_params_yellow = {'color_mean':'gold', 'color_std': 'lightyellow', 'color_sh':'yellow'}
color_params_purple = {'color_mean':'darkviolet', 'color_std': 'orchid', 'color_sh':'violet'}
color_params_orange = {'color_mean':'orange', 'color_std': 'darkorange', 'color_sh':'navajowhite'}
color_params_brown = {'color_mean':'chocolate', 'color_std': 'saddlebrown', 'color_sh':'sandybrown'}






dict_COLOR = {'E':color_params_red,
           'F': color_params_blue,
            'A': color_params_green,
             'B': color_params_yellow,
              'C': color_params_purple,
              'D': color_params_orange}

dict_COLOR_APS = {'A':color_params_red,
           'B': color_params_blue}

dict_color_diff_solvers = {'APS_A': color_params_red , 
                    'APS_B': color_params_blue, 
                    'RS': color_params_green, 
                    'SA': color_params_orange,
                     'RME': color_params_purple}



dshyp = {'2':{'attr':{'1': {'RS':'64_64_lr_0005', 'SA': 'B'},
                               '2': {'RS':'64_64_lr_0005', 'SA': 'B'},
                                '3': {'RS':'64_64_lr_0005', 'SA': 'B'},
                                 '4': {'RS':'64_64_lr_0005', 'SA': 'B'}},
                          'rep': {'1': {'RS':'64_64_lr_0005', 'SA': 'B'},
                                  '2': {'RS':'64_64_lr_0005', 'SA': 'B'},
                                   '3': {'RS':'64_64_lr_0005', 'SA': 'B'},
                                    '4': {'RS':'64_64_lr_0005', 'SA': 'B'}},
                            'dd': {'1': {'RS':'64_64_lr_0005', 'SA': 'B'}, 
                                    '2': {'RS':'B', 'SA': 'B'},
                                      '3':{'RS':'64_64_lr_0005', 'SA': 'B'},
                                       '4': {'RS':'B', 'SA': 'B'}},
                             'pd': {'1': {'RS':'B', 'SA': 'D'},
                                     '2': {'RS':'B', 'SA': 'D'},
                                     '3': {'RS':'B', 'SA': 'D'},
                                      '4': {'RS':'B', 'SA': 'D'}}},
                   '3':{'attr':{'1': {'RS':'B', 'SA': 'A'},
                               '2': {'RS':'C', 'SA': 'A'},
                                '3': {'RS':'B', 'SA': 'B'},
                                 '4': {'RS':'B', 'SA': 'B'}},
                          'rep': {'1': {'RS':'B', 'SA': 'B'},
                                  '2': {'RS':'64_64_lr_0005', 'SA': 'A'},
                                   '3': {'RS':'A', 'SA': 'B'},
                                    '4': {'RS':'C', 'SA': 'B'}},
                            'dd': {'1': {'RS':'B', 'SA': 'B'}, 
                                    '2': {'RS':'64_64_lr_0005', 'SA': 'A'},
                                      '3':{'RS':'B', 'SA': 'B'},
                                       '4': {'RS':'C', 'SA': 'B'}},
                             'pd': {'1': {'RS':'D', 'SA': 'D'},
                                     '2': {'RS':'A', 'SA': 'A'},
                                     '3': {'RS':'C', 'SA': 'B'},
                                      '4': {'RS':'64_64_lr_0005', 'SA': 'B'}}}}

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
            plt.savefig(dir_string+pr_str +'_'+".pdf",bbox_inches='tight')
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
            plt.savefig(dir_string+save_string +'_'+pr_str+".pdf",bbox_inches='tight')
    else:
        print('ERROR')
        
def grid_plot_all_combinations(dDict, dict_COLOR, fig_size_ = (22,12), save_boolean = False, file_str = 'file' ,dir_string ='' ,step_ =2, margin_ = 3.5):
    
    fig = plt.figure(figsize = fig_size_)
    y_axis_t_ = obtain_y_axis_range(dDict, step = step_, margin = margin_)
    for d in dDict:
        ax1 = fig.add_subplot(2,2,int(d))
        plot_quality_vs_time_1(dDict[d], dict_COLOR ,save_boolean = False, only_mean = True, multigrid = True, y_axis_boolean=True, y_axis_t= y_axis_t_)
    if save_boolean ==True:
        plt.savefig(dir_string +file_str+".pdf",bbox_inches='tight')
    plt.show()
    

def create_dict_DF_APS(exp, att_str, comb_num):
    dict_DF = OrderedDict()
    exp_dir = 'experiment_'+str(exp)+'_scaled/'
    for hyp_c in ['cs_1', 'cs_500']:
        for f in os.listdir(exp_dir + '/' +'APS'+ '/'+ att_str+'/' + hyp_c):
            if 'comb_'+str(comb_num) in f:
                if hyp_c == 'cs_1':
                    dict_DF['A'] = pd.read_json(exp_dir + '/' +'APS' + '/'+ att_str+'/' + hyp_c +'/'+f ,orient = 'split')
                elif hyp_c == 'cs_500':
                    dict_DF['B'] = pd.read_json(exp_dir + '/' +'APS' + '/'+ att_str+'/' + hyp_c +'/'+f ,orient = 'split')
    return dict_DF
    
def create_dict_DF_RS_SA(exp,  solver, att_str, comb_num):
    dict_DF = OrderedDict()
    exp_dir = 'experiment_'+str(exp)+'_scaled'
    #for hyp_c in os.listdir(exp_dir + '/' +solver + '/'+ att_str):
    for hyp_c in ['A', 'B', 'C', 'D','64_64_lr_0005', '64_64_lr_01']:
        for f in os.listdir(exp_dir + '/' +solver + '/'+ att_str+'/' + hyp_c):
            if 'comb_'+str(comb_num) in f:
                if hyp_c == 'A':
                    dict_DF['A'] = pd.read_json(exp_dir + '/' +solver + '/'+ att_str+'/' + hyp_c +'/'+f ,orient = 'split')
                elif hyp_c == 'B':
                    dict_DF['B'] = pd.read_json(exp_dir + '/' +solver + '/'+ att_str+'/' + hyp_c +'/'+f ,orient = 'split')
                elif hyp_c == 'C':
                    dict_DF['C'] = pd.read_json(exp_dir + '/' +solver + '/'+ att_str+'/' + hyp_c +'/'+f ,orient = 'split')
                elif hyp_c == 'D':
                    dict_DF['D'] = pd.read_json(exp_dir + '/' +solver + '/'+ att_str+'/' + hyp_c +'/'+f ,orient = 'split')
                elif hyp_c == '64_64_lr_0005' :
                    dict_DF['E'] = pd.read_json(exp_dir + '/' +solver + '/'+ att_str+'/' + hyp_c +'/'+f ,orient = 'split')
                elif hyp_c == '64_64_lr_01':
                    dict_DF['F'] = pd.read_json(exp_dir + '/' +solver + '/'+ att_str+'/' + hyp_c +'/'+f ,orient = 'split')
    return dict_DF


def create_dict_DF_solver(exp_, solver_, att_str_, comb_num_):
    
    if solver_ == 'APS':
        dict_DF = create_dict_DF_APS(exp = exp_, att_str = att_str_, comb_num = comb_num_)
    elif solver_ == 'RS':
        dict_DF = create_dict_DF_RS_SA(exp = exp_, solver = solver_, att_str = att_str_, comb_num = comb_num_)
    elif solver_ == 'SA':
        dict_DF = create_dict_DF_RS_SA(exp = exp_, solver = solver_, att_str = att_str_, comb_num = comb_num_)
    
    return dict_DF

def obtain_best_combination_loop(exp_list, solver_list, att_list, comb_list):
    ld = []
    for exp in exp_list:
        for solver in solver_list:
            for att in att_list:
                for comb in comb_list:
                    dict_DF = create_dict_DF_solver(exp_ = exp , solver_ = solver , att_str_ = att, comb_num_ = comb)
                    b_hc = choose_best_combination(dict_DF)
                    ld.append({'exp':exp,
                               'solver':solver,
                                'att': att,
                                'comb_num': comb,
                                'best_hyp': b_hc})
    return pd.DataFrame(ld)

def grid_plot_all_combinations_APS(dDict, dict_COLOR, fig_size_ = (22,12), save_boolean = False, file_str = 'file' ,dir_string ='',y_axis_b = False,y_axis_t = '' ):
    
    fig = plt.figure(figsize = fig_size_)
    for d in dDict:
        ax1 = fig.add_subplot(2,2,int(d))
        if y_axis_b == False:
            plot_quality_vs_time_1(dDict[d], dict_COLOR ,save_boolean = False, only_mean = False, multigrid = True)
        elif y_axis_b ==True:
            plot_quality_vs_time_1(dDict[d], dict_COLOR ,save_boolean = False, only_mean = False, y_axis_boolean=True, multigrid = True, y_axis_t= y_axis_t)

        if d !='2':
            ax1.get_legend().remove()
    if save_boolean ==True:
        plt.savefig(dir_string +file_str+".pdf",bbox_inches='tight')
    plt.show()
    
def obtain_y_axis_range(dDict, step = 3, margin = 3.5):
    
    l = []
    for d in (dDict):
        for hyp in dDict[d]:
            l.append(dDict[d][hyp])
            
    max_val = max(pd.concat(l)['scaled_eu']) + margin
    min_val = min(pd.concat(l)['scaled_eu']) - margin
    
    return np.arange(min_val, max_val, step)

    
def create_appendix_APS_plots(exp, att_str_list, dict_color = dict_COLOR_APS,save_boolean_ = False, f_str = '', dir_str = '', 
                            step_ = 3.5, margin_ = 3.5):
    
    dDict = {}
    solver_str = 'APS'
    comb_l = ['1','2','3','4']
    for att in att_str_list:
        for comb in comb_l:
            dDict[comb] = create_dict_DF_solver(exp_ = exp , solver_ = 'APS' , att_str_ = att, comb_num_ = comb)
        y_axis_t_ = obtain_y_axis_range(dDict, step = step_, margin = margin_)
        descr_str = '_exp_'+ str(exp) +'_'+'APS'+ '_' + att + '_' 
        grid_plot_all_combinations_APS(dDict, dict_color, fig_size_ = (22,12), save_boolean = save_boolean_, file_str = f_str + descr_str
                                       ,dir_string = dir_str, y_axis_b =True, y_axis_t = y_axis_t_ )
    

def create_appendix_RS_MCTS_plots(exp, att_str_list, solver_list, comb_list ,dict_color = dict_COLOR, 
                                  save_boolean_ = False, save_str_ = '', dir_str = ''):
    
    
    for solver in solver_list:
        for att in att_str_list:
            for comb in comb_list:
                dict_DF = create_dict_DF_solver(exp, solver, att, comb)
                descr_str = '_exp_'+ str(exp) +'_'+solver + '_' + att + '_' + str(comb)  
                grid_solver_quality(dict_DF, dict_color, y_axis_boolean_ = False, 
                                save_boolean = save_boolean_, save_string = save_str_ + descr_str, dir_string = dir_str)
                
                
                
                
def create_dict_DF_APS_selected(exp, att_str, hyp_comb_list, comb_num):
    dict_DF = OrderedDict()
    exp_dir = 'experiment_'+str(exp)+'_scaled/'
    for hyp_c in hyp_comb_list:
        for f in os.listdir(exp_dir + '/' +'APS'+ '/'+ att_str+'/' + hyp_c):
            if 'comb_'+str(comb_num) in f:
                if hyp_c == 'cs_1':
                    dict_DF['APS_A'] = pd.read_json(exp_dir + '/' +'APS' + '/'+ att_str+'/' + hyp_c +'/'+f ,orient = 'split')
                elif hyp_c == 'cs_500':
                    dict_DF['APS_B'] = pd.read_json(exp_dir + '/' +'APS' + '/'+ att_str+'/' + hyp_c +'/'+f ,orient = 'split')
    return dict_DF
    
def create_dict_DF_RS_SA_selected(exp,  solver, att_str, hyp_comb_list, comb_num):
    
    dict_DF = OrderedDict()
    exp_dir = 'experiment_'+str(exp)+'_scaled'
    for hyp_c in hyp_comb_list:
        for f in os.listdir(exp_dir + '/' +solver + '/'+ att_str+'/' + hyp_c):
            if 'comb_'+str(comb_num) in f:
                dict_DF[solver] = pd.read_json(exp_dir + '/' +solver + '/'+ att_str+'/' + hyp_c +'/'+f ,orient = 'split')
                
    return dict_DF

def create_dict_RME_selected(exp, att_str, comb_num):
    
    dict_DF = OrderedDict()
    exp_dir = 'experiment_'+str(exp)+'_scaled'
    for f in os.listdir(exp_dir + '/' +'RME'+ '/'+ att_str+'/'):
        if 'comb_'+str(comb_num) in f:
            dict_DF['RME'] = pd.read_json(exp_dir + '/' +'RME' + '/'+ att_str +'/'+f ,orient = 'split')
            
    return dict_DF

def create_dict_DF_solver(exp_, solver_, att_str_, hyp_comb_list_ ,comb_num_):
    
    if solver_ == 'APS':
        dict_DF = create_dict_DF_APS_selected(exp = exp_, 
                                              att_str = att_str_, 
                                              hyp_comb_list = hyp_comb_list_ ,
                                              comb_num = comb_num_)
    elif solver_ == 'RS':
        dict_DF = create_dict_DF_RS_SA_selected(exp = exp_, 
                                                solver = solver_, 
                                                att_str = att_str_,
                                                hyp_comb_list = hyp_comb_list_,
                                                comb_num = comb_num_)
    elif solver_ == 'SA':
        dict_DF = create_dict_DF_RS_SA_selected(exp = exp_, 
                                                solver = solver_, 
                                                att_str = att_str_,
                                                hyp_comb_list = hyp_comb_list_,
                                                comb_num = comb_num_)
    elif solver_== 'RME':
        dict_DF = create_dict_RME_selected(exp = exp_, 
                                           att_str = att_str_, 
                                           comb_num = comb_num_)
        
        
    return dict_DF


                

def create_specific_dDICT_grid(exp, att_str, comb_num , aps_hp =  ['cs_1','cs_500'] ,hyp_d = dshyp ):
    
    dDict = {}
    
    for solver in  ['SA','APS','RS', 'RME']:
        
        if solver in ['SA','RS']:
            hyp_list = [hyp_d[str(exp)][att_str][str(comb_num)][solver]]
            dDF = create_dict_DF_solver(exp_ = exp , 
                                        solver_ = solver, 
                                        att_str_ = att_str, 
                                        hyp_comb_list_ = hyp_list,
                                        comb_num_ = comb_num)
        elif solver in ['RME']:
            hyp_list = []
            dDF = create_dict_DF_solver(exp_ = exp, 
                                        solver_ = solver, 
                                        att_str_ = att_str, 
                                        hyp_comb_list_ = hyp_list,
                                        comb_num_ = comb_num)
        elif solver in ['APS']:
            hyp_list = aps_hp
            dDF = create_dict_DF_solver(exp_ = exp, 
                                        solver_ = solver, 
                                        att_str_ = att_str, 
                                        hyp_comb_list_ = hyp_list,
                                        comb_num_ = comb_num)
        dDict = {**dDict,**dDF}
    
    return dDict 

def create_gen_dict_for_grid(exp_, att_str_, comb_list = ['1','2','3','4']):
    genD = {}
    for comb in comb_list:
        genD[comb] = create_specific_dDICT_grid(exp = exp_, att_str = att_str_, comb_num = comb)
    return genD

def generate_main_plots(attacker_list, exp_list, dict_color = dict_color_diff_solvers):
    for exp in exp_list:
        for att in attacker_list:
            zD = create_gen_dict_for_grid(exp_ = exp, att_str_ = att)
            exp_str = str(exp)
            attacker_str = str(zD['1']['SA']['attacker'][0])
            file_str_ = 'main_exp_' + exp_str + '_' + attacker_str 
            grid_plot_all_combinations(zD, dict_color,fig_size_ = (35,20) ,save_boolean = True, file_str = file_str_ ,dir_string ='images/experiment_'+str(exp_str)+'/main/')

