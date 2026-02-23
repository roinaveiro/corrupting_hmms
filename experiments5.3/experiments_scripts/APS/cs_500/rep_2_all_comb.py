from solver_attacker_time_experiment_utils_v1 import *
from params_experiments_2 import *
import datetime
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)



att_str = 'rep'   ### THING TO CHANGE
comb_list_num = ['2', '3','4', '1']
solver_str = ['APS']

par_att_d  = params_dict[att_str]

for num in comb_list_num:
    par_att_comb_d  = par_att_d[num]
    for sol_str in solver_str:
        par_att_comb_s_d = par_att_comb_d[sol_str]
        
        print('trans shape',params_dict['trans'].shape)
        print('emiss shape',params_dict['emiss'].shape)
        print('prior shape',params_dict['prior'].shape)
        print('X shape',params_dict['X'].shape)
        print('rho', par_att_comb_d['rho'][0])
        print('k', par_att_comb_d['k'])
        
        att = ss_attacker(params_dict['prior'],            #THING TO CHANGE
                             params_dict['trans'], 
                             params_dict['emiss'], 
                             par_att_comb_d['rho'],
                             params_dict['X'], 
                             w1 = par_att_d['w1'],
                             w2 = par_att_d['w2'] , 
                             t = par_att_d['t'],
                             state = par_att_d['state'],
                             c = par_att_d['c'],
                             k_value= par_att_comb_d['k'])



        print('Starting COMBINATION # ...',  num)
        print('Starting ', sol_str ,' solver ...')
        start = time.time()

        parallel_ver_hor_solve_attck_problem(attacker = att, 
                                   solver_str = sol_str, 
                                   time_list = params_dict['time_range'], 
                                   N = params_dict['N'],
                                  params_solver = par_att_comb_d['params_solver'],
                                   num_jobs = params_dict['n_jobs'], 
                                   save_ite = par_att_comb_d['save_ite_b'],
                                   save_dir_ite = par_att_comb_s_d['save_ite_d'], 
                                    save_glb = par_att_comb_d['save_gb_b'] , 
                                    save_dir_glb = par_att_comb_s_d['save_gb_d'] , 
                                   exp_str = par_att_comb_d['id_str'],
                                   flag_dir = par_att_comb_s_d['flag_d'])

        end = time.time()

        print(sol_str ,' solver finished ...')
        print('Time for solving ', sol_str ,'...', str(datetime.timedelta(seconds=(end-start))))
        
    print('COMBINATION # ...', num, ' finished !')

