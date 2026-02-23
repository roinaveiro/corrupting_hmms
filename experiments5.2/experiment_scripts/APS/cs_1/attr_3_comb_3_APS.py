from solver_attacker_time_experiment_utils_v1 import *
from params_experiments_3_APS_RS import *
import datetime



att_st = 'atr'   ### THING TO CHANGE
comb_list_num = ['3']
par_d  = params_dict[att_st]



for num in comb_list_num:
        
    mat = params_dict['mat'][num]
    
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
                         k_value= params_dict['k'])
    
    
    
    print('Starting COMBINATION # ...',  comb_list_num)
    print('Starting APS solver ...')
    
    start = time.time()
    parallel_ver_hor_solve_attck_problem(attacker = att, 
                                      solver_str = 'APS', 
                                     time_list = params_dict['time_range'], 
                                     N = params_dict['N'],  
                                    params_solver = par_d[num]['params_solver'],
                                     num_jobs = params_dict['n_jobs'], 
                                     save_ite = par_d[num]['save_ite_b'], 
                                     save_dir_ite = par_d[num]['save_ite_d_APS'], 
                                     save_glb = par_d[num]['save_gb_b'] , 
                                      save_dir_glb = par_d[num]['save_gb_d_APS'], 
                                     exp_str = par_d[num]['id_str'],
                                      flag_dir = par_d[num]['flag_d_APS'])

    
    end = time.time()
    print('APS solver finished ...')
    print('Time for solving APS ...', str(datetime.timedelta(seconds=(end-start))))
    print('COMBINATION # ...',  num, ' finished !')

