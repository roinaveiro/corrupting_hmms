from solver_attacker_time_experiment_utils_v2 import *
from params_experiments_3_RME import *
import datetime



att_st = 'atr'   ### THING TO CHANGE
comb_list_num = ['1']
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
    
    
    
    print('Starting COMBINATION # ...',  num)
    print('Starting RME solver ...')
    
    start = time.time()
    parallel_ver_hor_solve_attck_problem(attacker = att, 
                                      solver_str = 'RME', 
                                     time_list = params_dict['time_range'], 
                                     params_solver = par_d[num]['params_solver'],
                                     N = params_dict['N'],  
                                     num_jobs = params_dict['n_jobs'], 
                                     save_ite = par_d[num]['save_ite_b'], 
                                     save_dir_ite = par_d[num]['save_ite_d_RME'], 
                                     save_glb = par_d[num]['save_gb_b'] , 
                                      save_dir_glb = par_d[num]['save_gb_d_RME'], 
                                     exp_str = par_d[num]['id_str'],
                                      flag_dir = par_d[num]['flag_d_RME'])

    
    end = time.time()
    print('RME solver finished ...')
    print('Time for solving RME ...', str(datetime.timedelta(seconds=(end-start))))
    print('COMBINATION # ...',  num, ' finished !')

