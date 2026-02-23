import numpy as np

setup_dir = 'unc_structure_setups/'

mat = np.load(setup_dir + 'unc_structure_experiment_comb_X_20_T_20_Q_20.npy',
              allow_pickle = True).item()

unc_comb = {'lu_rho': 0.95* np.ones(mat['emiss'].shape[1]),
             'hu_rho':0.75* np.ones(mat['emiss'].shape[1]),
              'lu_k': 10000,
              'hu_k': 100}


col_sch_param = np.arange(1, 1000000, 10)
RS_iter_param = 5000
#########################################
iters_mcts_param = -1
eps_mcts_param = -1
lr_mcts_param = -1
#############################
iters_sa_param = 50
lr_sa_param = 0.005
eps_sa_param = 0.05
#################################
N_RME_param = 100

solver_param_dict = {'APS': {'cooling_schedule':col_sch_param},
                     'MCTS': {'RS_iters': RS_iter_param, 'mcts_iters':iters_mcts_param ,
                               'eps': eps_mcts_param, 'lr':lr_mcts_param, 'flag': 'MCTS'},
                      'SA': {'RS_iters': RS_iter_param, 'sa_iters':iters_sa_param ,
                       'eps': eps_sa_param, 'lr':lr_sa_param, 'flag': 'SA'},
                       'RME': {'N': N_RME_param}}





params_dict = {'trans': mat['trans'],
                'emiss': mat['emiss'],
                'prior': mat['prior'],
                 'X': mat['X'],
                'time_range':list(range(15,1215,15)),
                'n_jobs': 10,
                 'N':10,
                'atr': {'t': 17, 'state': 4, 'c': 1, 'w1':20, 'w2': 1,
                        '1': {'rho': unc_comb['lu_rho'],
                               'k': unc_comb['lu_k'],
                                'save_ite_b': True,
                               'save_gb_b': True,
                                'id_str':'unc_structure_comb_1_',
                              'params_solver': solver_param_dict,
                'APS':{'save_ite_d': 'str_unc/APS/ss_atr_att/ite/comb1/',
                                        'save_gb_d': 'str_unc/APS/ss_atr_att/all/' ,
                                        'flag_d':'flag/APS/ss_atr_att/ite/comb1/'},
                'MCTS': {'save_ite_d': 'str_unc/MCTS/ss_atr_att/ite/comb1/',
                                        'save_gb_d': 'str_unc/MCTS/ss_atr_att/all/',
                                        'flag_d': 'flag/MCTS/ss_atr_att/ite/comb1/'},
                'SA': {'save_ite_d': 'str_unc/SA/ss_atr_att/ite/comb1/',    #
                                        'save_gb_d': 'str_unc/SA/ss_atr_att/all/',   #
                                        'flag_d': 'flag/SA/ss_atr_att/ite/comb1/'},  #
                 'RME': {'save_ite_d': 'str_unc/RME/ss_atr_att/ite/comb1/',
                                        'save_gb_d': 'str_unc/RME/ss_atr_att/all/',
                                        'flag_d': 'flag/RME/ss_atr_att/ite/comb1/'}},
                         '2': {'rho':unc_comb['lu_rho'],
                               'k': unc_comb['hu_k'],
                                'save_ite_b': True,
                               'save_gb_b': True,
                                'id_str': 'unc_structure_comb_2_' ,
                               'params_solver': solver_param_dict,
                'APS':{'save_ite_d': 'str_unc/APS/ss_atr_att/ite/comb2/',
                                        'save_gb_d': 'str_unc/APS/ss_atr_att/all/' ,
                                        'flag_d':'flag/APS/ss_atr_att/ite/comb2/'},
                'MCTS': {'save_ite_d': 'str_unc/MCTS/ss_atr_att/ite/comb2/',
                                        'save_gb_d': 'str_unc/MCTS/ss_atr_att/all/',
                                        'flag_d': 'flag/MCTS/ss_atr_att/ite/comb2/'},
                'SA': {'save_ite_d': 'str_unc/SA/ss_atr_att/ite/comb2/',              #
                                        'save_gb_d': 'str_unc/SA/ss_atr_att/all/',    #
                                        'flag_d': 'flag/SA/ss_atr_att/ite/comb2/'},   #
                 'RME': {'save_ite_d': 'str_unc/RME/ss_atr_att/ite/comb2/',
                                        'save_gb_d': 'str_unc/RME/ss_atr_att/all/',
                                        'flag_d': 'flag/RME/ss_atr_att/ite/comb2/'}},
                        '3': {'rho': unc_comb['hu_rho'],
                               'k': unc_comb['lu_k'],
                                'save_ite_b': True,
                               'save_gb_b': True,
                                'id_str': 'unc_structure_comb_3_',
                              'params_solver': solver_param_dict,
               'APS':{'save_ite_d': 'str_unc/APS/ss_atr_att/ite/comb3/',
                                        'save_gb_d': 'str_unc/APS/ss_atr_att/all/',
                                        'flag_d':'flag/APS/ss_atr_att/ite/comb3/'},
                'MCTS': {'save_ite_d': 'str_unc/MCTS/ss_atr_att/ite/comb3/',
                                        'save_gb_d': 'str_unc/MCTS/ss_atr_att/all/',
                                        'flag_d': 'flag/MCTS/ss_atr_att/ite/comb3/'},
                'SA': {'save_ite_d': 'str_unc/SA/ss_atr_att/ite/comb3/',               #
                                        'save_gb_d': 'str_unc/SA/ss_atr_att/all/',      #
                                        'flag_d': 'flag/SA/ss_atr_att/ite/comb3/'},    #
                 'RME': {'save_ite_d': 'str_unc/RME/ss_atr_att/ite/comb3/',
                                        'save_gb_d': 'str_unc/RME/ss_atr_att/all/',
                                        'flag_d': 'flag/RME/ss_atr_att/ite/comb3/'}},
                        '4': {'rho': unc_comb['hu_rho'],
                               'k': unc_comb['hu_k'],
                                'save_ite_b': True,
                               'save_gb_b': True,
                                'id_str': 'unc_structure_comb_4_',
                              'params_solver': solver_param_dict,
               'APS':{'save_ite_d': 'str_unc/APS/ss_atr_att/ite/comb4/',
                                        'save_gb_d': 'str_unc/APS/ss_atr_att/all/' ,
                                        'flag_d':'flag/APS/ss_atr_att/ite/comb4/'},
                'MCTS': {'save_ite_d': 'str_unc/MCTS/ss_atr_att/ite/comb4/',
                                        'save_gb_d': 'str_unc/MCTS/ss_atr_att/all/',
                                        'flag_d': 'flag/MCTS/ss_atr_att/ite/comb4/'},
                'SA': {'save_ite_d': 'str_unc/SA/ss_atr_att/ite/comb4/',   #
                                        'save_gb_d': 'str_unc/SA/ss_atr_att/all/',   #
                                        'flag_d': 'flag/SA/ss_atr_att/ite/comb4/'},   #
                 'RME': {'save_ite_d': 'str_unc/RME/ss_atr_att/ite/comb4/',       
                                        'save_gb_d': 'str_unc/RME/ss_atr_att/all/',
                                        'flag_d': 'flag/RME/ss_atr_att/ite/comb4/'}}},
                        
                  'rep': {'t':16, 'state': 7, 'c': -1, 'w1':15, 'w2': 1,
                         '1': {'rho': unc_comb['lu_rho'],
                               'k': unc_comb['lu_k'],
                                'save_ite_b': True,
                               'save_gb_b': True,
                                'id_str': 'unc_structure_comb_1_',
                               'params_solver':solver_param_dict,
                 'APS':{'save_ite_d': 'str_unc/APS/ss_rep_att/ite/comb1/',
                                        'save_gb_d': 'str_unc/APS/ss_rep_att/all/',
                                        'flag_d':'flag/APS/ss_rep_att/ite/comb1/'},
                'MCTS': {'save_ite_d': 'str_unc/MCTS/ss_rep_att/ite/comb1/',
                                        'save_gb_d': 'str_unc/MCTS/ss_rep_att/all/',
                                        'flag_d': 'flag/MCTS/ss_rep_att/ite/comb1/'},
                'SA': {'save_ite_d': 'str_unc/SA/ss_rep_att/ite/comb1/',         #
                                        'save_gb_d': 'str_unc/SA/ss_rep_att/all/',  #
                                        'flag_d': 'flag/SA/ss_rep_att/ite/comb1/'},  #
                 'RME': {'save_ite_d': 'str_unc/RME/ss_rep_att/ite/comb1/',
                                        'save_gb_d': 'str_unc/RME/ss_rep_att/all/',
                                        'flag_d': 'flag/RME/ss_rep_att/ite/comb1/'}},
                         '2': {'rho':unc_comb['lu_rho'],
                               'k': unc_comb['hu_k'],
                                'save_ite_b': True,
                               'save_gb_b': True,
                                'id_str': 'unc_structure_comb_2_',
                               'params_solver': solver_param_dict,
               'APS':{'save_ite_d': 'str_unc/APS/ss_rep_att/ite/comb2/',
                                        'save_gb_d': 'str_unc/APS/ss_rep_att/all/',
                                        'flag_d':'flag/APS/ss_rep_att/ite/comb2/'},
                'MCTS': {'save_ite_d': 'str_unc/MCTS/ss_rep_att/ite/comb2/',
                                        'save_gb_d': 'str_unc/MCTS/ss_rep_att/all/',
                                        'flag_d': 'flag/MCTS/ss_rep_att/ite/comb2/'},
                'SA': {'save_ite_d': 'str_unc/SA/ss_rep_att/ite/comb2/',       #
                                        'save_gb_d': 'str_unc/SA/ss_rep_att/all/',  #
                                        'flag_d': 'flag/SA/ss_rep_att/ite/comb2/'},  #
                 'RME': {'save_ite_d': 'str_unc/RME/ss_rep_att/ite/comb2/',
                                        'save_gb_d': 'str_unc/RME/ss_rep_att/all/',
                                        'flag_d': 'flag/RME/ss_rep_att/ite/comb2/'}},
                        '3': {'rho': unc_comb['hu_rho'],
                               'k': unc_comb['lu_k'],
                                'save_ite_b': True,
                               'save_gb_b': True,
                                'id_str': 'unc_structure_comb_3_',
                              'params_solver': solver_param_dict,
               'APS':{'save_ite_d': 'str_unc/APS/ss_rep_att/ite/comb3/',
                                        'save_gb_d': 'str_unc/APS/ss_rep_att/all/',
                                        'flag_d':'flag/APS/ss_atr_att/ite/comb3/'},
                'MCTS': {'save_ite_d': 'str_unc/MCTS/ss_rep_att/ite/comb3/',
                                        'save_gb_d': 'str_unc/MCTS/ss_rep_att/all/',
                                        'flag_d': 'flag/MCTS/ss_rep_att/ite/comb3/'},
                'SA': {'save_ite_d': 'str_unc/SA/ss_rep_att/ite/comb3/',              #
                                        'save_gb_d': 'str_unc/SA/ss_rep_att/all/',    #
                                        'flag_d': 'flag/SA/ss_rep_att/ite/comb3/'},   #
                 'RME': {'save_ite_d': 'str_unc/RME/ss_rep_att/ite/comb3/',
                                        'save_gb_d': 'str_unc/RME/ss_rep_att/all/',
                                        'flag_d': 'flag/RME/ss_rep_att/ite/comb3/'}},
                         
                        '4': {'rho': unc_comb['hu_rho'],
                               'k': unc_comb['hu_k'],
                                'save_ite_b': True,
                               'save_gb_b': True,
                                'id_str': 'unc_structure_comb_4_',
                              'params_solver': solver_param_dict,
               'APS':{'save_ite_d': 'str_unc/APS/ss_rep_att/ite/comb4/',
                                        'save_gb_d': 'str_unc/APS/ss_rep_att/all/',
                                        'flag_d':'flag/APS/ss_atr_att/ite/comb4/'},
                'MCTS': {'save_ite_d': 'str_unc/MCTS/ss_rep_att/ite/comb4/',
                                        'save_gb_d': 'str_unc/MCTS/ss_rep_att/all/',
                                        'flag_d': 'flag/MCTS/ss_rep_att/ite/comb4/'},
                'SA': {'save_ite_d': 'str_unc/SA/ss_rep_att/ite/comb4/',            #
                                        'save_gb_d': 'str_unc/SA/ss_rep_att/all/',   #
                                        'flag_d': 'flag/SA/ss_rep_att/ite/comb4/'},   #
                 'RME': {'save_ite_d': 'str_unc/RME/ss_rep_att/ite/comb1/',
                                        'save_gb_d': 'str_unc/RME/ss_atr_att/all/',
                                        'flag_d': 'flag/RME/ss_atr_att/ite/comb1/'}}},
                        
                'dd': {'t':16, 'w1':3, 'w2': 1,
                          '1': {'rho': unc_comb['lu_rho'],
                               'k': unc_comb['lu_k'],
                                'save_ite_b': True,
                               'save_gb_b': True,
                                'id_str': 'unc_structure_comb_1_',
                                'params_solver': solver_param_dict,
            'APS':{'save_ite_d': 'str_unc/APS/dd_att/ite/comb1/',
                                        'save_gb_d': 'str_unc/APS/dd_att/all/',
                                        'flag_d':'flag/APS/dd_att/ite/comb1/'},
                'MCTS': {'save_ite_d': 'str_unc/MCTS/dd_att/ite/comb1/',
                                        'save_gb_d': 'str_unc/MCTS/dd_att/all/',
                                        'flag_d': 'flag/MCTS/dd_att/ite/comb1/'},
                'SA': {'save_ite_d': 'str_unc/SA/dd_att/ite/comb1/',             #
                                        'save_gb_d': 'str_unc/SA/dd_att/all/',     #
                                        'flag_d': 'flag/SA/dd_att/ite/comb1/'},    #
                 'RME': {'save_ite_d': 'str_unc/RME/dd_att/ite/comb1/',
                                        'save_gb_d': 'str_unc/RME/dd_att/all/',
                                        'flag_d': 'flag/RME/dd_att/ite/comb1/'}},
                         '2': {'rho':unc_comb['lu_rho'],
                               'k': unc_comb['hu_k'],
                                'save_ite_b': True,
                               'save_gb_b': True,
                                'id_str': 'unc_structure_comb_2_',
                               'params_solver': solver_param_dict,
              'APS':{'save_ite_d': 'str_unc/APS/dd_att/ite/comb2/',
                                        'save_gb_d': 'str_unc/APS/dd_att/all/',
                                        'flag_d':'flag/APS/dd_att/ite/comb2/'},
                'MCTS': {'save_ite_d': 'str_unc/MCTS/dd_att/ite/comb2/',
                                        'save_gb_d': 'str_unc/MCTS/dd_att/all/',
                                        'flag_d': 'flag/MCTS/dd_att/ite/comb2/'},
                'SA': {'save_ite_d': 'str_unc/SA/dd_att/ite/comb2/',             #
                                        'save_gb_d': 'str_unc/SA/dd_att/all/',   #
                                        'flag_d': 'flag/SA/dd_att/ite/comb2/'},   #
                 'RME': {'save_ite_d': 'str_unc/RME/dd_att/ite/comb2/',
                                        'save_gb_d': 'str_unc/RME/dd_att/all/',
                                        'flag_d': 'flag/RME/dd_att/ite/comb2/'}},
                        '3': {'rho': unc_comb['hu_rho'],
                               'k': unc_comb['lu_k'],
                                'save_ite_b': True,
                               'save_gb_b': True,
                                'id_str': 'unc_structure_comb_3_',
                              'params_solver':solver_param_dict,
               'APS':{'save_ite_d': 'str_unc/APS/dd_att/ite/comb3/',
                                        'save_gb_d': 'str_unc/APS/dd_att/all/',
                                        'flag_d':'flag/APS/dd_att/ite/comb3/'},
                'MCTS': {'save_ite_d': 'str_unc/MCTS/ss_atr_att/ite/comb1/',
                                        'save_gb_d': 'str_unc/MCTS/dd_att/all/',
                                        'flag_d': 'flag/MCTS/dd_att/ite/comb3/'},
                'SA': {'save_ite_d': 'str_unc/SA/dd_att/ite/comb3/',              #
                                        'save_gb_d': 'str_unc/SA/dd_att/all/',   #
                                        'flag_d': 'flag/SA/dd_att/ite/comb3/'},  #
                 'RME': {'save_ite_d': 'str_unc/RME/dd_att/ite/comb3/',
                                        'save_gb_d': 'str_unc/RME/dd_att/all/',
                                        'flag_d': 'flag/RME/dd_att/ite/comb3/'}},
                         
                        '4': {'rho': unc_comb['hu_rho'],
                               'k': unc_comb['hu_k'],
                                'save_ite_b': True,
                               'save_gb_b': True,
                                'id_str': 'unc_structure_comb_4_',
                              'params_solver': solver_param_dict,
        'APS':{'save_ite_d': 'str_unc/APS/dd_att/ite/comb3/',
                                        'save_gb_d': 'str_unc/APS/dd_att/all/',
                                        'flag_d':'flag/APS/dd_att/ite/comb3/'},
                'MCTS': {'save_ite_d': 'str_unc/MCTS/dd_att/ite/comb3/',
                                        'save_gb_d': 'str_unc/MCTS/dd_att/all/',
                                        'flag_d': 'flag/MCTS/dd_att/ite/comb3/'},   
                'SA': {'save_ite_d': 'str_unc/SA/dd_att/ite/comb4/',       #
                                        'save_gb_d': 'str_unc/SA/dd_att/all/',  #
                                        'flag_d': 'flag/SA/dd_att/ite/comb4/'},  #
                 'RME': {'save_ite_d': 'str_unc/RME/dd_att/ite/comb3/',
                                        'save_gb_d': 'str_unc/RME/dd_att/all/',
                                        'flag_d': 'flag/RME/dd_att/ite/comb3/'}}},
                    
                 'pd': {'seq':np.zeros(mat['X'].shape[0],dtype = int), 'w1':2.55, 'w2': 1,
                         '1': {'rho': unc_comb['lu_rho'],
                               'k': unc_comb['lu_k'],
                                'save_ite_b': True,
                               'save_gb_b': True,
                                'id_str': 'unc_structure_comb_1_',
                               'params_solver': solver_param_dict,
           'APS':{'save_ite_d': 'str_unc/APS/pd_att/ite/comb1/',
                                        'save_gb_d': 'str_unc/APS/pd_att/all/',
                                        'flag_d':'flag/APS/pd_att/ite/comb1/'},
                'MCTS': {'save_ite_d': 'str_unc/MCTS/pd_att/ite/comb1/',
                                        'save_gb_d': 'str_unc/MCTS/pd_att/all/',
                                        'flag_d': 'flag/MCTS/pd_att/ite/comb1/'},
                'SA': {'save_ite_d': 'str_unc/SA/pd_att/ite/comb1/',            #
                                        'save_gb_d': 'str_unc/SA/pd_att/all/',      #
                                        'flag_d': 'flag/SA/pd_att/ite/comb1/'},    #
                 'RME': {'save_ite_d': 'str_unc/RME/pd_att/ite/comb1/',
                                        'save_gb_d': 'str_unc/RME/pd_att/all/',
                                        'flag_d': 'flag/RME/pd_att/ite/comb1/'}},
                         '2': {'rho':unc_comb['lu_rho'],
                               'k': unc_comb['hu_k'],
                                'save_ite_b': True,
                               'save_gb_b': True,
                                'id_str': 'unc_structure_comb_2_',
                               'params_solver':solver_param_dict,
                'APS':{'save_ite_d': 'str_unc/APS/ss_atr_att/ite/comb2/',
                                        'save_gb_d': 'str_unc/APS/ss_atr_att/all/',
                                        'flag_d':'flag/APS/pd_att/ite/comb2/'},
                'MCTS': {'save_ite_d': 'str_unc/MCTS/pd_att/ite/comb1/',
                                        'save_gb_d': 'str_unc/MCTS/pd_att/all/',
                                        'flag_d': 'flag/MCTS/pd_att/ite/comb2/'},
                'SA': {'save_ite_d': 'str_unc/SA/pd_att/ite/comb2/',             #
                                        'save_gb_d': 'str_unc/SA/pd_att/all/',     #
                                        'flag_d': 'flag/SA/pd_att/ite/comb2/'},    #
                 'RME': {'save_ite_d': 'str_unc/RME/pd_att/ite/comb2/',
                                        'save_gb_d': 'str_unc/RME/pd_att/all/',
                                        'flag_d': 'flag/RME/pd_att/ite/comb2/'}},
                        '3': {'rho': unc_comb['hu_rho'],
                               'k': unc_comb['lu_k'],
                                'save_ite_b': True,
                               'save_gb_b': True,
                                'id_str': 'unc_structure_comb_3_',
                              'params_solver': solver_param_dict,
          'APS':{'save_ite_d': 'str_unc/APS/pd_att/ite/comb3/',
                                        'save_gb_d': 'str_unc/APS/pd_att/all/',
                                        'flag_d':'flag/APS/pd_att/ite/comb3/'},
                'MCTS': {'save_ite_d': 'str_unc/MCTS/pd_att/ite/comb3/',
                                        'save_gb_d': 'str_unc/MCTS/pd_att/all/',
                                        'flag_d': 'flag/MCTS/pd_att/ite/comb3/'},
                'SA': {'save_ite_d': 'str_unc/SA/pd_att/ite/comb3/',            #
                                        'save_gb_d': 'str_unc/SA/pd_att/all/',   #
                                        'flag_d': 'flag/SA/pd_att/ite/comb3/'},   #
                 'RME': {'save_ite_d': 'str_unc/RME/pd_att/ite/comb3/',
                                        'save_gb_d': 'str_unc/RME/pd_att/all/',
                                        'flag_d': 'flag/RME/pd_att/ite/comb3/'}},
                         
                        '4': {'rho': unc_comb['hu_rho'],
                               'k': unc_comb['hu_k'],
                                'save_ite_b': True,
                               'save_gb_b': True,
                                'id_str':'unc_structure_comb_4_',
                              'params_solver': solver_param_dict,
           'APS':{'save_ite_d': 'str_unc/APS/pd_att/ite/comb4/',
                                        'save_gb_d': 'str_unc/APS/pd_att/all/',
                                        'flag_d':'flag/APS/pd_att/ite/comb4/'},
                'MCTS': {'save_ite_d': 'str_unc/MCTS/pd_att/ite/comb1/',
                                        'save_gb_d': 'str_unc/MCTS/pd_att/all/',
                                        'flag_d': 'flag/MCTS/pd_att/ite/comb4/'},
                'SA': {'save_ite_d': 'str_unc/SA/pd_att/ite/comb4/',                  #
                                        'save_gb_d': 'str_unc/SA/pd_att/all/',        #
                                        'flag_d': 'flag/SA/pd_att/ite/comb4/'},       #
                 'RME': {'save_ite_d': 'str_unc/RME/pd_att/ite/comb4/',
                                        'save_gb_d': 'str_unc/RME/ss_atr_att/all/',
                                        'flag_d': 'flag/RME/pd_att/ite/comb4/'}}}}

