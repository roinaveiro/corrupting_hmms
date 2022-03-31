import numpy as np

priors     = np.array([0.5,0.3,0.2])
transition = np.array([[0.85, 0.05,0.1],
                       [0.05, 0.9,0.05],
                        [1/2, 1/4, 1/4]])

emission   = np.array([[0.699, 0.05, 0.1, 0.05, 0.1, 0.001],
                    [0.001, 0.1, 0.1, 0.299, 0.3, 0.2],
                       [0.1, 0.2, 0.1, 0.2, 0.1, 0.3]])

hmm_params = {'priors': priors,
               'transition': transition,
                'emission': emission}

rho_probs = np.ones(6)

cert_params = {'rho': rho_probs,
                 'k': 100000000 }

params_exp = {'ratio': {'init_rt': 0.1,
                        'fn_rt':  300,
                       'rt_st':  0.1},
              'contour': {'start': 1,
                          'stop': 1000,
                          'n_values': 20},
                'box': True,
                 'state': 1,
                  't':5,
                   'c':-1,
                   'N1':1,
                    'N2':1}

X_obs = np.array([ [4], [3], [5], [3], [4] ])
