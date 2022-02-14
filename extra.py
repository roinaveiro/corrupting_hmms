import numpy as np
from hmm_utils import HMM
from smoothing_state_attacker import ss_attacker
from params import *


def monte_carlo_enumeration_state_attraction_repulsion(hmm, ss_att, g_w, N, x_vector, utility_function ):
    
    n_states, n_obs = g_w['B'].shape
    Z_set = hmm.generate_z(len(x_vector))
    Z_len = len(Z_set)
    u_z_vector = np.zeros(Z_len) 
    
    for i_z,z_matrix in (enumerate(Z_set)):
        
        sum_u_values  = 0 
        
        for n in range(N):
            
            transtion_mat = g_w['A']
            emission_mat = g_w['B']
            priors_vec = g_w['pi']
            hmm_n = HMM(n_states, n_obs)
            hmm_n.startprob_ = priors_vec
            hmm_n.transmat_ = transtion_mat
            hmm_n.emissionprob_ = emission_mat
            rho_matrix = np.ones([len(x_vector),n_obs]) 
            y_t = hmm.attack_X(X= x_vector, z_matrix = z_matrix, rho_matrix = rho_matrix).astype('int')
            u_value = utility_function(hmm = hmm_n, z_matrix = z_matrix, x_obs_vector = x_vector, y_t = y_t)
            sum_u_values += u_value
            
        u_z_vector[i_z] = sum_u_values/N
                        
    arg_max = np.argmax(u_z_vector)
    z_star = Z_set[arg_max]
    
    return z_star 








if __name__ == "__main__":

    priors     = np.array([0.5,0.5])
    transition = np.array([[0.95, 0.05],[0.1, 0.9]])
    emission   = np.array([[1/6, 1/6, 1/6, 1/6, 1/6, 1/6],
                        [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]])

    hmm = HMM(n_components = 2,  n_obs = 6)

    hmm.startprob_ = priors
    hmm.transmat_ = transition
    hmm.emissionprob_ = emission
    
    print('-------------- Initialize HMM -------------')
    print(hmm)

    print('Priors'   , hmm.startprob_)

    print('Transition matrix')
    print(hmm.transmat_)

    print('Emission probabilities')
    print(hmm.emissionprob_)

    X = hmm.sample(5)[0]
    print('Observation X')
    print(X)

    print('-------------- Initialize smoothing_state_attacker -------------------------')
    ss_att = ss_attacker(w1 = 6767 ,w2=  1,t = 1, state=1,c=1)
    
    g_w_dict = {'A': transition,'B': emission,'pi': priors}
    
        
    print('----------------------- Find optimal attack ------------------------')
    
    print(monte_carlo_enumeration_state_attraction_repulsion(hmm = hmm,ss_att=ss_att, g_w = g_w_dict, N = 10, x_vector = X, utility_function = ss_att.utility))




