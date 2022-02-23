import numpy as np
from hmm_utils import HMM
from smoothing_state_attacker import ss_attacker
from params import *


def monte_carlo_enumeration(hmm, k_value, attacker,  N, x_vector, theta_prob_vec):
    
    Z_set = hmm.generate_z(len(x_vector))
    u_z_matrix = np.zeros((len(Z_set), N)) 
    transtion_mat_list = hmm.sample_mat(hmm.transmat_, n= N, k= k_value)
    emission_mat_list = hmm.sample_mat(hmm.emissionprob_, n= N, k= k_value)
    priors_vec_list = hmm.sample_mat(np.array([list(hmm.startprob_)]), n=N, k=k_value)
    rho_matrix_list = hmm.sample_rho(T= N, theta_v = theta_prob_vec)
    
    for i_z,z_matrix in (enumerate(Z_set)):  
        
        for n in range(N):
            hmm_n = HMM(hmm.n_components, hmm.n_obs)
            hmm_n.startprob_ = priors_vec_list[n].reshape(-1)
            hmm_n.transmat_ = transtion_mat_list[n]
            hmm_n.emissionprob_ = emission_mat_list[n]
            rho_matrix = rho_matrix_list[n]
            y_t = hmm.attack_X(X= x_vector, z_matrix = z_matrix, rho_matrix = rho_matrix).astype('int')
            u_value = attacker.utility(hmm = hmm_n, z_matrix = z_matrix, x_obs_vector = x_vector, y_t = y_t)
            u_z_matrix[i_z][n] = u_value
            
    u_z_vector = np.mean(u_z_matrix, axis=1)     
    arg_max = np.argmax(u_z_vector)
    z_star = Z_set[arg_max]
    
    return z_star, u_z_vector







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
    
    print(monte_carlo_enumeration(hmm = hmm, k_value = 100000, attacker = ss_att, N = 5, x_vector = X, theta_prob_vec =np.array([0.99,0.99,0.99,0.99,0.99,0.99])))



