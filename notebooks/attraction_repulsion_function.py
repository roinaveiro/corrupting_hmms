import numpy as np
from itertools import product
from tqdm import tqdm_notebook as tqdm
from hmm_utils import *
import warnings
warnings.filterwarnings('ignore')




def compute_y_t(z_matrix, rho_matrix, x_vector):
    t, n_obs = z_matrix.shape
    j_obs = np.arange(0,z_matrix.shape[1])
    y_t_1 = np.dot(z_matrix* rho_matrix, j_obs)
    y_t_2 = np.sum(z_matrix *(1-rho_matrix)* x_vector, axis = 1)
    y_t = (y_t_1 + y_t_2).reshape(z_matrix.shape[0], 1)
    
    return y_t


def state_attraction_repulsion_f1(t, state, alpha, beta, c ):
    ## We consider that time starts at t = 1
    p  = np.exp(alpha[t-1] + beta[t-1])
    p /= np.sum(p)

    return c* p[state]


"""
def compute_y_t(Z_M, rho_M, X_M):
    A = Z_M * rho_M 
    Y = np.where(A == 1)[1]
    Y[ np.where(~A.any(axis=1))[0]] = X_M[np.where(~A.any(axis=1))[0]]
    return Y
"""

def f2_function(z_matrix, x_obs_vector):
    z_vector = np.argmax(z_matrix ,axis=1)
    vec_diff = (z_vector - x_obs_vector.reshape(-1))
    
    return np.count_nonzero(vec_diff)

def utility_u1_state_attraction_repulsion_function(hmm, t, c, state, w1, w2, z_matrix, x_obs_vector, y_t):
    alpha = hmm.alpha(y_t)
    beta = hmm.beta(y_t)
    
    return   w1 * state_attraction_repulsion_f1(t, state, alpha, beta, c ) - w2 * f2_function(z_matrix, x_obs_vector)


def generate_rho_sample(t, n_obs, theta_v = None, params_beta = {'a':1,'b':1}):
    
    if theta_v == None:
        params_beta['size'] = n_obs
        theta_v = np.random.beta(**params_beta)
        
    rho =  np.zeros([t, n_obs])
    for n in range(n_obs):
        rho[:,n] = np.random.choice([0,1],p=[theta_v[n],
                                           1-theta_v[n]],size=t)
        

    return rho

def generate_Z_set(n_obs,t):
    diag_matrix =  np.diag(np.ones(n_obs))
    return np.array( list(product((diag_matrix), repeat = t)))


def monte_carlo_enumeration_state_attraction_repulsion(g_w, N, x_vector, utility_function ,utility_params):
    n_states, n_obs = g_w['B'].shape
    state = utility_params['state']
    Z_set = generate_Z_set(n_obs,len(x_vector))
    Z_len = len(Z_set)
    u_z_vector = np.zeros(Z_len)
    
    for i_z,z_matrix in tqdm(enumerate(Z_set)):
        
        sum_u_values  = 0 
        
        for n in range(N):
            transtion_mat = g_w['A']
            emission_mat = g_w['B']
            priors_vec = g_w['pi']
            hmm_n = HMM(n_states)
            hmm_n.startprob_ = priors_vec
            hmm_n.transmat_ = transtion_mat
            hmm_n.emissionprob_ = emission_mat
            rho_matrix = np.ones([len(x_vector),n_obs]) 
            y_t = compute_y_t(z_matrix, rho_matrix, x_vector).astype('int')
            u_value = utility_function(hmm = hmm_n, z_matrix = z_matrix, x_obs_vector = x_vector, y_t = y_t, **utility_params)
            sum_u_values += u_value
            
        u_z_vector[i_z] = sum_u_values/N
                        
    arg_max = np.argmax(u_z_vector)
    z_star = Z_set[arg_max]
    
    return z_star 