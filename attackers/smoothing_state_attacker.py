import numpy as np
from hmm_utils import HMM
from params import *
from itertools import product


class ss_attacker():

    def __init__(self, prior_m, transition_m, emission_m, rho_probs,
         X, w1, w2, t, state, c, k_value):

        ## We consider that time starts at t = 1
        if t<=0:
            raise ValueError('t must be >=1')

        self.w1 = w1
        self.w2 = w2

        self.t     = t
        self.state = state
        self.c = c

        # These are the expected values to be used for sampling
        self.prior_m = prior_m
        self.transition_m = transition_m
        self.emission_m = emission_m
        self.rho_probs = rho_probs

        # This is proportional to precision in Dirichlet samples
        self.k_value = k_value 


        # This is the untainted vector of observations
        self.X = X 

        # Useful hmm parameters
        self.n_components = self.transition_m.shape[0]
        self.n_obs        = self.emission_m.shape[1]
        self.T            = self.X.shape[0]



    def utility(self, z, hmm):

        Y = self.attack_X(z_matrix = z, 
            rho_matrix = hmm.rho).astype('int')

        # Compute first term of utility
        alpha = hmm.alpha(Y)
        beta = hmm.beta(Y)
        p  = np.exp(alpha[self.t -1] + beta[self.t -1])
        p /= np.sum(p)
        p1 = self.c * p[self.state]

        # Compute second term of utility
        vec_diff = (np.argmax(z ,axis=1) - self.X.reshape(-1))
        p2 = np.count_nonzero(vec_diff)
  
        
        return  self.w1 * p1 - self.w2 * p2


    def sample_hmm(self):

        hmm_n = HMM(self.n_components, self.n_obs)

        hmm_n.startprob_ = hmm_n.sample_mat(self.prior_m[np.newaxis,:], 
                            n = 1, k = self.k_value).squeeze()
        hmm_n.transmat_ = hmm_n.sample_mat(self.transition_m, 
                            n = 1, k = self.k_value).squeeze()
        hmm_n.emissionprob_ = hmm_n.sample_mat(self.emission_m, 
                            n = 1, k = self.k_value).squeeze()

        hmm_n.rho = hmm_n.sample_rho(self.T, self.rho_probs, n = 1).squeeze()

        return hmm_n


    def expected_utility(self, z, N=1000):

        exp_util = 0.0
        for j in range(N):
            # Sample HMM 
            hmm_sample = self.sample_hmm()
            exp_util += self.utility(z, hmm_sample)

        return exp_util/N


    def attack_X(self, rho_matrix, z_matrix):
        '''
        Given attack matrix z and success matrix rho, transforms
        X into attacked version
        
        Parameters
        ----------
        '''

        j_obs = np.arange(0,z_matrix.shape[1])
        y_t_1 = np.dot(z_matrix* rho_matrix, j_obs)
        y_t_2 = np.sum(z_matrix *(1-rho_matrix)* self.X, axis = 1)
        y_t = (y_t_1 + y_t_2).reshape(z_matrix.shape[0], 1)

        return y_t


    def generate_attacks(self):
        
        diag_matrix =  np.diag(np.ones(self.n_obs))
        return np.array( list(product((diag_matrix), repeat = self.T)))
    
    



if __name__ == "__main__":

    from hmm_utils import HMM

    priors     = np.array([0.5,0.5])
    transition = np.array([[0.95, 0.05],[0.1, 0.9]])
    emission   = np.array([[1/6, 1/6, 1/6, 1/6, 1/6, 1/6],
                        [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]])

    
    print('Initialize HMM')
    # initialize HMM
    hmm = HMM(2, 6)
    hmm.startprob_ = priors
    hmm.transmat_ = transition
    hmm.emissionprob_ = emission

    print(hmm)

    print('Priors', hmm.startprob_)
    print('Transition matrix')
    print(hmm.transmat_)
    print('Emission probabilities')
    print(hmm.emissionprob_)
    
    print('Observations --- X')
    X = hmm.sample(5)[0]    
    print('Attacked observations --Y_t')
    y_t = np.ones((5)).reshape(5,1).astype('int')
    print(y_t)
    print('---------attack matrix ------------')
    z_matrix = np.zeros((len(X),hmm.n_obs))
    z_matrix[:,0] = 1
    print(z_matrix)
    
    
    print('Initialize smoothing_state_attacker')
    ss_att = ss_attacker(w1 = 2 ,w2=  2,t = 1, state=1,c=1)
    print('w1',ss_att.w1)
    print('w2',ss_att.w1)
    print('t', ss_att.t)
    print('state',ss_att.state)
    print('c', ss_att.c)
    print('Utility f1')
    print(ss_att.state_attraction_repulsion_f1(hmm.alpha(X), hmm.beta(X)))
    print('Utility f2')
    print(ss_att.f2_function(z_matrix, X))
    print('Utility')
    print(ss_att.utility(hmm, z_matrix, X, y_t))






