import numpy as np
from hmm_utils import HMM
from params import *
from itertools import product
from scipy.special import softmax


class dec_attacker():

    def __init__(self, prior_m, transition_m, emission_m, rho_probs,
         X, w1, w2, seq, k_value):

        self.w1  = w1
        self.w2  = w2

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

        #Sequence the attacker is interested in
        self.seq = seq 

        # Mask. For each t has 1 in positions the attacker is interested
        # and -1 in the rest
        # self.seq_mask = -np.ones([self.T, self.n_components])
        self.seq_mask = np.zeros([self.T, self.n_components])
        for i in range(self.seq_mask.shape[0]):
            self.seq_mask[i, self.seq[i]] = 1.


        # Scale for getting positive utilities
        self.scale = self.w1*self.T + self.w2*self.T + 1.0



    def utility(self, z, hmm):

        Y = self.attack_X(z_matrix = z, 
            rho_matrix = hmm.rho).astype('int')

        # Compute first term of utility
        V, _ = hmm.nu(Y)
        V = softmax(V)
        p1 = np.sum(V * self.seq_mask)

        # Compute second term of utility
        vec_diff = (np.argmax(z ,axis=1) - self.X.reshape(-1))
        p2 = np.count_nonzero(vec_diff)
  
        
        return  self.w1 * p1 - self.w2 * p2 + self.scale


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

    def sample_attack(self):
        return np.diag(np.ones(self.n_obs))[np.random.choice(
                                        np.arange(self.n_obs), self.T)]

    
    



if __name__ == "__main__":

    pass






