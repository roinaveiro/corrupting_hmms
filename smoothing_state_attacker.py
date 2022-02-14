import numpy as np
from params import *


class ss_attacker():

    def __init__(self, w1, w2, t, state, c):
        ## We consider that time starts at t = 1
        if t<=0:
            raise ValueError('t must be >=1')

        self.w1 = w1
        self.w2 = w2

        self.t     = t
        self.state = state
        self.c = c


    def expected_utility(self, A, N=1000):

        exp_util = 0.0
        for j in range(N):
            # Sample HMM params
            # Sample rhos
            exp_util += self.utility(A)
            pass

        return exp_util/N
    
    def state_attraction_repulsion_f1(self, alpha, beta):
       
        p  = np.exp(alpha[self.t -1] + beta[self.t -1])
        p /= np.sum(p)

        return self.c* p[self.state]
    
    def f2_function(self ,z_matrix, x_obs_vector):
        
        z_vector = np.argmax(z_matrix ,axis=1)
        vec_diff = (z_vector - x_obs_vector.reshape(-1))
        
        return np.count_nonzero(vec_diff)


    def utility(self, hmm,  z_matrix, x_obs_vector, y_t):
        
        alpha = hmm.alpha(y_t)
        beta = hmm.beta(y_t)
        
        return   self.w1 * self.state_attraction_repulsion_f1(alpha, beta) - self.w2 * self.f2_function(z_matrix, x_obs_vector)




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






