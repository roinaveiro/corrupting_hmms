import numpy as np
from hmmlearn import hmm


class HMM(hmm.MultinomialHMM):

    '''
    Builts HMM

    Parameters
    ----------
    n_components : int
        Number of hidden states

    '''
    
    def __init__(self, n_components, n_obs):
        
        super().__init__(n_components)  ### import requirement hmm version 0.2.6
        self.n_obs = n_obs
        
  
    def smoothing(self, X, t):
        ## We consider that time starts at t = 1
        if t<=0:
            raise ValueError('t must be >=1')


        return self.predict_proba(X)[t-1]
    
    def alpha(self, X):

        alphas = self._do_forward_pass( 
            self._compute_log_likelihood(X) )[1]

        return alphas
    
    def beta(self, X):
        
        betas = self._do_backward_pass( 
            self._compute_log_likelihood(X) )
            
        return betas
    
    
    def nu(self, X):
        
        V   = np.zeros([len(X), self.n_components]) 
        Ptr = np.zeros([len(X), self.n_components]) 

        lemissionprob_ = np.log(self.emissionprob_)
        ltransmat_     = np.log(self.transmat_.T)
        lstartprob_   = np.log(self.startprob_)

        # Init
        V[0] = lemissionprob_[:, X[0].item()] + lstartprob_

        # Forward
        for t in np.arange(1 , len(X) ):


            V[t]   = ( lemissionprob_[:, X[t].item()] + 
                np.max( ltransmat_ + V[t-1], axis=1 ) )


            Ptr[t] =  np.argmax( ltransmat_ + V[t-1], axis=1 )

        # Backward

        z_max = np.ones(len(X), dtype=int)

        z_max[-1] = np.argmax(V[-1])

        for t in np.arange(len(X)-2, -1, -1):    
            z_max[t] = Ptr[t+1, z_max[t+1]]
            
        return V, z_max


    def sample_mat(self, p, n=1, k=1000):
        '''
        Sample transition/emission matrix

        Parameters
        ----------
        p : numpy.array
            Each row is the mean of the Dirichlet we sample from.
        n : int
            Number of samples
        k : int
            The larger k the smaller the variance

        '''

        mat = np.zeros( [n, p.shape[0], p.shape[1]] )

        for i in range(n):

            mat[i] = np.apply_along_axis(lambda x: np.random.dirichlet(x, 1), 1, k*p).reshape(p.shape[0],-1)

        return mat

    def sample_rho(self, T, theta_v, n=1):
        '''
        Sample success matrix

        Parameters
        ----------
        
        '''
        
        rho =  np.zeros([n, T, self.n_obs])

        for i in range(n):

            for j in range(self.n_obs):
                rho[i,:,j] = np.random.choice([1,0],p=[theta_v[j], 
                            1-theta_v[j]], size= T)

        return rho


    



        



if __name__ == "__main__":

    priors     = np.array([0.5,0.5])
    transition = np.array([[0.95, 0.05],[0.1, 0.9]])
    emission   = np.array([[1/6, 1/6, 1/6, 1/6, 1/6, 1/6],
                        [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]])

    hmm = HMM(n_components = 2,  n_obs = 6)

    hmm.startprob_ = priors
    hmm.transmat_ = transition
    hmm.emissionprob_ = emission
    
    
    print('---------- Initialize HMM ---------')

    # initialize HMM

    hmm.startprob_ = priors
    hmm.transmat_ = transition
    hmm.emissionprob_ = emission

    print(hmm)

    print('Priors'   , hmm.startprob_)

    print('Transition matrix')
    print(hmm.transmat_)

    print('Emission probabilities')
    print(hmm.emissionprob_)

    X = hmm.sample(5)[0]
    print('Observation X')
    print(X)
    print('Probability of states at time t=4 ---smoothing')
    print(hmm.smoothing(X,4))
    print('Compute alphas--(log probabilities)')
    print(hmm.alpha(X))
    print('Compute betas--(log probabilities)')
    print(hmm.beta(X))
    print('Compute sample matrix ---(high variance)')
    print('----transition matrix----')
    print(hmm.sample_mat(hmm.transmat_, n=1, k=10000))
    print('----emission matrix----')
    print(hmm.sample_mat(hmm.emissionprob_, n=1, k=10000))
    print('----priors vector-----')
    print(hmm.sample_mat(np.array([[0.5,0.5]]) ,n=1, k=10000))
    print('Compute sample rho - Prob. succesful attack')
    rho_matrix = hmm.sample_rho(len(X), [0.5, 0.7, 0.6, 0.2 , 0.5, 0.7])
    print(rho_matrix)
    print('Compute an attack over observations')
    print('---------observations--------------')
    print(X)
    print('---------prob.succesful attack------')
    print(rho_matrix)
    print('---------attack matrix ------------')
    z_matrix = np.zeros((len(X),hmm.n_obs))
    z_matrix[:,0] = 1
    print(z_matrix)
    print('------------compute attacked observations ----------')
    Y = (hmm.attack_X(X, rho_matrix, z_matrix))
    print(Y)
    print('------------ generate Z set ------------------')
    Z_set = hmm.generate_z(5)
    print(Z_set)











