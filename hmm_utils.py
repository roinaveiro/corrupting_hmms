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
        
        super().__init__(n_components)
        self.n_obs = n_obs
        
  
    def smoothing(self, X, t):
        return self.predict_proba(X)[t]
    
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
        ltransmat_     = np.log(self.transmat_)
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
            The larget k the smaller the variance

        '''

        mat = np.zeros( [n, p.shape[0], p.shape[1]] )

        for i in range(n):

            mat[i] = np.apply_along_axis(lambda x: 
                np.random.dirichlet(x, 1), 1, k*p).reshape(p.shape[0],-1)

        return mat

    def sample_rho(self, t, theta):
        '''
        Sample success matrix

        Parameters
        ----------

        '''


    def attack_X(self, X, rho, z):
        '''
        Given attack matrix z and success matrix rho, transforms
        X into attacked version
        
        Parameters
        ----------
    

        '''
        pass

    def generate_z(self, t):
        pass

    



        



if __name__ == "__main__":

    priors     = np.array([0.5,0.5])
    transition = np.array([[0.95, 0.05],[0.1, 0.9]])
    emission   = np.array([[1/6, 1/6, 1/6, 1/6, 1/6, 1/6],
                        [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]])

    m = HMM(2)

    m.startprob_ = priors
    m.transmat_ = transition
    m.emissionprob_ = emission

    X = np.atleast_2d([5, 0, 5, 2, 3, 5, 5, 5, 5, 2, 3, 5, 3, 5,
     0, 5, 4, 3, 2, 1, 5, 5, 5, 0, 5, 2, 5, 5, 5, 5, 1, 5, 5, 4,
      5, 2, 0, 3, 1, 5, 3, 1, 3, 3, 2, 1, 1, 4, 5, 4, 4, 1, 0, 0,
       2, 4, 1, 4, 3, 0, 5, 5, 1, 5, 5, 0, 2, 1, 5, 4, 5, 5, 5, 1,
        5, 2, 2, 2, 4, 1, 0, 2, 4, 0, 2, 5, 3, 1, 4, 3, 0, 1, 4, 4,
         5, 3, 2, 5, 2, 1, 1, 2, 4, 2, 2, 4, 1, 4, 5, 5, 3, 5, 4, 5,
          4, 5, 4, 0, 0, 5, 3, 0, 0, 5, 0, 5, 5, 2, 1, 2, 0, 2, 2, 1,
           5, 1, 5, 1, 0, 1, 3, 3, 1, 3, 3, 4, 4, 3, 5, 2, 0, 0, 5, 5,
            5, 5, 5, 1, 5, 4, 2, 0, 2, 0, 2, 3, 1, 5, 0, 3, 5, 2, 5, 5]).T


    print(m.decode(X))

    
        

    




