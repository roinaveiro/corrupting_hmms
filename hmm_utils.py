import numpy as np
from hmmlearn import hmm


class HMM(hmm.MultinomialHMM):
    
    def __init__(self, n_components):
        
        super().__init__(n_components)
        
  
    def smoothing(self, X, t):
        return self.predict_proba(X)[t]
    
    def alpha(self, X):
        pass
    
    def beta(self, X):
        pass
    
    def nu(self, X):
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

    
        

    




