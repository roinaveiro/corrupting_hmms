import numpy as np
from hmm_utils import HMM
from params import *


class ss_attacker():

    def __init__(self):

        self.w1 = w1
        self.w2 = w2


    def expected_utility(A):
        pass




if __name__ == "__main__":

    priors     = np.array([0.5,0.5])
    transition = np.array([[0.95, 0.05],[0.1, 0.9]])
    emission   = np.array([[1/6, 1/6, 1/6, 1/6, 1/6, 1/6],
                        [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]])

    m = HMM(n_components=2)

    m.startprob_ = priors
    m.transmat_ = transition
    m.emissionprob_ = emission

    print(m.transmat_)
        

    




