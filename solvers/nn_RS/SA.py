import numpy as np
import time
from scipy.special import softmax
from solvers.simulated_annealing import return_mat


class simulated_annealing():
    '''
    '''

    def __init__(self, attacker, F_eval, n_iter=50):


        self.attacker         = attacker
        self.F_eval           = F_eval  
        self.n_iter           = n_iter
        self.z_star           = self.attacker.sample_attack()
        self.quality_star     = self.F_eval(self.z_star)

    def s(self, t, den=5, l=5):
        return np.exp(-l*t/den) # Watch out!


    def generate_candidate(self, t, z_init):

        idx = np.random.choice(self.attacker.T)

        Z_candidates = np.apply_along_axis( 
            lambda x: return_mat(x, z_init, idx), 1, np.eye(self.attacker.n_obs))

        energies = np.zeros( len(Z_candidates) )

        for i, z in enumerate(Z_candidates):
            energies[i] = self.F_eval(z) / self.s(t, den=self.attacker.T)

        p = softmax(energies)

        candidate_idx = np.random.choice( 
            np.arange(len(Z_candidates) ), p=p )
        
        return Z_candidates[candidate_idx]


    def iterate(self):


        z_init = self.z_star

        for t in np.arange(self.n_iter):

            z_init = self.generate_candidate(t, z_init)
            quality_init = self.F_eval(z_init)

            if quality_init >= self.quality_star:
                self.z_star = z_init


        return self.z_star

       



