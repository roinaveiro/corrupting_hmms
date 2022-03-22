import numpy as np
import time
from scipy.special import softmax
from solvers.simulated_annealing import return_mat


class aps_gibbs():
    '''
    '''

    def __init__(self, attacker, max_iter=1000, burnin=0.1, verbose=True):


        self.attacker         = attacker
        self.burnin           = burnin
        self.max_iter         = max_iter
        self.verbose          = verbose

        z_init = self.attacker.sample_attack()
        self.z_samples = z_init[np.newaxis]

        # self.Z_set = self.attacker.generate_attacks()          
        

        

    def update_probs_metropolis(self, H, hmms_old, value_old, z):
    
        hmms = []
        value = 0
        
        for h in range(H):
            
            hmm_sample = self.attacker.sample_hmm()
            value += np.log( self.attacker.utility(z, hmm_sample) )
            hmms.append(hmm_sample)
            
        value /= H
        prob =  np.exp(H*value - H*value_old)
        
        if np.random.uniform() < prob:
            
            #print("Accept!")
            return hmms, value
        
        else:
            
            return hmms_old, value_old

    def update_z(self, hmms, z, idx):
    
        Z_candidates = np.apply_along_axis( 
            lambda x: return_mat(x, z, idx), 1, np.eye(self.attacker.n_obs))

        energies = np.zeros( len(Z_candidates) )

        for i, z_new in enumerate(Z_candidates):
            
            for hmm_sample in hmms:
                energies[i] += np.log( self.attacker.utility(z_new, hmm_sample) )
                
        p = softmax(energies)
        #print(p)

        candidate_idx = np.random.choice( 
            np.arange(len(Z_candidates) ), p=p )
        
        return Z_candidates[candidate_idx]


    def initialize(self):

        ##
        hmms = []
        value = 0

        # z_init = self.Z_set[ np.random.choice(self.Z_set.shape[0]) ]
        z_init = self.z_samples[-1] #self.attacker.sample_attack()
        init_temp = self.cooling_sch(t=1)

        for temp in range( init_temp ):

            hmm_sample = self.attacker.sample_hmm()
            value += np.log( self.attacker.utility(z_init, hmm_sample) )
            hmms.append(hmm_sample)


        value /= init_temp


        return hmms, value, z_init

    def update_all(self, i, temp, z_init, hmms, value):

        # Update every z
        for idx in range(self.attacker.T):
            z_init = self.update_z(hmms, z_init, idx)

        self.z_samples = np.append(self.z_samples, 
                        z_init[np.newaxis], axis=0)

        # Update z at random
        # idx = np.random.choice(self.attacker.T)
        # z_init = self.update_z(hmms, z_init, idx)
        # self.z_samples.append(z_init)

        # Update value with new z
        value = 0
        for n in range(len(hmms)):
            value += np.log( 
                self.attacker.utility(z_init, hmms[n]) )
        value /= len(hmms)

        hmms, value = self.update_probs_metropolis(temp, 
                                            hmms, value, z_init)

        return z_init, hmms, value



    def iterate(self, simulation_seconds=None ):

        if simulation_seconds is None :

            hmms, value, z_init = self.initialize()

            for i in range(1, self.max_iter):

                current_temp = self.cooling_sch(i)
            
                if self.verbose:
                    
                    if i%10 == 0:
                        print("Percentage completed:", 
                        np.round( 100*i/self.max_iter, 2) )
                        print("Current state", z_init)

                z_init, hmms, value = self.update_all(i, current_temp, z_init, hmms, value)
                    
            z_star = self.extract_solution()
            return z_star, self.z_samples

        else: 

            end_time = time.time() + simulation_seconds

            hmms, value, z_init = self.initialize()

            assert(time.time() < end_time)
            i = 1

            while time.time() < end_time:

                current_temp = self.cooling_sch(i)
                z_init, hmms, value = self.update_all(i, current_temp, z_init, hmms, value)
                i+=1
      
            z_star, quality = self.extract_solution()
            return z_star, quality

    def cooling_sch(self, t, l=5):

        temp = np.int( np.exp(l*t/self.attacker.T) )
        return temp

    
    def get_mode(self, samples):
        vals, counts = np.unique( samples, return_counts=True, axis=0 )
        index = np.argmax(counts)
        return vals[index]

    def extract_solution(self):
        self.z_samples = np.array(self.z_samples)

        z_star = np.zeros_like(self.z_samples[0])
        burnin_end = np.int(self.burnin * len(self.z_samples))
        
        for j in range(z_star.shape[0]):
            z_star[j] = self.get_mode(self.z_samples[  burnin_end: , j ])

        solution_quality = self.attacker.expected_utility(z_star, N=10000)
        return z_star, solution_quality



                    
        




