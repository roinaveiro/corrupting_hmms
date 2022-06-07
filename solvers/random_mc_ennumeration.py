import numpy as np
import time
from scipy.special import softmax
from solvers.simulated_annealing import return_mat


class random_mc_ennumeration():
    '''
    '''

    def __init__(self, attacker, N=100, verbose=True):


        self.attacker    = attacker
        self.verbose     = verbose
        self.N           = N

   

    def initialize(self):

        self.current_z = self.attacker.sample_attack()
        self.current_quality = self.attacker.expected_utility(self.current_z, N=self.N)




    def iterate(self, simulation_seconds=None, Q=1000 ):

        if simulation_seconds is None :

            self.initialize()

            for i in range(Q):

                candidate_z = self.attacker.sample_attack()
                candidate_quality = self.attacker.expected_utility(candidate_z, N=self.N)

                if candidate_quality >= self.current_quality:

                    self.current_z = candidate_z
                    self.current_quality = candidate_quality
       
            final_quality = self.attacker.expected_utility(self.current_z, N=10000)
            return self.current_z, final_quality

        else: 

            end_time = time.time() + simulation_seconds

            self.initialize()

            assert(time.time() < end_time)

            while time.time() < end_time:

                candidate_z = self.attacker.sample_attack()
                candidate_quality = self.attacker.expected_utility(candidate_z, N=self.N)

                if candidate_quality >= self.current_quality:

                    self.current_z = candidate_z
                    self.current_quality = candidate_quality

      
            final_quality = self.attacker.expected_utility(self.current_z, N=10000)
            return self.current_z, final_quality                 
        




