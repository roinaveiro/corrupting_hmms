import numpy as np
import time

from solvers.nn_RS.game import boolean_optim_state
from solvers.nn_RS.nodes import boolean_optim_mcts_node
from solvers.nn_RS.mcts import boolean_optim_mcts

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from solvers.nn_RS.models import FeedForwardNN


class nn_RS():
    '''
    '''


    def __init__(self, attacker, RS_iters=1000, mcts_iters=100, eps=0.1, lr=0.05, verbose=True):


        self.attacker = attacker

        self.model  = FeedForwardNN(self.attacker.T*self.attacker.n_obs, 1)
        self.optim  = Adam(self.model.parameters(), lr=lr)
        self.loss   = nn.MSELoss()

        self.possible_values = np.arange(self.attacker.n_obs)
        self.initial_conf = -1*np.ones(self.attacker.T)


        self.Z_set = self.attacker.generate_attacks()
        self.mcts_iters = mcts_iters

        self.z_best = 0.0 #Something better?
        self.value_best = 0.0

        self.RS_iters = RS_iters
        self.eps = eps

        self.verbose = verbose


    def update(self, output, action):

        nn_output = self.model.forward(action.flatten())
        ll = self.loss(torch.tensor(output, dtype=torch.float), nn_output)
        self.optim.zero_grad()
        ll.backward()
        self.optim.step()

    def policy(self, eps=0.1, iters=None):

        # Search best action

        if np.random.uniform() < eps:

            return self.Z_set[ np.random.choice(self.Z_set.shape[0]) ]
    
        else:

            F_eval = (lambda x: 
                      self.model.forward(self.ohe(x).flatten()).detach().numpy().item())

            init_state = boolean_optim_state(self.initial_conf, 
            self.possible_values, F_eval)

            root_node  = boolean_optim_mcts_node(init_state)
            search     = boolean_optim_mcts(root_node)

            if iters is None:
                best_action = search.iterate(simulations_number=self.mcts_iters)

            else:
                best_action = search.iterate(simulations_number=iters)

            return self.ohe(best_action)  

            
        
    def evaluate(self, z):
        # Increase N for less noisy estimates
        return self.attacker.expected_utility(z, N=10)

    def iterate(self, simulation_seconds=None):

        if simulation_seconds is None :
        
            for i in range(self.RS_iters):

                if self.verbose:
                    if i%50 == 0:
                        print("Percentage completed:", 
                        np.round(100*i/self.RS_iters, 2)  )

                        print("Current action: ")
                        print(self.policy(eps=0.0))

                action = self.policy(self.eps)
                value  = self.evaluate(action)

                if value >= self.value_best:
                    self.z_best = action

                self.update( value, action )

        else:

            end_time = time.time() + simulation_seconds
            while time.time() < end_time:

                action = self.policy(self.eps)
                value  = self.evaluate(action)

                if value >= self.value_best:
                    self.z_best = action
                    self.value_best = value

                self.update( value, action )

        
        z_star = self.policy(eps=0.0, iters=10000)
        solution_quality = self.attacker.expected_utility(z_star, N=10000)
        solution_quality_best = self.attacker.expected_utility(self.z_best, N=10000)

        if solution_quality >= solution_quality_best:
            return z_star, solution_quality
        else:
            return self.z_best, solution_quality_best 


        

    def ohe(self, x):
        return np.diag(np.ones(self.attacker.n_obs))[ x.astype(int) ]




