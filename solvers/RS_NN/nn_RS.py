import numpy as np
from solvers.RS_NN.game import boolean_optim_state
from solvers.RS_NN.nodes import boolean_optim_mcts_node
from solvers.RS_NN.mcts import boolean_optim_mcts

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from solvers.RS_NN.models import FeedForwardNN





class nn_RS():
    '''
    
    '''


    def __init__(self, attacker, RS_iters = 1000, mcts_iters=1000, lr=0.05):


        self.attacker = attacker

        self.model  = FeedForwardNN(self.attacker.T, 1)
        self.optim  = Adam(self.model.parameters(), lr=lr)
        self.loss   = nn.MSELoss()

        self.possible_values = np.arange(self.attacker.n_obs)
        self.initial_conf = -1*np.ones(self.T)


        self.Z_set = self.attacker.generate_attacks()
        self.mcts_iters = mcts_iters

        self.RS_iters = RS_iters


    def update(self, output, nn_output):
    
        ll = self.loss(output, nn_output)
        self.optim.zero_grad()
        ll.backward()
        self.optim.step()

    def policy(self, eps=0.1):

        # Search best action

        if np.random.uniform() < eps:

            init_state = boolean_optim_state(self.initial_conf, 
            self.possible_values, 
            lambda x: self.model.forward(x).detach().numpy().item())

            root_node  = boolean_optim_mcts_node(init_state)
            search     = boolean_optim_mcts(root_node)

            best_action = search.iterate(simulations_number=self.mcts_iters)
            return self.ohe(best_action)     
    
        else:

            return self.Z_set[ np.random.choice(self.Z_set.shape[0]) ]
        
    def evaluate(self, z):
        return self.attacker.expected_utility(z, N=1)

    def iterate(self):
        
        for i in range(self.RS_iters):

            # Decide
            # Evaluate
            # Update

        

    def ohe(self, x):
        return np.diag(np.ones(self.attacker.n_obs))[x]




