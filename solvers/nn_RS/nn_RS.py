import numpy as np
import time

from solvers.nn_RS.game import boolean_optim_state
from solvers.nn_RS.nodes import boolean_optim_mcts_node
from solvers.nn_RS.mcts import boolean_optim_mcts

from solvers.nn_RS.SA import simulated_annealing

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from solvers.nn_RS.models import FeedForwardNN


class nn_RS():
    '''
    '''


    def __init__(self, attacker, flag, RS_iters=1000, mcts_iters=100, sa_iters=100, eps=0.1, lr=0.05, verbose=True):


        self.attacker = attacker

        self.model  = FeedForwardNN(self.attacker.T*self.attacker.n_obs, 1)
        self.optim  = Adam(self.model.parameters(), lr=lr)
        # lr is adjustable
        # We could add regularization
        # self.optim  = Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)

        self.loss   = nn.MSELoss()

        self.possible_values = np.arange(self.attacker.n_obs)
        self.initial_conf = -1*np.ones(self.attacker.T)


        # self.Z_set = self.attacker.generate_attacks()
        self.mcts_iters = mcts_iters
        self.sa_iters   = sa_iters

        self.z_best = 0.0 #Something better?
        self.value_best = -1000000

        self.RS_iters = RS_iters
        self.eps = eps

        self.verbose = verbose
        self.flag = flag

        self.solution_quality_best = -1000000


    def update(self, output, action):

        nn_output = self.model.forward(action.flatten())
        ll = self.loss(torch.tensor(output, dtype=torch.float), nn_output)
        self.optim.zero_grad()
        ll.backward()
        self.optim.step()

    def policy(self, eps=0.1, iters=None):

        # Search best action
        # eps is adjustable

        if np.random.uniform() < eps:
            
            # return self.Z_set[ np.random.choice(self.Z_set.shape[0]) ]
            return self.attacker.sample_attack()
    
        else:

            if self.flag == 'MCTS':

                F_eval = (lambda x: 
                        self.model.forward(self.ohe(x).flatten()).detach().numpy().item())

                init_state = boolean_optim_state(self.initial_conf, 
                self.possible_values, F_eval)

                root_node  = boolean_optim_mcts_node(init_state)
                search     = boolean_optim_mcts(root_node)

                if iters is None:
                    best_action = search.iterate(simulations_number=self.mcts_iters)
                    # self.mcts_iters is adjustable

                else:
                    best_action = search.iterate(simulations_number=iters)


                return self.ohe(best_action)  

            elif self.flag == 'SA':

                F_eval = (lambda x: 
                        self.model.forward(x.flatten()).detach().numpy().item())

                ##
                if iters is None:
                    sa = simulated_annealing(self.attacker, F_eval, self.sa_iters)
                    best_action = sa.iterate()
                   
                else:
                    sa = simulated_annealing(self.attacker, F_eval, iters)
                    best_action = sa.iterate()


                

                return best_action

            
        
    def evaluate(self, z):
        # Increase N for less noisy estimates
        # Adjustable
        return self.attacker.expected_utility(z, N=10)

    def iterate(self, simulation_seconds=None):

        if simulation_seconds is None :
        
            for i in range(self.RS_iters):

                action = self.policy(self.eps)
                value  = self.evaluate(action)

                if self.verbose:
                    if i%50 == 0:
                        print("Percentage completed:", 
                        np.round(100*i/self.RS_iters, 2)  )


                        print("Best value: ")
                        print(self.value_best)

                

                if value >= self.value_best:
                    self.z_best = action
                    self.value_best = value

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


        if self.flag == 'MCTS':
            z_star = self.policy(eps=0.0, iters=10000)

        elif self.flag == 'SA':
            z_star = self.policy(eps=0.0, iters=500)


        solution_quality = self.attacker.expected_utility(z_star, N=10000)
        # print(solution_quality)


        if solution_quality >= self.solution_quality_best:

            self.z_best = z_star
            self.solution_quality_best = solution_quality

        return self.z_best, self.solution_quality_best 


        

    def ohe(self, x):
        return np.diag(np.ones(self.attacker.n_obs))[ x.astype(int) ]




