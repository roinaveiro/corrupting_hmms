import time
class boolean_optim_mcts(object):

    def __init__(self, node):
        """
        MonteCarloTreeSearchNode
        Parameters
        ----------
        node : nodes.mcts_node
        """
        self.root = node
        self.best_state = 0 #Something better...
        self.best_result = 0.

    def iterate(self, simulations_number=None, total_simulation_seconds=None):
        """
        Parameters
        ----------
        simulations_number : int
            number of simulations performed to get the best action
        total_simulation_seconds : float
            Amount of time the algorithm has to run. Specified in seconds
        Returns
        -------
        """

        if simulations_number is None :
            assert(total_simulation_seconds is not None)
            end_time = time.time() + total_simulation_seconds
            while time.time() < end_time:
                v = self._tree_policy()
                reached_state, reward = v.rollout()

                if reward >= self.best_result:
                    self.best_state = reached_state
                    self.best_result = reward

                v.backpropagate(reward)
        else :
            for _ in range(0, simulations_number):            
                v = self._tree_policy()
                reached_state, reward = v.rollout()

                if reward >= self.best_result:
                    self.best_state  = reached_state
                    self.best_result = reward

                v.backpropagate(reward)
        # to select best child go for exploitation only
        return self.best_state

    def _tree_policy(self):
        """
        selects node to run rollout/playout for
        Returns
        -------
        """
        current_node = self.root
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node