import numpy as np


class boolean_optim_move():
    def __init__(self, idx, value):
        self.idx = idx
        self.value = value



class boolean_optim_state():
    '''
    -1 means that variable is not instantiated
    '''


    def __init__(self, state, possible_values, evaluate_function):

        self.state = state
        self.possible_values = possible_values
        self.evaluate_function = evaluate_function


    @property
    def game_result(self):
        # check if game is over
        if np.all( self.state != -1 ):
            return self.evaluate()

        # # Trick to introduce cost constraints!
        # elif np.sum(self.state >= 1) == self.max_controls:
        #     self.state[self.state == -1] = 0.0
        #     return self.evaluate

        else:
            # if not over - no result
            return None


    def is_game_over(self):
        return self.game_result is not None

    def evaluate(self):
        if not self.is_game_over:
            raise ValueError(
                    "Game is not over!"
                )

        # result = np.sum(self.state)
        result = self.evaluate_function(self.state)#np.dot(self.state, w)
        return result

    def is_move_legal(self, move):
        #check if move is legal
        return True

    def move(self, move):
        if not self.is_move_legal(move):
            raise ValueError(
                "Move is not legal!"
            )
        new_state = np.copy(self.state)
        new_state[move.idx] = move.value

      
        return boolean_optim_state(new_state, 
                self.possible_values, self.evaluate_function)           

    def get_legal_actions(self):
        indices = np.where(self.state == -1)[0]
        return [
            boolean_optim_move(idx, value)
            for idx in indices 
            for value in self.possible_values
        ]
