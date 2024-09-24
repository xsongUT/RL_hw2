import numpy as np
from typing import override
from interfaces.policy import Policy

class Policy_DeterministicGreedy(Policy):
    def __init__(self, Q: np.ndarray[np.float64]):
        """
        Parameters:
        - Q (np.ndarray): Q function; numpy array shape of [nS,nA]
        each row is for a state, so here we have nS rows
        each column is for an action, so we have nA columns
        each cell is a Q value 
        """
        self.Q = Q

    @override
    def action(self, state: int) -> int:
        """
        Chooses the action that maximizes the Q function for the given state.

        Parameters:
            - state (int): state index

        Returns:
            - int: index of the action to take
        """

        ### TODO: Implement the action method ###
        return int(np.argmax(self.Q[state]))  
    # Return the index of ction with the highest Q value for a state(each row in self.Q is a state)


    @override
    def action_prob(self, state: int, action: int) -> float:
        """
        Returns the probability of taking the action if we are in the given state.

        Since this is a greedy policy, this will be a 1 or a 0.

        Parameters:
            - state (int): state index
            - action (int): action index

        Returns:
            - float: the probability of taking the action in the given state
        """

        ### TODO: Implement the action_prob method ###
        return 1.0 if action == int(np.argmax(self.Q[state])) else 0.0