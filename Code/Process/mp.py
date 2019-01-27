import numpy as np
import matplotlib.pyplot as plt
from typing import Mapping
from utils.generic_typevars import S
from utils.utils import is_equal

class MP:
    def __init__(self, transitions: Mapping[S, Mapping[S, float]]):
        if self.check_mp(transitions):
            self.transitions = transitions
            self.states = list(self.transitions.keys())
            self.transition_matrix = self.generate_matrix(self.transitions)
        else:
            raise ValueError
    
    def check_mp(self, transitions) -> bool:
        states = set(transitions.keys())
        # Check the successor states is the subset of current states
        b1 = set().union(*list(transitions.values())).issubset(states)
        # Check the probabilities are positive
        b2 = all(all(p >= 0 for p in transitions[state].values()) for state in transitions)
        # Check the sum of probabilities for each state equals to 1
        b3 = all(is_equal(sum(transitions[state].values()), 1.) for state in transitions)        
        return b1 and b2 and b3
    
    def generate_matrix(self, dict_obj) -> np.array:
        matrix = np.zeros((len(self.states), len(self.states)))
        
        for i in range(len(self.states)):
            for j in range(len(self.states)):
                if self.states[j] in dict_obj[self.states[i]]:
                    matrix[i, j] = dict_obj[self.states[i]][self.states[j]]
        return matrix
    
    def simulation(self, iters: int) -> None:
        p0 = np.random.rand(len(self.states))
        p0 /= sum(p0)
        simulation = np.zeros((iters, len(self.states)))
        
        for i in range(iters):
            p0 = p0.dot(self.transition_matrix)
            simulation[i, :] = p0
        
        for i in range(len(self.states)):
            plt.plot(range(iters), simulation[:, i])
    


if __name__ == '__main__':
    transitions = {
        1: {1: 0.2, 2: 0.3, 3: 0.1, 4: 0.1, 5: 0.3},
        2: {1: 0.15, 2: 0.35, 3: 0.25, 5: 0.25},
        3: {1: 0.55, 2: 0.12, 4: 0.33},
        4: {1: 0.2, 2: 0.5, 3: 0.2, 5: 0.1},
        5: {5: 1}
    }
    
    mp = MP(transitions)
    print('States:', mp.states, '\n')
    print('Transition Matrix:\n', mp.transition_matrix, '\n')
    print('Probability Distribution Simulation:')
    mp.simulation(40)