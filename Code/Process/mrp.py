import numpy as np
from mp import MP
from typing import Mapping
from utils.generic_typevars import S

class MRP(MP):
    def __init__(self, transitions: Mapping[S, Mapping[S, float]], \
                 rewards: Mapping[S, float], gamma: float):
        if self.check_mp(transitions):
            self.transitions = transitions
            self.states = self.get_all_states()
            self.rewards = rewards
            self.reward_func = self.get_reward_func()
            self.transition_matrix = self.generate_matrix(self.transitions)
            self.gamma = gamma
        else:
            raise ValueError
    
    def get_reward_func(self):
        return list(self.rewards.values())
    
    def get_value_func(self) -> np.array:
        return np.linalg.inv(np.eye(len(self.states)) - self.gamma * self.transition_matrix \
                             ).dot(self.reward_func)



if __name__ == '__main__':
    transitions = {
        1: {1: 0.2, 2: 0.3, 3: 0.1, 4: 0.1, 5: 0.3},
        2: {1: 0.15, 2: 0.35, 3: 0.25, 5: 0.25},
        3: {1: 0.55, 2: 0.12, 4: 0.33},
        4: {1: 0.2, 2: 0.5, 3: 0.2, 5: 0.1},
        5: {5: 1}
    }
    rewards = {
        1: 8.4, 2: 8.4, 3: 5, 4: 6.8, 5: 2
    }
    mrp = MRP(transitions, rewards, 0.5)
    print('States:', mrp.states, '\n')
    print('Transition Matrix:\n', mrp.transition_matrix, '\n')
    print('Reward Function (Wiki):\n', mrp.reward_func, '\n')
    print('Value Function:\n', mrp.get_value_func(), '\n')
