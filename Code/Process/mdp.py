import numpy as np
from mp import MP
from mrp import MRP
from policy import Policy
from typing import Mapping, List
from utils.generic_typevars import S, A
from utils.utils import sum_dicts 

class MDP(MP):
    def __init__(self, transitions: Mapping[S, Mapping[A, Mapping[S, float]]], \
                 rewards: Mapping[S, Mapping[A, Mapping[S, float]]], gamma: float) -> None:
        self.transitions = transitions
        self.states = self.get_all_states()
        self.actions = self.get_all_actions()
        self.rewards = rewards
        self.gamma = gamma
        
    def get_all_actions(self) -> List:
        return list(set().union(*list(self.transitions.values())))
    
    def get_mrp(self, pol: Policy) -> MRP:
        transitions = {s: sum_dicts([{s1: p * v2 for s1, v2 in v[a].items()}
                        for a, p in pol.data[s].items()])
                        for s, v in self.transitions.items()}
        rewards = {s: sum(p * v[a] for a, p in pol.data[s].items())
                    for s, v in self.rewards.items()}
        return MRP(transitions, rewards, self.gamma)
        
    def get_state_value_func(self, pol: Policy) -> Mapping[S, float]:
        mrp = self.get_mrp(pol)
        value_func = mrp.get_value_func()
        return {mrp.states[i]: value_func[i] for i in range(len(mrp.states))}
    
    def get_action_value_func(self, pol: Policy) -> Mapping[S, Mapping[A, float]]:
        value_func = self.get_state_value_func(pol)
        return {s:  {a: r + self.gamma * sum(p * value_func[s1]
                for s1, p in self.transitions[s][a].items())
                for a, r in v.items()}
                for s, v in self.rewards.items()} 
    


if __name__ == '__main__':
    transitions = {
        1: {
            'a': {1: 0.3, 2: 0.6, 3: 0.1},
            'b': {2: 0.3, 3: 0.7},
            'c': {1: 0.2, 2: 0.4, 3: 0.4}
        },
        2: {
            'a': {1: 0.3, 2: 0.6, 3: 0.1},
            'c': {1: 0.2, 2: 0.4, 3: 0.4}
        },
        3: {
            'b': {3: 1.0}
        }
    }
    rewards = {
        1: {'a': 5, 'b': 4, 'c': -6},
        2: {'a': 5, 'c': -6},
        3: {'b': 0}
    }
    policy_data = {
        1: {'a': 0.4, 'b': 0.6},
        2: {'a': 0.7, 'c': 0.3},
        3: {'b': 1.0}
    }
    mdp = MDP(transitions, rewards, 0.5)
    pol = Policy(policy_data)
    print('States:', mdp.states, '\n')
    print('Actions', mdp.actions, '\n')
    print('Transition Matrix of MDP:\n', mdp.transitions, '\n')
    print('Reward Function of MDP:\n', mdp.rewards, '\n')
    print('Policy:\n', pol, '\n')
    mrp = mdp.get_mrp(pol)
    print('Transition Matrix of MRP:\n', mrp.transition_matrix, '\n')
    print('Reward Function of MRP:\n', mrp.reward_func, '\n')
