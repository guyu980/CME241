from typing import Mapping, Generic, Dict
from utils.generic_typevars import S, A

class Policy(Generic[S, A]):
    def __init__(self, data: Dict[S, Mapping[A, float]]):
        self.data = data
    
    def get_state_probabilities(self, state: S) -> Mapping[A, float]:
        return self.data[state]
    
    def get_state_action_probability(self, state: S, action: A) -> float:
        return self.get_state_probabilities(state).get(action, 0.)
    
    def __repr__(self):
        return self.data.__repr__()