from typing import Optional
from td_algo_enum import TDAlgorithm
from rl_tabular.rl_tabular_base import RLTabularBase
from Process.policy import Policy
from Process.mp_funcs import get_rv_gen_func_single
from Process.mdp_rep_for_rl_tabular import MDPRepForRLTabular
from Process.mp_funcs import get_expected_action_value
from utils.standard_typevars import VFDictType, QFDictType


class TD0(RLTabularBase):

    def __init__(
            self,
            mdp_rep_for_rl: MDPRepForRLTabular,
            exploring_start: bool,
            algorithm: TDAlgorithm,
            softmax: bool,
            epsilon: float,
            epsilon_half_life: float,
            learning_rate: float,
            learning_rate_decay: float,
            num_episodes: int,
            max_steps: int
    ) -> None:

        super().__init__(
            mdp_rep_for_rl=mdp_rep_for_rl,
            exploring_start=exploring_start,
            softmax=softmax,
            epsilon=epsilon,
            epsilon_half_life=epsilon_half_life,
            num_episodes=num_episodes,
            max_steps=max_steps
        )
        self.algorithm: TDAlgorithm = algorithm
        self.learning_rate: float = learning_rate
        self.learning_rate_decay: Optional[float] = learning_rate_decay

    def get_value_func_dict(self, pol: Policy) -> VFDictType:
        sa_dict = self.mdp_rep.state_action_dict
        vf_dict = {s: 0.0 for s in sa_dict.keys()}
        act_gen_dict = {s: get_rv_gen_func_single(pol.get_state_probabilities(s))
                        for s in sa_dict.keys()}
        episodes = 0
        updates = 0

        while episodes < self.num_episodes:
            state = self.mdp_rep.init_state_gen()
            steps = 0
            terminate = False

            while not terminate:
                action = act_gen_dict[state]()
                next_state, reward = \
                    self.mdp_rep.state_reward_gen_dict[state][action]()
                vf_dict[state] += self.learning_rate *\
                    (updates / self.learning_rate_decay + 1) ** -0.5 *\
                    (reward + self.mdp_rep.gamma * vf_dict[next_state] -
                     vf_dict[state])
                updates += 1
                steps += 1
                terminate = steps >= self.max_steps or \
                    state in self.mdp_rep.terminal_states
                state = next_state

            episodes += 1

        return vf_dict

    def get_qv_func_dict(self, pol: Optional[Policy]) -> QFDictType:
        control = pol is None
        this_pol = pol if pol is not None else self.get_init_policy()
        sa_dict = self.mdp_rep.state_action_dict
        qf_dict = {s: {a: 0.0 for a in v} for s, v in sa_dict.items()}
        episodes = 0
        updates = 0

        while episodes < self.num_episodes:
            if self.exploring_start:
                state, action = self.mdp_rep.init_state_action_gen()
            else:
                state = self.mdp_rep.init_state_gen()
                action = get_rv_gen_func_single(
                    this_pol.get_state_probabilities(state)
                )()
            steps = 0
            terminate = False

            while not terminate:
                next_state, reward = \
                    self.mdp_rep.state_reward_gen_dict[state][action]()
                next_action = get_rv_gen_func_single(
                    this_pol.get_state_probabilities(next_state)
                )()
                if self.algorithm == TDAlgorithm.QLearning and control:
                    next_qv = max(qf_dict[next_state][a] for a in
                                  qf_dict[next_state])
                elif self.algorithm == TDAlgorithm.ExpectedSARSA and control:
                    next_qv = get_expected_action_value(
                        qf_dict[next_state],
                        self.softmax,
                        self.epsilon_func(episodes)
                    )
                else:
                    next_qv = qf_dict[next_state][next_action]

                qf_dict[state][action] += self.learning_rate *\
                    (updates / self.learning_rate_decay + 1) ** -0.5 *\
                    (reward + self.mdp_rep.gamma * next_qv -
                     qf_dict[state][action])
                updates += 1
                if control:
                    if self.softmax:
                        this_pol.edit_state_action_to_softmax(
                            state,
                            qf_dict[state]
                        )
                    else:
                        this_pol.edit_state_action_to_epsilon_greedy(
                            state,
                            qf_dict[state],
                            self.epsilon_func(episodes)
                        )
                steps += 1
                terminate = steps >= self.max_steps or \
                    state in self.mdp_rep.terminal_states
                state = next_state
                action = next_action

            episodes += 1

        return qf_dic
