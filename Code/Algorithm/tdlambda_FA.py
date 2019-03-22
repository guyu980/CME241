from typing import Mapping, Optional
from td_algo_enum import TDAlgorithm
from rl_func_approx.rl_func_approx_base import RLFuncApproxBase
from func_approx_spec import FuncApproxSpec
from Process.mdp_rep_for_rl_fa import MDPRepForRLFA
from Process.mp_funcs import get_rv_gen_func_single
from helper_funcs import get_soft_policy_func_from_qf
from Process.mp_funcs import get_expected_action_value
import numpy as np
from utils.generic_typevars import S, A
from utils.standard_typevars import VFType, QFType, PolicyActDictType


class TDLambda(RLFuncApproxBase):

    def __init__(
        self,
        mdp_rep_for_rl: MDPRepForRLFA,
        exploring_start: bool,
        algorithm: TDAlgorithm,
        softmax: bool,
        epsilon: float,
        epsilon_half_life: float,
        lambd: float,
        num_episodes: int,
        batch_size: int,
        max_steps: int,
        fa_spec: FuncApproxSpec,
        offline: bool
    ) -> None:

        super().__init__(
            mdp_rep_for_rl=mdp_rep_for_rl,
            exploring_start=exploring_start,
            softmax=softmax,
            epsilon=epsilon,
            epsilon_half_life=epsilon_half_life,
            num_episodes=num_episodes,
            max_steps=max_steps,
            fa_spec=fa_spec
        )
        self.algorithm: TDAlgorithm = algorithm
        self.gamma_lambda: float = self.mdp_rep.gamma * lambd
        self.batch_size: int = batch_size
        self.offline: bool = offline

    def get_value_func_fa(self, polf: PolicyActDictType) -> VFType:
        episodes = 0

        while episodes < self.num_episodes:
            et = [np.zeros_like(p) for p in self.vf_fa.params]
            state = self.mdp_rep.init_state_gen()
            steps = 0
            terminate = False

            states = []
            targets = []
            while not terminate:
                action = get_rv_gen_func_single(polf(state))()
                next_state, reward =\
                    self.mdp_rep.state_reward_gen_func(state, action)
                target = reward + self.mdp_rep.gamma *\
                    self.vf_fa.get_func_eval(next_state)
                delta = target - self.vf_fa.get_func_eval(state)
                if self.offline:
                    states.append(state)
                    targets.append(target)
                else:
                    et = [et[i] * self.gamma_lambda + g for i, g in
                          enumerate(self.vf_fa.get_sum_objective_gradient(
                              [state],
                              np.ones(1)
                          )
                          )]
                    self.vf_fa.update_params_from_gradient(
                        [-e * delta for e in et]
                    )
                steps += 1
                terminate = steps >= self.max_steps or\
                    self.mdp_rep.terminal_state_func(state)
                state = next_state

            if self.offline:
                avg_grad = [g / len(states) for g in
                            self.vf_fa.get_el_tr_sum_loss_gradient(
                                states,
                                targets,
                                self.gamma_lambda
                            )]
                self.vf_fa.update_params_from_gradient(avg_grad)
            episodes += 1

        return self.vf_fa.get_func_eval

    # noinspection PyShadowingNames
    def get_qv_func_fa(self, polf: Optional[PolicyActDictType]) -> QFType:
        control = polf is None
        this_polf = polf if polf is not None else self.get_init_policy_func()
        episodes = 0

        while episodes < self.num_episodes:
            et = [np.zeros_like(p) for p in self.qvf_fa.params]
            if self.exploring_start:
                state, action = self.mdp_rep.init_state_action_gen()
            else:
                state = self.mdp_rep.init_state_gen()
                action = get_rv_gen_func_single(this_polf(state))()
            steps = 0
            terminate = False

            states_actions = []
            targets = []
            while not terminate:
                next_state, reward = \
                    self.mdp_rep.state_reward_gen_func(state, action)
                next_action = get_rv_gen_func_single(this_polf(next_state))()
                if self.algorithm == TDAlgorithm.QLearning and control:
                    next_qv = max(self.qvf_fa.get_func_eval((next_state, a)) for a in
                                  self.state_action_func(next_state))
                elif self.algorithm == TDAlgorithm.ExpectedSARSA and control:
                    next_qv = get_expected_action_value(
                        {a: self.qvf_fa.get_func_eval((next_state, a)) for a in
                         self.state_action_func(next_state)},
                        self.softmax,
                        self.epsilon_func(episodes)
                    )
                else:
                    next_qv = self.qvf_fa.get_func_eval((next_state, next_action))

                target = reward + self.mdp_rep.gamma * next_qv
                delta = target - self.qvf_fa.get_func_eval((state, action))

                if self.offline:
                    states_actions.append((state, action))
                    targets.append(target)
                else:
                    et = [et[i] * self.gamma_lambda + g for i, g in
                          enumerate(self.qvf_fa.get_sum_objective_gradient(
                              [(state, action)],
                              np.ones(1)
                          )
                          )]
                    self.qvf_fa.update_params_from_gradient(
                        [-e * delta for e in et]
                    )
                if control and self.batch_size == 0:
                    this_polf = get_soft_policy_func_from_qf(
                        self.qvf_fa.get_func_eval,
                        self.state_action_func,
                        self.softmax,
                        self.epsilon_func(episodes)
                    )
                steps += 1
                terminate = steps >= self.max_steps or \
                    self.mdp_rep.terminal_state_func(state)

                state = next_state
                action = next_action

            if self.offline:
                avg_grad = [g / len(states_actions) for g in
                            self.qvf_fa.get_el_tr_sum_loss_gradient(
                                states_actions,
                                targets,
                                self.gamma_lambda
                            )]
                self.qvf_fa.update_params_from_gradient(avg_grad)

            episodes += 1

            if control and self.batch_size != 0 and\
                    episodes % self.batch_size == 0:
                this_polf = get_soft_policy_func_from_qf(
                    self.qvf_fa.get_func_eval,
                    self.state_action_func,
                    self.softmax,
                    self.epsilon_func(episodes - 1)
                )


        return lambda st: lambda act, st=st: self.qvf_fa.get_func_eval((st, act))
