from typing import Callable, Tuple, Optional, Sequence, Mapping
from td_algo_enum import TDAlgorithm
from rl_func_approx.rl_func_approx_base import RLFuncApproxBase
from func_approx_spec import FuncApproxSpec
from Process.mdp_rep_for_rl_fa import MDPRepForRLFA
from helper_funcs import get_soft_policy_func_from_qf
import numpy as np
from utils.generic_typevars import S, A
from utils.standard_typevars import VFType, QFType, PolicyActDictType


class TDLambdaExact(RLFuncApproxBase):

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
        state_feature_funcs: Sequence[Callable[[S], float]],
        sa_feature_funcs: Sequence[Callable[[Tuple[S, A]], float]],
        learning_rate: float,
        learning_rate_decay: float
    ) -> None:
        super().__init__(
            mdp_rep_for_rl=mdp_rep_for_rl,
            exploring_start=exploring_start,
            softmax=softmax,
            epsilon=epsilon,
            epsilon_half_life=epsilon_half_life,
            num_episodes=num_episodes,
            max_steps=max_steps,
            fa_spec=FuncApproxSpec(
                state_feature_funcs=state_feature_funcs,
                sa_feature_funcs=sa_feature_funcs,
                dnn_spec=None,
                learning_rate=learning_rate,
                add_unit_feature=False
            )
        )
        self.vf_w: np.ndarray = np.zeros(self.vf_fa.num_features)
        self.qvf_w: np.ndarray = np.zeros(self.qvf_fa.num_features)
        self.vf_fa.params = [self.vf_w]
        self.qvf_fa.params = [self.qvf_w]
        self.algorithm: TDAlgorithm = algorithm
        self.gamma_lambda: float = self.mdp_rep.gamma * lambd
        self.batch_size: int = batch_size
        self.learning_rate_decay: float = learning_rate_decay

    # noinspection PyShadowingNames
    def get_value_func_fa(self, polf: PolicyActDictType) -> VFType:
        episodes = 0
        updates = 0

        while episodes < self.num_episodes:
            et = np.zeros(self.vf_fa.num_features)
            state = self.mdp_rep.init_state_gen()
            features = self.vf_fa.get_feature_vals(state)
            old_vf_fa = 0.
            steps = 0
            terminate = False

            while not terminate:
                action = get_rv_gen_func_single(polf(state))()
                next_state, reward =\
                    self.mdp_rep.state_reward_gen_func(state, action)
                next_features = self.vf_fa.get_feature_vals(next_state)
                vf_fa = features.dot(self.vf_w)
                next_vf_fa = next_features.dot(self.vf_w)
                target = reward + self.mdp_rep.gamma * next_vf_fa
                delta = target - vf_fa
                alpha = self.vf_fa.learning_rate *\
                    (updates / self.learning_rate_decay + 1) ** -0.5
                et = et * self.gamma_lambda + features *\
                    (1 - alpha * self.gamma_lambda * et.dot(features))
                self.vf_w += alpha * (et * (delta + vf_fa - old_vf_fa) -
                                      features * (vf_fa - old_vf_fa))
                updates += 1
                steps += 1
                terminate = steps >= self.max_steps or\
                    self.mdp_rep.terminal_state_func(state)
                old_vf_fa = next_vf_fa
                state = next_state
                features = next_features

            episodes += 1

        return lambda x: self.vf_fa.get_feature_vals(x).dot(self.vf_w)

    # noinspection PyShadowingNames
    def get_qv_func_fa(self, polf: Optional[PolicyActDictType]) -> QFType:
        control = polf is None
        this_polf = polf if polf is not None else self.get_init_policy_func()
        episodes = 0
        updates = 0

        while episodes < self.num_episodes:
            et = np.zeros(self.qvf_fa.num_features)
            if self.exploring_start:
                state, action = self.mdp_rep.init_state_action_gen()
            else:
                state = self.mdp_rep.init_state_gen()
                action = get_rv_gen_func_single(this_polf(state))()
            features = self.qvf_fa.get_feature_vals((state, action))
            old_qvf_fa = 0.
            steps = 0
            terminate = False

            while not terminate:
                next_state, reward = \
                    self.mdp_rep.state_reward_gen_func(state, action)
                next_action = get_rv_gen_func_single(this_polf(next_state))()
                next_features = self.qvf_fa.get_feature_vals((next_state, next_action))
                qvf_fa = features.dot(self.qvf_w)
                if self.algorithm == TDAlgorithm.QLearning and control:
                    next_qvf_fa = max(self.qvf_fa.get_feature_vals(
                        (next_state, a)).dot(self.qvf_w) for a in
                                      self.state_action_func(next_state))
                elif self.algorithm == TDAlgorithm.ExpectedSARSA and control:
                    next_qvf_fa = get_expected_action_value(
                        {a: self.qvf_fa.get_feature_vals((next_state, a)).dot(self.qvf_w)
                         for a in self.state_action_func(next_state)},
                        self.softmax,
                        self.epsilon_func(episodes)
                    )
                else:
                    next_qvf_fa = next_features.dot(self.qvf_w)

                target = reward + self.mdp_rep.gamma * next_qvf_fa
                delta = target - qvf_fa
                alpha = self.vf_fa.learning_rate * \
                    (updates / self.learning_rate_decay + 1) ** -0.5
                et = et * self.gamma_lambda + features * \
                    (1 - alpha * self.gamma_lambda * et.dot(features))
                self.qvf_w += alpha * (et * (delta + qvf_fa - old_qvf_fa) -
                                       features * (qvf_fa - old_qvf_fa))

                if control and self.batch_size == 0:
                    this_polf = get_soft_policy_func_from_qf(
                        lambda sa: self.qvf_fa.get_feature_vals(sa).dot(self.qvf_w),
                        self.state_action_func,
                        self.softmax,
                        self.epsilon_func(episodes)
                    )
                updates += 1
                steps += 1
                terminate = steps >= self.max_steps or \
                    self.mdp_rep.terminal_state_func(state)
                old_qvf_fa = next_qvf_fa
                state = next_state
                action = next_action
                features = next_features

            episodes += 1

            if control and self.batch_size != 0 and\
                    episodes % self.batch_size == 0:
                this_polf = get_soft_policy_func_from_qf(
                    self.qvf_fa.get_func_eval,
                    self.state_action_func,
                    self.softmax,
                    self.epsilon_func(episodes - 1)
                )

        return lambda st: lambda act, st=st: self.qvf_fa.get_feature_vals(
            (st, act)).dot(self.qvf_w)
