from typing import Mapping, Optional, Tuple, Sequence
from algorithms.rl_func_approx.rl_func_approx_base import RLFuncApproxBase
from algorithms.func_approx_spec import FuncApproxSpec
from processes.mp_funcs import get_rv_gen_func_single
from processes.mdp_rep_for_rl_fa import MDPRepForRLFA
from algorithms.helper_funcs import get_returns_from_rewards_terminating
from algorithms.helper_funcs import get_returns_from_rewards_non_terminating
from algorithms.helper_funcs import get_soft_policy_func_from_qf
from algorithms.helper_funcs import get_nt_return_eval_steps
import numpy as np
from utils.generic_typevars import S, A
from utils.standard_typevars import VFType, QFType, PolicyActDictType


class MonteCarlo(RLFuncApproxBase):

    def __init__(
        self,
        mdp_rep_for_rl: MDPRepForRLFA,
        exploring_start: bool,
        softmax: bool,
        epsilon: float,
        epsilon_half_life: float,
        num_episodes: int,
        max_steps: int,
        fa_spec: FuncApproxSpec
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
        self.nt_return_eval_steps = get_nt_return_eval_steps(
            max_steps,
            mdp_rep_for_rl.gamma,
            1e-4
        )

    def get_mc_path(
        self,
        polf: PolicyActDictType,
        start_state: S,
        start_action: Optional[A] = None
    ) -> Sequence[Tuple[S, A, float]]:

        res = []
        state = start_state
        steps = 0
        terminate = False

        while not terminate:
            action = get_rv_gen_func_single(polf(state))()\
                if (steps > 0 or start_action is None) else start_action
            next_state, reward =\
                self.mdp_rep.state_reward_gen_func(state, action)
            res.append((state, action, reward))
            steps += 1
            terminate = steps >= self.max_steps or\
                self.mdp_rep.terminal_state_func(state)
            state = next_state
        return res

    def get_value_func_fa(self, polf: PolicyActDictType) -> VFType:
        episodes = 0

        while episodes < self.num_episodes:
            start_state = self.mdp_rep.init_state_gen()
            mc_path = self.get_mc_path(
                polf,
                start_state,
                start_action=None
            )

            rew_arr = np.array([x for _, _, x in mc_path])
            if self.mdp_rep.terminal_state_func(mc_path[-1][0]):
                returns = get_returns_from_rewards_terminating(
                    rew_arr,
                    self.mdp_rep.gamma
                )
            else:
                returns = get_returns_from_rewards_non_terminating(
                    rew_arr,
                    self.mdp_rep.gamma,
                    self.nt_return_eval_steps
                )

            sgd_pts = [(mc_path[i][0], r) for i, r in enumerate(returns)]
            self.vf_fa.update_params(*zip(*sgd_pts))

            episodes += 1

        return self.vf_fa.get_func_eval

    # noinspection PyShadowingNames
    def get_qv_func_fa(self, polf: Optional[PolicyActDictType]) -> QFType:
        control = polf is None
        this_polf = polf if polf is not None else self.get_init_policy_func()
        episodes = 0

        while episodes < self.num_episodes:
            if self.exploring_start:
                start_state, start_action = self.mdp_rep.init_state_action_gen()
            else:
                start_state = self.mdp_rep.init_state_gen()
                start_action = None
            mc_path = self.get_mc_path(
                this_polf,
                start_state,
                start_action
            )
            rew_arr = np.array([x for _, _, x in mc_path])
            if self.mdp_rep.terminal_state_func(mc_path[-1][0]):
                returns = get_returns_from_rewards_terminating(
                    rew_arr,
                    self.mdp_rep.gamma
                )
            else:
                returns = get_returns_from_rewards_non_terminating(
                    rew_arr,
                    self.mdp_rep.gamma,
                    self.nt_return_eval_steps
                )

            sgd_pts = [((mc_path[i][0], mc_path[i][1]), r) for i, r in
                       enumerate(returns)]
            # MC is offline update and so, policy improves after each episode
            self.qvf_fa.update_params(*zip(*sgd_pts))

            if control:
                this_polf = get_soft_policy_func_from_qf(
                    self.qvf_fa.get_func_eval,
                    self.state_action_func,
                    self.softmax,
                    self.epsilon_func(episodes)
                )
            episodes += 1

        return lambda st: lambda act, st=st: self.qvf_fa.get_func_eval((st, act))
