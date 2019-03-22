from typing import Mapping, Set, Sequence, Optional, Callable, Tuple
from Process.policy import Policy
from Process.det_policy import DetPolicy
import numpy as np
from scipy.linalg import toeplitz
from operator import itemgetter
from collections import Counter
from utils.generic_typevars import S, A
from utils.standard_typevars import SAf, PolicyType, PolicyActDictType


def get_uniform_policy(state_action_dict: Mapping[S, Set[A]]) -> Policy:
    return Policy({s: {a: 1. / len(v) for a in v} for s, v in
                   state_action_dict.items()})


def get_uniform_policy_func(state_action_func: Callable[[S], Set[A]]) \
        -> Callable[[S], Mapping[A, float]]:

    # noinspection PyShadowingNames
    def upf(s: S, state_action_func=state_action_func) -> Mapping[A, float]:
        actions = state_action_func(s)
        return {a: 1. / len(actions) for a in actions}

    return upf


def get_returns_from_rewards_terminating(
    rewards: Sequence[float],
    gamma: float
) -> np.ndarray:
    sz = len(rewards)
    return toeplitz(
        np.insert(np.zeros(sz - 1), 0, 1.),
        np.power(gamma, np.arange(sz))
    ).dot(rewards)


def get_returns_from_rewards_non_terminating(
    rewards: Sequence[float],
    gamma: float,
    points: Optional[int] = None
) -> np.ndarray:
    cnt = points if points is not None else len(rewards)
    return toeplitz(
        np.insert(np.zeros(cnt - 1), 0, 1.),
        np.concatenate((
            np.power(gamma, np.arange(len(rewards) - cnt + 1)),
            np.zeros(cnt - 1)
        ))
    ).dot(rewards)


def get_det_policy_from_qf_dict(qf_dict: SAf) -> DetPolicy:
    return DetPolicy({s: max(v.items(), key=itemgetter(1))[0]
                      for s, v in qf_dict.items()})


def get_soft_policy_from_qf_dict(
    qf_dict: SAf,
    softmax: bool,
    epsilon: float
) -> Policy:
    if softmax:
        ret = Policy({s: get_softmax_action_probs(v) for s, v in
                      qf_dict.items()})
    else:
        ret = Policy({s: get_epsilon_action_probs(v, epsilon) for s, v in
                      qf_dict.items()})
    return ret


def get_soft_policy_func_from_qf(
    qf: Callable[[Tuple[S, A]], float],
    state_action_func: Callable[[S], Set[A]],
    softmax: bool,
    epsilon: float
) -> Callable[[S], Mapping[A, float]]:

    # noinspection PyShadowingNames
    def sp_func(
        s: S,
        qf=qf,
        state_action_func=state_action_func,
        softmax=softmax,
        epsilon=epsilon
    ) -> Mapping[A, float]:
        av_dict = {a: qf((s, a)) for a in state_action_func(s)}
        return get_softmax_action_probs(av_dict) if softmax else\
            get_epsilon_action_probs(av_dict, epsilon)

    return sp_func


def get_vf_dict_from_qf_dict_and_policy(
    qf_dict: SAf,
    pol: Policy
) -> Mapping[A, float]:
    return {s: sum(pol.get_state_action_probability(s, a) * q
            for a, q in v.items()) for s, v in qf_dict.items()}


def get_policy_func_for_fa(
    pol_func: Callable[[S], Callable[[A], float]],
    state_action_func: Callable[[S], Set[A]]
) -> Callable[[S], Mapping[A, float]]:

    # noinspection PyShadowingNames
    def pf(
        s: S,
        pol_func=pol_func,
        state_action_func=state_action_func
    ) -> Mapping[A, float]:
        return {a: pol_func(s)(a) for a in state_action_func(s)}

    return pf


def get_nt_return_eval_steps(
    max_steps: int,
    gamma: float,
    eps: float
) -> int:
    low_limit = 0.2 * max_steps
    high_limit = float(max_steps - 1)
    if gamma == 0.:
        val = high_limit
    elif gamma == 1.:
        val = low_limit
    else:
        val = min(
            high_limit,
            max(
                low_limit,
                max_steps - np.log(eps) / np.log(gamma)
            )
        )
    return int(np.floor(val))


def get_epsilon_decay_func(
    epsilon,
    epsilon_half_life
) -> Callable[[int], float]:

    # noinspection PyShadowingNames
    def epsilon_decay(
        t: int,
        epsilon=epsilon,
        epsilon_half_life=epsilon_half_life
    ) -> float:
        return epsilon * np.exp(-np.log(2) / epsilon_half_life * t)

    return epsilon_decay


def get_pdf_from_samples(samples: Sequence[A]) -> Mapping[A, float]:
    num_samples = len(samples)
    c = Counter(samples)
    return {k: v / num_samples for k, v in c.items()}


def get_policy_as_action_dict(polf: PolicyType, num_samples: int)\
        -> PolicyActDictType:

    def pf(s: S) -> Mapping[A, float]:
        return get_pdf_from_samples(polf(s)(num_samples))

    return pf
