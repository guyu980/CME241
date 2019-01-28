from typing import Mapping, TypeVar, Sequence

X = TypeVar('X')

eps = 1e-8

def is_equal(a: float, b:float) -> bool:
    return abs(a - b) <= eps

def sum_dicts(dicts: Sequence[Mapping[X, float]]) -> Mapping[X, float]:
    return {k: sum(d.get(k, 0.) for d in dicts)
            for k in set.union(*[set(d1) for d1 in dicts])}
