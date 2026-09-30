"""Set-partitioning B&B must equal brute-force enumeration of all exact covers."""
import itertools
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvrp_parser import Instance  # noqa: E402
from evaluation import Objective  # noqa: E402
from set_partition import RoutePool, solve_set_partitioning  # noqa: E402


def make(n, seed):
    rng = random.Random(seed)
    coords = [(50, 50)] + [(rng.randint(0, 100), rng.randint(0, 100)) for _ in range(n)]
    dem = [0] + [rng.randint(1, 10) for _ in range(n)]
    inst = Instance(name=f"sp{seed}", coords=coords, demand=dem, capacity=25, num_vehicles=n)
    inst.build_distance_matrix()
    return inst


def brute(pool, inst):
    items = [(c, frozenset(k)) for k, (c, o) in pool.routes.items()]
    full = frozenset(range(1, inst.n_customers + 1))
    best = float("inf")
    for r in range(1, inst.num_vehicles + 1):
        for comb in itertools.combinations(items, r):
            sets = [s for _, s in comb]
            if sum(len(s) for s in sets) == len(full) and frozenset().union(*sets) == full:
                best = min(best, sum(c for c, _ in comb))
    return best


def test():
    for seed in range(15):
        inst = make(7, seed)
        obj = Objective(inst, "distance")
        pool = RoutePool(inst, obj)
        rng = random.Random(seed)
        cust = list(range(1, 8))
        for c in cust:
            pool.add([c])
        for _ in range(25):
            k = rng.randint(2, 4)
            r = rng.sample(cust, k)
            if sum(inst.demand[c] for c in r) <= inst.capacity:
                pool.add(r)
        ref = brute(pool, inst)
        routes, info = solve_set_partitioning(pool, float("inf"), 10)
        got = sum(obj.route_cost(r) for r in routes)
        assert abs(got - ref) < 1e-6, (seed, got, ref)
        assert info["proven_optimal_over_pool"]
    print("all set-partitioning tests passed")


if __name__ == "__main__":
    test()
