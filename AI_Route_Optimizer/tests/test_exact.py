"""Held-Karp must match brute force over all permutations (distance and CO2)."""
import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation import Objective  # noqa: E402
from exact_route import RouteOptimiser  # noqa: E402
from tests.test_moves import random_instance  # noqa: E402


def test_held_karp_vs_bruteforce():
    for seed in range(15):
        inst = random_instance(9, seed, "none")
        inst.meta = {"green": {}}
        for kind in ("distance", "co2"):
            obj = Objective(inst, kind)
            ro = RouteOptimiser(inst, obj, limit=12)
            rng = random.Random(seed)
            route = rng.sample(range(1, 10), rng.randint(3, 7))
            best = min(obj.route_cost(list(p)) for p in itertools.permutations(route))
            order, _ = ro.optimise(route)
            assert abs(obj.route_cost(order) - best) < 1e-9, (kind, seed)


if __name__ == "__main__":
    test_held_karp_vs_bruteforce()
    print("Held-Karp matches brute force")
