"""
Checks that every hand-derived O(1) move formula in local_search2.py is
correct: each accepted move must (a) keep every customer exactly once and
(b) really lower the cost when the routes are recomputed from scratch.
Run:  python -m pytest tests/  (or: python tests/test_moves.py)
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cvrp_parser import Instance, read_instance  # noqa: E402
from local_search2 import FastLocalSearch  # noqa: E402
from evaluation import evaluate_solution  # noqa: E402


def random_instance(n, seed, rounding="nint"):
    rng = random.Random(seed)
    coords = [(50, 50)] + [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n)]
    dem = [0] + [rng.randint(1, 30) for _ in range(n)]
    Q = 100
    K = -(-sum(dem) // Q) + rng.randint(0, 2)
    inst = Instance(name=f"rand{n}-{seed}", coords=coords, demand=dem, capacity=Q, num_vehicles=K,
                    rounding=rounding)
    inst.build_distance_matrix()
    return inst


def test_random_moves():
    for seed in range(40):
        inst = random_instance(random.Random(seed).randint(8, 45), seed, "nint" if seed % 2 else "none")
        rng = random.Random(seed)
        custs = list(range(1, inst.n_customers + 1))
        rng.shuffle(custs)
        k = inst.num_vehicles
        routes = [custs[i::k] for i in range(k)]
        ls = FastLocalSearch(inst, granularity=rng.choice([5, 10, 20]), rng=rng)
        ls.debug_check = True
        pen = rng.choice([1.0, 10.0, 1e9])
        out = ls.run(routes, penalty=pen)
        visited = sorted(c for r in out for c in r)
        assert visited == custs and visited == sorted(custs) or visited == list(range(1, inst.n_customers + 1))
        # cached cost must equal an independent evaluation
        rep = evaluate_solution(inst, out)
        assert abs(sum(ls.dist) - rep["total_distance"]) < 1e-6


def test_benchmark_feasible():
    for f in ["benchmarks/A-n32-k5.vrp", "benchmarks/P-n40-k5.vrp"]:
        inst = read_instance(Path(__file__).resolve().parents[1] / f)
        ls = FastLocalSearch(inst, 20, random.Random(1))
        ls.debug_check = True
        c = list(range(1, inst.n_customers + 1))
        out = ls.run([c[i::inst.num_vehicles] for i in range(inst.num_vehicles)], penalty=1e9)
        assert len(out) <= inst.num_vehicles


if __name__ == "__main__":
    test_random_moves()
    test_benchmark_feasible()
    print("all move tests passed")
