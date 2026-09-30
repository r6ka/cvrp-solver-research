"""
run_all.py – reproduces EVERY result of version 2.0
===================================================

  1. Official benchmarks A-n32-k5, B-n31-k5, P-n40-k5 (v2.0, with trace.txt)
  2. Independent validation of every benchmark solution (validate.py)
  3. Comparison with our re-implementations of the methods in the
     manual-implementation reference repositories (classic_methods.py)
  4. Robustness: 10 random seeds per benchmark
  5. Generalisation: 8 extra CVRPLIB instances never used during development
  6. Scaling + ablation on 100-200 customer X instances:
       v1.0 | v2.0 full | v2.0 without bandit | v2.0 without exact steps
  7. Self-created case (SCG-CPAC Narathiwat): distance and CO2 objectives

Usage:
    python run_all.py            # full run (~25 min on a 2-core laptop)
    python run_all.py --fast     # quick check (~5 min)
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from classic_methods import METHODS  # noqa: E402
from cvrp_parser import read_instance, read_reference_value  # noqa: E402
from evaluation import Objective, evaluate_solution  # noqa: E402
from baselines import nearest_fill, sweep_fill  # noqa: E402
from solver import main as solver_main, solve  # noqa: E402

BENCH = ["A-n32-k5", "B-n31-k5", "P-n40-k5"]
EXTRA = ["A-n33-k5", "A-n45-k6", "A-n60-k9", "B-n45-k5", "B-n57-k9", "P-n50-k7", "P-n76-k5", "E-n51-k5"]
XINST = [("X-n101-k25", 28), ("X-n157-k13", 15), ("X-n200-k36", 38)]   # fleet = BKS routes + 2
OWN = "own_case/scg_cpac_narathiwat.json"


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s); st.flush()
        return len(s)


def bks_of(sol_path):
    for line in open(sol_path):
        if line.startswith("Cost"):
            return float(line.split()[1])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    T_MAIN = 15 if a.fast else 30
    T_CLASSIC = 3 if a.fast else 10
    T_SEED, N_SEEDS = (5, 3) if a.fast else (10, 10)
    T_EXTRA = 8 if a.fast else 20
    T_X = 15 if a.fast else 45

    os.chdir(ROOT)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    log = open(out / "run_log.txt", "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log)
    t_start = time.time()

    # 1 ── official benchmarks ------------------------------------------------
    print("\n########## 1. OFFICIAL BENCHMARKS (v2.0) ##########")
    solver_main([f"benchmarks/{b}.vrp" for b in BENCH] +
                ["--time", str(T_MAIN), "--seed", "1", "--trace", "--out", "results"])

    # 2 ── validation -----------------------------------------------------------
    print("\n########## 2. INDEPENDENT VALIDATION (validate.py, no solver code) ##########")
    for b in BENCH:
        r = subprocess.run([sys.executable, "validate.py", f"benchmarks/{b}.vrp",
                            str(out / b / "solution.sol")], capture_output=True, text=True)
        print(r.stdout.rstrip())
        (out / b / "validation.txt").write_text(r.stdout)

    # 3 ── classic reference methods -----------------------------------------
    print(f"\n########## 3. REFERENCE-REPO METHODS (our re-implementations, {T_CLASSIC}s for timed ones) ##########")
    rows = []
    for b in BENCH:
        path = f"benchmarks/{b}.vrp"
        inst = read_instance(path)
        ref = read_reference_value(path)
        for name, fn, timed in METHODS:
            t0 = time.time()
            routes = fn(inst, time_limit=T_CLASSIC, seed=1)
            rep = evaluate_solution(inst, routes)
            rows.append([b, name, rep["total_distance"], rep["vehicles_used"], rep["feasible"],
                         100 * (rep["total_distance"] - ref) / ref, round(time.time() - t0, 1)])
        v2 = json.loads((out / b / "result.json").read_text())
        rows.append([b, "v2.0 (ours)", v2["result"]["total_distance"], v2["result"]["vehicles_used"],
                     v2["result"]["feasible"], 100 * (v2["result"]["total_distance"] - ref) / ref,
                     v2["meta"]["runtime_s"]])
    with open(out / "comparison_reference_methods.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance", "method", "distance", "vehicles", "feasible", "gap_pct", "seconds"])
        w.writerows(rows)
    print(f"{'instance':9s} {'method':50s} {'dist':>6s} {'veh':>3s} {'feas':>5s} {'gap %':>7s}")
    for r in rows:
        print(f"{r[0]:9s} {r[1]:50s} {r[2]:6.0f} {r[3]:3d} {str(r[4]):>5s} {r[5]:7.2f}")

    # 4 ── robustness -----------------------------------------------------------
    print(f"\n########## 4. ROBUSTNESS: {N_SEEDS} SEEDS x {T_SEED}s ##########")
    seed_rows = []
    for b in BENCH:
        path = f"benchmarks/{b}.vrp"
        ref = read_reference_value(path)
        vals, ttb = [], []
        for s in range(1, N_SEEDS + 1):
            inst, res, rep = solve(path, "distance", T_SEED, s, verbose=False)
            vals.append(rep["total_distance"])
            ttb.append(res.history[-1][1] if res.history else None)
            seed_rows.append([b, s, rep["total_distance"], rep["feasible"], ttb[-1]])
        hits = sum(1 for v in vals if abs(v - ref) < 1e-9)
        print(f"{b}: best {min(vals):.0f}  mean {statistics.mean(vals):.1f}  worst {max(vals):.0f}  "
              f"optimum {hits}/{N_SEEDS}  mean time-to-best {statistics.mean(ttb):.2f}s")
    with open(out / "robustness_seeds.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance", "seed", "distance", "feasible", "time_to_best_s"])
        w.writerows(seed_rows)

    # 5 ── generalisation --------------------------------------------------------
    print(f"\n########## 5. GENERALISATION: {len(EXTRA)} EXTRA CVRPLIB INSTANCES ({T_EXTRA}s) ##########")
    gen_rows = []
    for e in EXTRA:
        path = f"extra_tests/{e}.vrp"
        inst, res, rep = solve(path, "distance", T_EXTRA, 1, verbose=False)
        ref = read_reference_value(path)
        gap = 100 * (rep["total_distance"] - ref) / ref
        gen_rows.append([e, inst.n_customers, inst.num_vehicles, rep["total_distance"], ref, gap, rep["feasible"]])
        print(f"{e:10s} n={inst.n_customers:3d} K={inst.num_vehicles}  ours {rep['total_distance']:6.0f}  "
              f"optimum {ref:6.0f}  gap {gap:5.2f}%  feasible {rep['feasible']}")
    with open(out / "generalisation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance", "customers", "vehicles", "distance", "optimum", "gap_pct", "feasible"])
        w.writerows(gen_rows)

    # 6 ── scaling + ablation ----------------------------------------------------
    print(f"\n########## 6. SCALING + ABLATION ON X INSTANCES ({T_X}s each) ##########")
    variants = [("v1.0", dict(engine="v1")),
                ("v2.0 full", dict()),
                ("v2.0 no bandit", dict(use_bandit=False)),
                ("v2.0 no exact steps", dict(use_exact=False, use_sp=False))]
    abl_rows = []
    for name, k in XINST:
        path = f"extra_tests/x/{name}.vrp"
        bks = bks_of(f"extra_tests/x/{name}.sol")
        line = f"{name:11s} BKS {bks:7.0f} |"
        for vname, flags in variants:
            flags = dict(flags)
            engine = flags.pop("engine", "v2")
            inst, res, rep = solve(path, "distance", T_X, 1, k, verbose=False, engine=engine, **flags)
            gap = 100 * (rep["total_distance"] - bks) / bks
            abl_rows.append([name, vname, rep["total_distance"], bks, gap, rep["feasible"], res.iterations])
            line += f" {vname}: {rep['total_distance']:7.0f} ({gap:+.2f}%) |"
        print(line)
    with open(out / "ablation_x_instances.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance", "variant", "distance", "bks", "gap_pct", "feasible", "iterations"])
        w.writerows(abl_rows)

    # 7 ── self-created case -----------------------------------------------------
    print("\n########## 7. SELF-CREATED CASE: SCG-CPAC NARATHIWAT ##########")
    solver_main([OWN, "--time", str(T_MAIN), "--seed", "1", "--trace", "--out", "results"])
    solver_main([OWN, "--time", str(T_MAIN), "--seed", "1", "--objective", "co2", "--trace", "--out", "results"])
    inst = read_instance(OWN)
    own_rows = []
    for label, routes in [("Sweep + fill trucks (manual-style plan)", sweep_fill(inst)),
                          ("Nearest neighbour", nearest_fill(inst))]:
        rep = evaluate_solution(inst, routes)
        own_rows.append([label, rep["total_distance"], rep["total_fuel_l"], rep["total_co2_kg"], rep["vehicles_used"]])
    for label, folder in [("v2.0 - min distance", inst.name), ("v2.0 - min CO2", inst.name + "_co2")]:
        rep = json.loads((out / folder / "result.json").read_text())["result"]
        own_rows.append([label, rep["total_distance"], rep["total_fuel_l"], rep["total_co2_kg"], rep["vehicles_used"]])
    with open(out / "own_case_comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "distance_km", "fuel_l", "co2_kg", "trucks"])
        w.writerows(own_rows)
    print(f"{'method':42s} {'km':>7s} {'fuel L':>7s} {'CO2 kg':>7s} {'trucks':>6s}")
    for r in own_rows:
        print(f"{r[0]:42s} {r[1]:7.1f} {r[2]:7.1f} {r[3]:7.1f} {r[4]:6d}")

    print(f"\nAll done in {time.time() - t_start:.0f} s. Results in {out}")
    sys.stdout = sys.__stdout__
    log.close()


if __name__ == "__main__":
    main()
