"""
solver.py  –  AI-Based Green Route Optimizer (Generalized CVRP Solver)
=====================================================================

ONE solver for ANY CVRP instance. It reads the problem from a file and
computes the routes itself (no stored answers, no instance-specific code).

Usage
    python solver.py benchmarks/A-n32-k5.vrp
    python solver.py benchmarks/*.vrp --time 60 --seed 1
    python solver.py own_case/scg_cpac_narathiwat.json --objective co2

Outputs (per instance, in results/<name>/ by default)
    solution.sol      – routes in the official CVRPLIB .sol format
    result.json       – full machine-readable report + run metadata
    result.txt        – human-readable report (same as the console)
    routes.png        – map of the routes
    convergence.png   – best distance found vs. time
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from cvrp_parser import read_instance, read_reference_value
from evaluation import Objective, evaluate_solution
from genetic_algorithm import GAConfig, HybridGA
from hgs2 import HGS2, HGS2Config
from construction import Tracer
from exact_route import RouteOptimiser

SOLVER_NAME = "AI-Based Green Route Optimizer v2.0 (hand-written HGS + exact route & set-partitioning steps)"
SOLVER_VERSION = "2.0.0"


def sha256_of(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fmt(x: float) -> str:
    return f"{x:.0f}" if abs(x - round(x)) < 1e-9 else f"{x:.2f}"


def solve(path, objective="distance", time_limit=60.0, seed=1, vehicles=None, verbose=True,
          engine="v2", tracer=None, **flags):
    inst = read_instance(path, vehicles)
    obj = Objective(inst, objective)
    if engine == "v1":
        result = HybridGA(inst, obj, GAConfig(time_limit=time_limit, seed=seed, verbose=verbose)).run()
        result.stats = {"engine": "v1 hybrid GA"}
    else:
        sp_time = min(10.0, 0.15 * time_limit)
        cfg = HGS2Config(time_limit=time_limit, seed=seed, verbose=verbose, sp_time=sp_time, **flags)
        result = HGS2(inst, obj, cfg, tracer).run()
    report = evaluate_solution(inst, result.routes)
    # exact-route certificate: is every route's visiting order provably optimal?
    ro = RouteOptimiser(inst, obj, limit=13)
    proven, better = 0, 0
    for r in result.routes:
        order, ok = ro.optimise(r)
        if ok:
            proven += 1
            better += obj.route_cost(order) < obj.route_cost(r) - 1e-9
    if not isinstance(getattr(result, "stats", None), dict):
        result.stats = {}
    result.stats["exact_route_certificate"] = {
        "routes": len(result.routes), "proven_optimal_order": proven - better,
        "checked_up_to_customers": 13}
    return inst, result, report


def text_report(inst, result, report, objective, seed, reference=None) -> str:
    L = []
    L.append("=" * 64)
    L.append(f" {SOLVER_NAME}")
    L.append("=" * 64)
    L.append(f" Instance          : {inst.name}")
    L.append(f" Input file        : {inst.source_file}")
    L.append(f" Customers         : {inst.n_customers}")
    L.append(f" Vehicle capacity  : {fmt(inst.capacity)}")
    L.append(f" Max vehicles      : {inst.num_vehicles}")
    L.append(f" Total demand      : {fmt(sum(inst.demand))}")
    L.append(f" Objective         : {objective}")
    L.append(f" Random seed       : {seed}")
    L.append("-" * 64)
    for r in report["routes"]:
        nodes = r["route"]
        path = " -> ".join(str(c) for c in nodes)
        L.append(f" Vehicle {r['vehicle']}: {path}")
        if inst.labels is not None:
            L.append("     " + " > ".join(inst.labels[c] for c in nodes[1:-1]))
        extra = f"   CO2 {r['co2_kg']:.1f} kg" if (inst.meta.get("green") or objective == "co2") else ""
        L.append(f"     load {fmt(r['load'])}/{fmt(r['capacity'])} "
                 f"({r['utilisation_pct']:.0f}%)   distance {fmt(r['distance'])}{extra}")
    L.append("-" * 64)
    L.append(f" Total Distance      : {fmt(report['total_distance'])}")
    L.append(f" Vehicles used       : {report['vehicles_used']} / {inst.num_vehicles}")
    L.append(f" Capacity violations : {report['capacity_violations']}")
    L.append(f" Missing customers   : {len(report['missing_customers'])}")
    L.append(f" Duplicate visits    : {len(report['duplicated_customers'])}")
    L.append(f" Feasible            : {'YES' if report['feasible'] else 'NO'}")
    if inst.meta.get("green") or objective == "co2":
        L.append(f" Estimated fuel      : {report['total_fuel_l']:.1f} L")
        L.append(f" Estimated CO2       : {report['total_co2_kg']:.1f} kg")
    L.append(f" Runtime             : {result.runtime:.1f} s  ({result.iterations} GA children)")
    cert = (getattr(result, "stats", None) or {}).get("exact_route_certificate")
    if cert:
        L.append(f" Exact route check   : {cert['proven_optimal_order']}/{cert['routes']} routes have a "
                 f"proven-optimal order (Held-Karp DP)")
    if reference is not None:
        gap = 100.0 * (report["total_distance"] - reference) / reference
        L.append(f" Published optimum   : {fmt(reference)}   (gap {gap:+.2f}%)  "
                 f"[read from file comment AFTER solving]")
    L.append("=" * 64)
    return "\n".join(L)


def sol_format(report) -> str:
    lines = []
    for r in report["routes"]:
        lines.append(f"Route #{r['vehicle']}: " + " ".join(str(c) for c in r["route"][1:-1]))
    lines.append(f"Cost {fmt(report['total_distance'])}")
    return "\n".join(lines) + "\n"


def save_outputs(out_dir, inst, result, report, objective, seed, time_limit, reference, text, command=""):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "solution.sol").write_text(sol_format(report))
    (out / "result.txt").write_text(text + "\n", encoding="utf-8")
    meta = {
        "solver": SOLVER_NAME,
        "solver_version": SOLVER_VERSION,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": command or ("python " + " ".join(sys.argv)),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "input_file": inst.source_file,
        "input_sha256": sha256_of(inst.source_file),
        "objective": objective,
        "seed": seed,
        "time_limit_s": time_limit,
        "runtime_s": round(result.runtime, 3),
        "ga_children": result.iterations,
        "reference_value_from_file_comment": reference,
        "convergence_history": result.history,
        "engine_statistics": getattr(result, "stats", None),
    }
    payload = {"meta": meta, "result": report}
    (out / "result.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        from plots import plot_routes, plot_convergence, render_console
        plot_routes(inst, report, out / "routes.png")
        render_console(text, out / "console_output.png",
                       f"> {meta['command']}\n  run at {meta['run_timestamp_utc']} (UTC)   "
                       f"input SHA-256 {meta['input_sha256'][:16]}...")
        plot_convergence(result.history, inst.name, out / "convergence.png", reference)
    except Exception as e:  # plotting is optional
        print(f"  (plotting skipped: {e})")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generalized CVRP solver (Hybrid GA + Local Search)")
    ap.add_argument("files", nargs="+", help=".vrp (TSPLIB/CVRPLIB) or .json instance files")
    ap.add_argument("--objective", choices=["distance", "co2"], default="distance")
    ap.add_argument("--time", type=float, default=60.0, help="time limit per instance (s)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--vehicles", type=int, default=None, help="override number of vehicles")
    ap.add_argument("--out", default="results", help="output folder")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--engine", choices=["v1", "v2"], default="v2")
    ap.add_argument("--trace", action="store_true", help="write trace.txt with every solving step")
    ap.add_argument("--no-bandit", action="store_true")
    ap.add_argument("--no-swap-star", action="store_true")
    ap.add_argument("--no-exact", action="store_true", help="disable Held-Karp route polishing")
    ap.add_argument("--no-sp", action="store_true", help="disable set-partitioning recombination")
    args = ap.parse_args(argv)
    raw = argv if argv is not None else sys.argv[1:]

    for f in args.files:
        print(f"\n>>> Solving {f}  (objective={args.objective}, seed={args.seed}, "
              f"time limit={args.time:.0f}s)")
        tracer = Tracer(args.trace)
        flags = {}
        if args.engine == "v2":
            flags = dict(use_bandit=not args.no_bandit, use_swap_star=not args.no_swap_star,
                         use_exact=not args.no_exact, use_sp=not args.no_sp)
        inst, result, report = solve(f, args.objective, args.time, args.seed, args.vehicles,
                                     verbose=not args.quiet, engine=args.engine, tracer=tracer, **flags)
        reference = read_reference_value(f)   # only for the gap line, after solving
        text = text_report(inst, result, report, args.objective, args.seed, reference)
        print(text)
        suffix = "" if args.objective == "distance" else f"_{args.objective}"
        out_dir = Path(args.out) / f"{inst.name}{suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)
        tracer.save(out_dir / "trace.txt")
        save_outputs(out_dir, inst, result, report,
                     args.objective, args.seed, args.time, reference, text,
                     command="python solver.py " + " ".join(
                         [f] + [x for x in raw if not x.endswith((".vrp", ".json"))]))


if __name__ == "__main__":
    main()
