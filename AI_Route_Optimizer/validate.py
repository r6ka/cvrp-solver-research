"""
validate.py – INDEPENDENT checker for a CVRP solution
====================================================

Does NOT import any solver code. It reads the original benchmark file and a
solution file (.sol, CVRPLIB format) and recomputes everything from scratch:

  * every customer visited exactly once
  * no route exceeds the vehicle capacity
  * number of routes <= number of vehicles
  * total distance recomputed with the TSPLIB EUC_2D rule
  * the "Cost" line written in the .sol file matches the recomputed cost

Anyone (organisers, judges) can use it to verify that the reported results
were produced by the solver and not edited by hand.

Usage:  python validate.py benchmarks/A-n32-k5.vrp results/A-n32-k5/solution.sol [max_vehicles]
        (max_vehicles defaults to the -kX value in the instance name)
"""
import math
import re
import sys


def read_vrp(path):
    txt = open(path).read().splitlines()
    cap, name, coords, dem, depot, sec = None, "", {}, {}, [], None
    for line in txt:
        s = line.strip()
        if not s or s.startswith("EOF"):
            continue
        if s.startswith("NAME"):
            name = s.split(":", 1)[1].strip()
        elif s.startswith("CAPACITY"):
            cap = float(s.split(":", 1)[1])
        elif s.startswith("NODE_COORD_SECTION"):
            sec = "c"
        elif s.startswith("DEMAND_SECTION"):
            sec = "d"
        elif s.startswith("DEPOT_SECTION"):
            sec = "p"
        elif sec == "c":
            i, x, y = s.split()[:3]; coords[int(i)] = (float(x), float(y))
        elif sec == "d":
            i, q = s.split()[:2]; dem[int(i)] = float(q)
        elif sec == "p" and s.split()[0] != "-1":
            depot.append(int(s.split()[0]))
    m = re.search(r"-k(\d+)", name)
    k = int(m.group(1)) if m else None
    return name, cap, coords, dem, depot[0], k


def main(vrp, sol, max_vehicles=None):
    name, cap, coords, dem, depot, k = read_vrp(vrp)
    if max_vehicles is not None:
        k = max_vehicles
    others = [i for i in sorted(coords) if i != depot]
    node = {0: depot}                     # solution index -> file node id
    node.update({i + 1: nid for i, nid in enumerate(others)})

    def d(a, b):
        (x1, y1), (x2, y2) = coords[node[a]], coords[node[b]]
        return int(math.hypot(x1 - x2, y1 - y2) + 0.5)

    routes, claimed = [], None
    for line in open(sol):
        if line.startswith("Route"):
            routes.append([int(t) for t in line.split(":", 1)[1].split()])
        elif line.startswith("Cost"):
            claimed = float(line.split()[1])

    ok = True
    seen = [c for r in routes for c in r]
    n = len(others)
    print(f"Instance {name}: {n} customers, capacity {cap:.0f}, vehicles {k}")
    if sorted(seen) != list(range(1, n + 1)):
        ok = False
        print("  ERROR: customers missing or visited twice")
    total = 0
    for i, r in enumerate(routes, 1):
        load = sum(dem[node[c]] for c in r)
        dist = d(0, r[0]) + sum(d(a, b) for a, b in zip(r, r[1:])) + d(r[-1], 0)
        total += dist
        flag = "OK" if load <= cap else "OVER CAPACITY"
        if load > cap:
            ok = False
        print(f"  Route {i}: load {load:.0f}/{cap:.0f} {flag:14s} distance {dist}")
    if k is not None and len(routes) > k:
        ok = False
        print(f"  ERROR: {len(routes)} routes > {k} vehicles")
    print(f"  Recomputed total distance: {total}")
    print(f"  Cost written in .sol file: {claimed:.0f}" if claimed is not None else "  (no Cost line)")
    if claimed is not None and abs(claimed - total) > 1e-6:
        ok = False
        print("  ERROR: written cost does not match the recomputed cost")
    print(f"  VALID SOLUTION: {'YES' if ok else 'NO'}")
    return ok


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print(__doc__); sys.exit(2)
    mv = int(sys.argv[3]) if len(sys.argv) == 4 else None
    sys.exit(0 if main(sys.argv[1], sys.argv[2], mv) else 1)
