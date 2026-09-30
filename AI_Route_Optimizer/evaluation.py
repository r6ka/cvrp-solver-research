"""
evaluation.py
-------------
* Objective functions used by the optimizer (fitness)
* Full solution evaluation + feasibility check (used for the reports)

Two objectives are available:

  "distance"  (used for the official benchmarks)
        cost(route) = sum of arc distances

  "co2"       (used for the green-logistics self-created case)
        A diesel truck burns more fuel when it is heavier, so the fuel used on
        each arc depends on the load still on the truck:
            fuel_per_km(load) = f_empty + (f_full - f_empty) * load / Q
            fuel(arc)         = distance(arc) * fuel_per_km(load on that arc)
            CO2               = fuel * emission_factor
        With this objective, the ORDER in which customers are visited matters,
        not only the distance (e.g. unload heavy customers early).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GreenParams:
    fuel_empty_l_per_km: float = 0.25   # litres/km, empty truck
    fuel_full_l_per_km: float = 0.38    # litres/km, fully loaded truck
    co2_kg_per_litre: float = 2.68      # kg CO2 per litre of diesel
    fuel_price_per_litre: float = 0.0   # optional, for cost reporting

    @classmethod
    def from_meta(cls, meta: dict | None) -> "GreenParams":
        g = (meta or {}).get("green", {})
        return cls(
            fuel_empty_l_per_km=g.get("fuel_empty_l_per_km", cls.fuel_empty_l_per_km),
            fuel_full_l_per_km=g.get("fuel_full_l_per_km", cls.fuel_full_l_per_km),
            co2_kg_per_litre=g.get("co2_kg_per_litre", cls.co2_kg_per_litre),
            fuel_price_per_litre=g.get("fuel_price_per_litre", cls.fuel_price_per_litre),
        )


def route_distance(route, D) -> float:
    if not route:
        return 0.0
    s = D[0][route[0]] + D[route[-1]][0]
    for a, b in zip(route, route[1:]):
        s += D[a][b]
    return s


def route_fuel(route, D, demand, Q, gp: GreenParams) -> float:
    """Litres of diesel for one route (load-dependent consumption)."""
    if not route:
        return 0.0
    load = sum(demand[c] for c in route)
    slope = (gp.fuel_full_l_per_km - gp.fuel_empty_l_per_km) / Q
    fuel = 0.0
    prev = 0
    for c in route:
        fuel += D[prev][c] * (gp.fuel_empty_l_per_km + slope * load)
        load -= demand[c]
        prev = c
    fuel += D[prev][0] * gp.fuel_empty_l_per_km      # return to depot empty
    return fuel


class Objective:
    """Callable route-cost function used by the GA and local search."""

    def __init__(self, inst, kind: str = "distance"):
        self.kind = kind
        self.D = inst.dist
        self.demand = inst.demand
        self.Q = inst.capacity
        self.gp = GreenParams.from_meta(inst.meta)

    def route_cost(self, route) -> float:
        if self.kind == "distance":
            return route_distance(route, self.D)
        if self.kind == "co2":
            return route_fuel(route, self.D, self.demand, self.Q, self.gp) * self.gp.co2_kg_per_litre
        raise ValueError(self.kind)

    # Is the cost of a route the same when it is driven backwards?
    @property
    def symmetric(self) -> bool:
        return self.kind == "distance"


def evaluate_solution(inst, routes) -> dict:
    """Independent full check of a solution. Returns a report dictionary."""
    D, q, Q = inst.dist, inst.demand, inst.capacity
    gp = GreenParams.from_meta(inst.meta)
    routes = [r for r in routes if r]

    visits = [c for r in routes for c in r]
    all_customers = set(range(1, inst.n_customers + 1))
    missing = sorted(all_customers - set(visits))
    duplicated = sorted({c for c in visits if visits.count(c) > 1})
    invalid = sorted({c for c in visits if c not in all_customers})

    per_route = []
    overload_total = 0.0
    for k, r in enumerate(routes, 1):
        load = sum(q[c] for c in r)
        dist = route_distance(r, D)
        fuel = route_fuel(r, D, q, Q, gp)
        over = max(0.0, load - Q)
        overload_total += over
        per_route.append({
            "vehicle": k,
            "route": [0] + list(r) + [0],
            "load": load,
            "capacity": Q,
            "utilisation_pct": 100.0 * load / Q,
            "distance": dist,
            "fuel_l": fuel,
            "co2_kg": fuel * gp.co2_kg_per_litre,
            "capacity_ok": over == 0,
        })

    total_distance = sum(r["distance"] for r in per_route)
    total_fuel = sum(r["fuel_l"] for r in per_route)
    feasible = (not missing and not duplicated and not invalid
                and overload_total == 0 and len(routes) <= inst.num_vehicles)
    return {
        "instance": inst.name,
        "n_customers": inst.n_customers,
        "capacity": Q,
        "max_vehicles": inst.num_vehicles,
        "vehicles_used": len(routes),
        "total_distance": total_distance,
        "total_fuel_l": total_fuel,
        "total_co2_kg": total_fuel * gp.co2_kg_per_litre,
        "capacity_violations": sum(1 for r in per_route if not r["capacity_ok"]),
        "missing_customers": missing,
        "duplicated_customers": duplicated,
        "invalid_customers": invalid,
        "feasible": feasible,
        "routes": per_route,
    }
