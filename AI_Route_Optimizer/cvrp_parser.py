"""
cvrp_parser.py
--------------
Reads CVRP problem data into one common structure (the `Instance` class).

Supported inputs
  * TSPLIB / CVRPLIB ``.vrp`` files (the official benchmark format)
  * ``.json`` files (used for our self-created case)

Nothing in this file (or anywhere in the solver) contains benchmark answers.
Every number the solver uses comes from the input file.

Node numbering used everywhere in this project
  0          = depot
  1 .. n     = customers
In a TSPLIB file the depot is node 1, so file node (i + 1) becomes our node i.
This is the same convention used by the official CVRPLIB ``.sol`` files.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Instance:
    name: str
    coords: list            # [(x, y)] for node 0..n   (0 = depot)
    demand: list            # [q]      for node 0..n   (demand[0] = 0)
    capacity: float         # vehicle capacity Q
    num_vehicles: int       # maximum number of vehicles (routes) allowed
    dist: list = field(default_factory=list)   # full distance matrix
    rounding: str = "nint"  # "nint" = TSPLIB EUC_2D rule, "none" = real numbers
    distance_factor: float = 1.0   # e.g. 1.3 = road/straight-line circuity factor
    labels: list | None = None     # optional names for the nodes
    meta: dict = field(default_factory=dict)  # extra data (e.g. green-logistics parameters)
    source_file: str = ""

    @property
    def n_customers(self) -> int:
        return len(self.coords) - 1

    def build_distance_matrix(self) -> None:
        """TSPLIB EUC_2D: d(i, j) = nint( sqrt((xi-xj)^2 + (yi-yj)^2) )."""
        n = len(self.coords)
        d = [[0.0] * n for _ in range(n)]
        for i in range(n):
            xi, yi = self.coords[i]
            for j in range(i + 1, n):
                xj, yj = self.coords[j]
                e = math.hypot(xi - xj, yi - yj) * self.distance_factor
                if self.rounding == "nint":
                    e = float(int(e + 0.5))          # TSPLIB nint()
                d[i][j] = d[j][i] = e
        self.dist = d


def read_vrp(path: str | Path, num_vehicles: int | None = None) -> Instance:
    """Parse a TSPLIB/CVRPLIB .vrp file."""
    path = Path(path)
    lines = path.read_text().splitlines()

    header: dict[str, str] = {}
    coords: dict[int, tuple] = {}
    demand: dict[int, float] = {}
    depots: list[int] = []
    section = None

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        up = line.upper()
        if up.startswith("EOF"):
            break
        if up.startswith("NODE_COORD_SECTION"):
            section = "coord"; continue
        if up.startswith("DEMAND_SECTION"):
            section = "demand"; continue
        if up.startswith("DEPOT_SECTION"):
            section = "depot"; continue
        if ":" in line and section is None:
            k, v = line.split(":", 1)
            header[k.strip().upper()] = v.strip()
            continue
        parts = line.split()
        if section == "coord":
            coords[int(parts[0])] = (float(parts[1]), float(parts[2]))
        elif section == "demand":
            demand[int(parts[0])] = float(parts[1])
        elif section == "depot":
            v = int(parts[0])
            if v == -1:
                section = None
            else:
                depots.append(v)

    if header.get("EDGE_WEIGHT_TYPE", "EUC_2D").upper() != "EUC_2D":
        raise ValueError("Only EUC_2D instances are supported")

    depot = depots[0] if depots else min(coords)
    # depot first, then customers in file order
    order = [depot] + [i for i in sorted(coords) if i != depot]
    inst_coords = [coords[i] for i in order]
    inst_demand = [demand.get(i, 0.0) for i in order]
    inst_demand[0] = 0.0

    name = header.get("NAME", path.stem).replace(".vrp", "")

    # Fleet size: command-line value > "VEHICLES" header > "-kX" in the name
    # > "No of trucks: X" in the comment > lower bound ceil(sum q / Q).
    capacity = float(header["CAPACITY"])
    if num_vehicles is None:
        if "VEHICLES" in header:
            num_vehicles = int(header["VEHICLES"])
        else:
            m = re.search(r"-k(\d+)", name)
            m2 = re.search(r"trucks:\s*(\d+)", header.get("COMMENT", ""), re.I)
            if m:
                num_vehicles = int(m.group(1))
            elif m2:
                num_vehicles = int(m2.group(1))
            else:
                num_vehicles = math.ceil(sum(inst_demand) / capacity)

    inst = Instance(
        name=name,
        coords=inst_coords,
        demand=inst_demand,
        capacity=capacity,
        num_vehicles=num_vehicles,
        rounding="nint",
        source_file=str(path),
    )
    inst.build_distance_matrix()
    return inst


def read_json(path: str | Path, num_vehicles: int | None = None) -> Instance:
    """Parse our own JSON case format (see own_case/*.json)."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    depot = data["depot"]
    customers = data["customers"]

    coords = [(depot["x"], depot["y"])] + [(c["x"], c["y"]) for c in customers]
    demand = [0.0] + [float(c["demand"]) for c in customers]
    labels = [depot.get("name", "Depot")] + [c.get("name", f"C{i+1}") for i, c in enumerate(customers)]

    inst = Instance(
        name=data.get("name", path.stem),
        coords=coords,
        demand=demand,
        capacity=float(data["vehicle"]["capacity"]),
        num_vehicles=num_vehicles or int(data["vehicle"]["count"]),
        rounding=data.get("rounding", "none"),
        distance_factor=float(data.get("road_circuity_factor", 1.0)),
        labels=labels,
        meta=data,
        source_file=str(path),
    )
    inst.build_distance_matrix()
    return inst


def read_instance(path: str | Path, num_vehicles: int | None = None) -> Instance:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return read_json(path, num_vehicles)
    return read_vrp(path, num_vehicles)


def read_reference_value(path: str | Path) -> float | None:
    """
    Reads the published optimal value from the COMMENT line of a .vrp file.
    Used ONLY by the reporting code to compute the gap (%) AFTER solving.
    The solver itself never sees this number.
    """
    try:
        text = Path(path).read_text()
    except OSError:
        return None
    m = re.search(r"(?:Optimal|Best)\s+value:\s*([\d.]+)", text, re.I)
    return float(m.group(1)) if m else None
