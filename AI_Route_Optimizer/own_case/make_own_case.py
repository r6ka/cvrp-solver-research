"""
make_own_case.py – builds our Self-created Case (Mission 3)

Scenario: "Green delivery of SCG-CPAC building materials in Narathiwat"
  * One (hypothetical) regional distribution centre in Mueang Narathiwat
  * 25 delivery points (hardware stores / construction sites) in 12 districts
  * Fleet: 6 six-wheel trucks, 12 tonnes payload each
  * Demand: tonnes of cement / concrete products ordered for the day

District-office coordinates are taken from Wikipedia (see README). Delivery
points are placed a few km around each district office with a fixed random
seed, so the case is reproducible. Coordinates are converted to kilometres
(x = east, y = north of the depot); straight-line distance is multiplied by a
road circuity factor of 1.3 to approximate real road distance.

Run once:  python own_case/make_own_case.py   -> own_case/scg_cpac_narathiwat.json
"""
import json
import math
import random
from pathlib import Path

DEPOT = ("CPAC Distribution Centre (Mueang Narathiwat)", 6.42611, 101.82306)

# district office (lat, lon) and number of delivery points in that district
DISTRICTS = [
    ("Mueang Narathiwat", 6.42611, 101.82306, 3),
    ("Tak Bai",           6.25889, 102.05500, 3),
    ("Bacho",             6.51694, 101.65167, 2),
    ("Yi-ngo",            6.40306, 101.70611, 2),
    ("Ra-ngae",           6.29639, 101.72833, 2),
    ("Rueso",             6.39472, 101.51833, 2),
    ("Si Sakhon",         6.23139, 101.50000, 1),
    ("Waeng",             5.92806, 101.88389, 2),
    ("Sukhirin",          5.93889, 101.77056, 1),
    ("Su-ngai Kolok",     6.02944, 101.96611, 3),
    ("Su-ngai Padi",      6.08528, 101.88028, 2),
    ("Chanae",            6.09833, 101.69333, 2),
]
COASTAL = {"Mueang Narathiwat", "Tak Bai"}
KINDS = ["Hardware store", "Construction site", "Precast yard", "Housing project"]

rng = random.Random(2026)
lat0, lon0 = DEPOT[1], DEPOT[2]
KM_LAT = 110.57
KM_LON = 111.32 * math.cos(math.radians(lat0))


def to_xy(lat, lon):
    return round((lon - lon0) * KM_LON, 2), round((lat - lat0) * KM_LAT, 2)


customers = []
for name, lat, lon, count in DISTRICTS:
    for k in range(count):
        x, y = to_xy(lat, lon)
        r = rng.uniform(1.0, 4.0) if not (name.startswith("Mueang") and k == 0) else rng.uniform(2.0, 3.0)
        # coastal districts: keep points on the land (west) side of the office
        a = rng.uniform(0.5 * math.pi, 1.5 * math.pi) if name in COASTAL else rng.uniform(0, 2 * math.pi)
        customers.append({
            "name": f"{name} - {rng.choice(KINDS)} {k + 1}",
            "district": name,
            "x": round(x + r * math.cos(a), 2),
            "y": round(y + r * math.sin(a), 2),
            "demand": round(rng.choice([0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.4, 2.5, 3.0, 3.2, 3.6]), 1),
        })

case = {
    "name": "SCG-CPAC-Narathiwat-25",
    "description": ("Self-created CVRP case: daily delivery of cement and concrete products "
                    "from one distribution centre to 25 customers in 12 districts of Narathiwat."),
    "units": {"distance": "km", "demand": "tonnes"},
    "road_circuity_factor": 1.3,
    "rounding": "none",
    "depot": {"name": DEPOT[0], "x": 0.0, "y": 0.0, "lat": DEPOT[1], "lon": DEPOT[2]},
    "vehicle": {"type": "6-wheel truck", "capacity": 12.0, "count": 6},
    "green": {
        "fuel_empty_l_per_km": 0.22,
        "fuel_full_l_per_km": 0.33,
        "co2_kg_per_litre": 2.68,
        "fuel_price_per_litre": 32.0,
        "_note": "Fuel figures are planning assumptions; 2.68 kg CO2/L is a standard diesel factor."
    },
    "customers": customers,
}

out = Path(__file__).with_name("scg_cpac_narathiwat.json")
out.write_text(json.dumps(case, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"wrote {out}  customers={len(customers)}  total demand={sum(c['demand'] for c in customers):.1f} t")
