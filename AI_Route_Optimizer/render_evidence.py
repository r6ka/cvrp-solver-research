"""
render_evidence.py – re-draw the evidence figures of an existing results folder
(console_output.png, routes.png, convergence.png) from the saved result files.
It only READS result.txt / result.json; it never changes any result.

Usage:  python render_evidence.py results/A-n32-k5 [results/B-n31-k5 ...]
"""
import json
import sys
from pathlib import Path

from cvrp_parser import read_instance
from plots import plot_convergence, plot_routes, render_console

for folder in sys.argv[1:]:
    f = Path(folder)
    data = json.loads((f / "result.json").read_text(encoding="utf-8"))
    meta, rep = data["meta"], data["result"]
    text = (f / "result.txt").read_text(encoding="utf-8").rstrip("\n")
    cmd = meta["command"].replace(str(Path.cwd()) + "/", "")
    render_console(text, f / "console_output.png",
                   f"> {cmd}\n  run at {meta['run_timestamp_utc']} (UTC)   "
                   f"input SHA-256 {meta['input_sha256'][:16]}...")
    inst = read_instance(meta["input_file"])
    plot_routes(inst, rep, f / "routes.png")
    plot_convergence(meta["convergence_history"], rep["instance"], f / "convergence.png",
                     meta.get("reference_value_from_file_comment"))
    print("rendered", f)
