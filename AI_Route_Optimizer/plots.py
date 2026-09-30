"""plots.py – route maps and convergence charts (matplotlib)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#20808D", "#A84B2F", "#1B474D", "#944454", "#FFC553",
          "#848456", "#6E4B8B", "#2F6FA8", "#C06C1F", "#4E8F3A"]


def _n(x):
    return f"{x:.0f}" if abs(x - round(x)) < 1e-9 else f"{x:.1f}"


def plot_routes(inst, report, path, title=None):
    fig, ax = plt.subplots(figsize=(8, 7), dpi=130)
    xs = [c[0] for c in inst.coords]
    ys = [c[1] for c in inst.coords]
    for k, r in enumerate(report["routes"]):
        pts = r["route"]
        col = COLORS[k % len(COLORS)]
        ax.plot([xs[p] for p in pts], [ys[p] for p in pts], "-", color=col, lw=2,
                label=f"Vehicle {r['vehicle']}: load {_n(r['load'])}/{_n(r['capacity'])}, "
                      f"dist {r['distance']:.0f}")
        for p in pts[1:-1]:
            ax.scatter(xs[p], ys[p], s=60, color=col, zorder=3, edgecolor="white", linewidth=0.8)
            ax.annotate(str(p), (xs[p], ys[p]), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.scatter(xs[0], ys[0], s=220, marker="s", color="black", zorder=4, label="Depot (0)")
    status = "FEASIBLE" if report["feasible"] else "INFEASIBLE"
    ax.set_title(title or f"{inst.name} - total distance {report['total_distance']:.0f} "
                          f"({report['vehicles_used']} vehicles, {status})", fontsize=11)
    ax.legend(fontsize=7, loc="best", framealpha=0.85)
    ax.grid(alpha=0.25)
    ax.set_aspect("equal", adjustable="datalim")
    if inst.meta.get("customers"):
        groups = {}
        for i, c in enumerate(inst.meta["customers"], 1):
            groups.setdefault(c.get("district", ""), []).append(inst.coords[i])
        for dname, pts in groups.items():
            if dname and not dname.startswith("Mueang"):
                gx = sum(p[0] for p in pts) / len(pts); gy = sum(p[1] for p in pts) / len(pts)
                ax.annotate(dname, (gx, gy), textcoords="offset points", xytext=(0, -14),
                            ha="center", fontsize=7.5, color="#555555", style="italic")
    if inst.meta.get("green"):
        ax.set_xlabel("x (km east of depot)"); ax.set_ylabel("y (km north of depot)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_convergence(history, name, path, reference=None):
    if not history:
        return
    fig, ax = plt.subplots(figsize=(7, 4), dpi=130)
    t = [h[1] for h in history]
    c = [h[2] for h in history]
    ax.step(t, c, where="post", color="#20808D", lw=2, marker="o", ms=3, label="best feasible found")
    if reference:
        ax.axhline(reference, color="#A84B2F", ls="--", lw=1.2, label=f"published optimum {reference:.0f}")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("objective")
    ax.set_title(f"{name} - convergence")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def render_console(text, path, title="Solver console output"):
    """Render a text report as a terminal-style PNG (evidence figure)."""
    lines = title.splitlines() + [""] + text.splitlines()
    line_h = 0.142                                   # inches per line
    h = 0.35 + line_h * len(lines)
    fig = plt.figure(figsize=(8.2, h), dpi=140)
    fig.patch.set_facecolor("#111418")
    y = 1 - 0.18 / h
    n_title = len(title.splitlines())
    fig.text(0.015, y, "\n".join(lines[:n_title]), color="#7fd1d9", family="monospace",
             fontsize=7.6, va="top", linespacing=1.3)
    fig.text(0.015, y - (n_title + 1) * line_h / h, "\n".join(lines[n_title + 1:]),
             color="#e6e6e6", family="monospace", fontsize=7.6, va="top", linespacing=1.3)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
