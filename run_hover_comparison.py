"""Hover comparison: PD vs LQR for Quad / Pentacopter / Y6.

Mirrors the multicopter_jax run_quadcopter / run_y6copter style but covers
all three configurations in one pass and adds a cross-config summary figure.

Outputs to assets/hover_comparison/:
  <config>_pd.png   — 4-panel hover plot (PD)
  <config>_lqr.png  — 4-panel hover plot (LQR)
  summary.png       — settle-time + steady-RPM bar chart (all configs × ctrl)

Usage:
  JAX_PLATFORMS=cpu conda run -n num_python python run_hover_comparison.py
"""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
_GI = REPO.parent
if str(_GI) not in sys.path:
    sys.path.insert(0, str(_GI))

import numpy as np
import jax, jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from multicopter_jax.loader     import copter_params, print_summary
from multicopter_jax.controller import build_lqr
from multicopter_jax.dynamics   import deriv, hover_state
from multicopter_jax.simulate   import run_hover, settle_time, plot_results

DESIGN_ROOT = Path("/home/kaiser/projects/gi/multicopter_design/resources/copter")
OUT = REPO / "assets" / "hover_comparison"
OUT.mkdir(parents=True, exist_ok=True)

TARGET   = (0.0, 0.0, 1.0)
T_SIM_PD  = 8.0
T_SIM_LQR = 6.0
DT = 1.0 / 500.0

CONFIGS = [
    {"name": "quadcopter",  "xml": "quadcopter.xml",  "mass": 1.0,  "color": "#00d4ff"},
    {"name": "pentacopter", "xml": "pentacopter.xml", "mass": 1.2,  "color": "#ff6b35"},
    {"name": "y6copter",    "xml": "y6copter.xml",    "mass": 1.5,  "color": "#7fff00"},
]


def lqr_stability(params) -> tuple[float, int]:
    """Return (max_real_eigenvalue, B12_rank) for the LQR closed-loop."""
    K, x0_lqr, u0_lqr, T_map, R_map = build_lqr(params)
    x0 = np.array(hover_state(TARGET))
    u0 = np.array(u0_lqr)
    A_full = np.array(jax.jacobian(lambda x: deriv(x, jnp.array(u0), params))(jnp.array(x0)))
    B_full = np.array(jax.jacobian(lambda u: deriv(jnp.array(x0), u, params))(jnp.array(u0)))
    A12 = T_map @ A_full @ R_map
    B12 = T_map @ B_full
    Acl = A12 - B12 @ K
    eigs = np.linalg.eigvals(Acl)
    return float(eigs.real.max()), int(np.linalg.matrix_rank(B12))


def run_one(cfg: dict) -> dict:
    name   = cfg["name"]
    xml    = DESIGN_ROOT / cfg["xml"]
    mass   = cfg["mass"]

    print(f"\n{'='*60}")
    print(f"  {name}  ({mass:.1f} kg)")
    print(f"{'='*60}")

    params = copter_params(xml, total_mass=mass)
    print_summary(params)

    # ── LQR stability check ────────────────────────────────────────────────
    print("\n  ── LQR stability check ──")
    max_eig, b12_rank = lqr_stability(params)
    lqr_ok = max_eig < 0
    print(f"  Max closed-loop eigenvalue: {max_eig:.4f}  ({'stable' if lqr_ok else 'UNSTABLE'})")
    print(f"  B12 rank: {b12_rank}/12")

    # ── PD simulation ─────────────────────────────────────────────────────
    print(f"\n  ── PD ({T_SIM_PD}s) ──")
    res_pd = run_hover(params, target_pos=TARGET, T_sim=T_SIM_PD, dt=DT,
                       controller="pd", start_pos=(0, 0, 0))
    ts_pd  = settle_time(res_pd["t"], res_pd["pos"], np.array(TARGET), tol=0.05)
    rpm_pd = float(res_pd["rpm"][-100:].mean())
    print(f"  Settle: {ts_pd:.3f}s   Final z: {res_pd['pos'][-1, 2]:.4f}m   Steady RPM: {rpm_pd:.0f}")

    plot_results(res_pd, params,
                 title=f"{name.capitalize()} — PD controller  ({mass:.1f} kg)",
                 save_path=OUT / f"{name}_pd.png")

    # ── LQR simulation ────────────────────────────────────────────────────
    res_lqr = ts_lqr = rpm_lqr = None
    if lqr_ok:
        print(f"\n  ── LQR ({T_SIM_LQR}s) ──")
        res_lqr = run_hover(params, target_pos=TARGET, T_sim=T_SIM_LQR, dt=DT,
                            controller="lqr", start_pos=(0, 0, 0))
        ts_lqr  = settle_time(res_lqr["t"], res_lqr["pos"], np.array(TARGET), tol=0.05)
        rpm_lqr = float(res_lqr["rpm"][-100:].mean())
        print(f"  Settle: {ts_lqr:.3f}s   Final z: {res_lqr['pos'][-1, 2]:.4f}m   Steady RPM: {rpm_lqr:.0f}")

        plot_results(res_lqr, params,
                     title=f"{name.capitalize()} — LQR controller  ({mass:.1f} kg)",
                     save_path=OUT / f"{name}_lqr.png")
    else:
        print("  ⚠  LQR skipped (unstable closed-loop)")

    return {
        "name":    name,
        "color":   cfg["color"],
        "mass":    mass,
        "N":       params["N"],
        "pd":  {"settle": ts_pd,  "rpm": rpm_pd,  "res": res_pd},
        "lqr": {"settle": ts_lqr, "rpm": rpm_lqr, "res": res_lqr, "ok": lqr_ok},
    }


def make_summary(results: list) -> None:
    """2-row summary: settle time + steady RPM for PD vs LQR across configs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Hover Comparison — PD vs LQR  (all configs)", fontsize=12)

    names  = [r["name"].capitalize() for r in results]
    colors = [r["color"] for r in results]
    x = np.arange(len(results))
    w = 0.35

    for ax, metric, label in [
        (axes[0], "settle", "Settle time (s, tol=5 cm)"),
        (axes[1], "rpm",    "Steady-state RPM"),
    ]:
        pd_vals  = [r["pd"][metric]  for r in results]
        lqr_vals = [r["lqr"][metric] if r["lqr"]["ok"] else None for r in results]

        bars_pd = ax.bar(x - w/2, pd_vals, w, color=colors,
                         edgecolor="#333", label="PD")

        lqr_plot = [v if v is not None else 0 for v in lqr_vals]
        bars_lqr = ax.bar(x + w/2, lqr_plot, w, color=colors,
                          edgecolor="#aaa", hatch="//", alpha=0.7, label="LQR")

        # annotate
        for bar, v in zip(bars_pd, pd_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        for bar, v in zip(bars_lqr, lqr_vals):
            if v is not None:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=8)
            else:
                ax.text(bar.get_x() + bar.get_width()/2, 0.02,
                        "—", ha="center", va="bottom", fontsize=9, color="gray")

        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = OUT / "summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Saved: {out}")
    plt.close(fig)


def make_overlay(results: list) -> None:
    """3-column overlay: altitude trajectory PD vs LQR for each config."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    fig.suptitle("Altitude trajectory — PD (solid) vs LQR (dashed)", fontsize=11)

    for ax, r in zip(axes, results):
        t_pd  = r["pd"]["res"]["t"]
        z_pd  = r["pd"]["res"]["pos"][:, 2]
        ax.plot(t_pd, z_pd, color=r["color"], lw=1.5, label="PD")

        if r["lqr"]["ok"]:
            t_lqr = r["lqr"]["res"]["t"]
            z_lqr = r["lqr"]["res"]["pos"][:, 2]
            ax.plot(t_lqr, z_lqr, "--", color=r["color"], lw=1.5, alpha=0.7, label="LQR")

        ax.axhline(1.0, color="k", ls=":", lw=0.8, alpha=0.4, label="target 1 m")
        ax.set_title(f"{r['name'].capitalize()}  (N={r['N']}, {r['mass']:.1f}kg)", fontsize=10)
        ax.set_xlabel("t (s)")
        ax.set_ylabel("z (m)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = OUT / "altitude_overlay.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)


def print_table(results: list) -> None:
    print("\n" + "="*70)
    print("  Hover Comparison Results")
    print("="*70)
    hdr = f"  {'Config':<14} {'N':<4} {'Mass':<7} {'PD settle':>10} {'LQR settle':>11} {'PD RPM':>8} {'LQR RPM':>9}"
    print(hdr)
    print("  " + "-"*(len(hdr)-2))
    for r in results:
        lqr_s = f"{r['lqr']['settle']:.3f}s" if r["lqr"]["ok"] else "  —"
        lqr_r = f"{r['lqr']['rpm']:.0f}"     if r["lqr"]["ok"] else "  —"
        print(f"  {r['name'].capitalize():<14} {r['N']:<4} {r['mass']:.1f}kg   "
              f"{r['pd']['settle']:>8.3f}s  {lqr_s:>10}  "
              f"{r['pd']['rpm']:>7.0f}  {lqr_r:>8}")
    print("="*70)


if __name__ == "__main__":
    print("🚁  drone_flow — Hover Comparison (PD vs LQR, all configs)")
    print(f"    Target: {TARGET}  |  tol=5 cm  |  dt={DT*1000:.1f} ms\n")

    results = []
    for cfg in CONFIGS:
        try:
            results.append(run_one(cfg))
        except Exception as e:
            import traceback
            print(f"\n  ERROR ({cfg['name']}): {e}")
            traceback.print_exc()

    print_table(results)
    make_summary(results)
    make_overlay(results)

    print(f"\n✅  Done — figures in {OUT}/")
    for f in sorted(OUT.iterdir()):
        print(f"    {f.name}")
