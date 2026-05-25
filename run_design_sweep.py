"""Design sweep: compare quad / pentacopter / Y6 using physics-fitted motor models.

For each copter XML this script:
  1. Loads the configuration and prints physics summary
  2. Shows the allocation matrix and hover trim condition
  3. Runs a 20 s waypoint mission (climb 2 m → forward 4 m → lateral 2 m → yaw 60°)
  4. Collects: settle time, peak tilt, hover RPM, motor load spread, energy estimate
  5. Produces a 3-column comparison figure saved to assets/

Usage:
  JAX_PLATFORMS=cpu conda run -n num_python python run_design_sweep.py
"""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from pathlib import Path
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
_GI = REPO.parent
if str(_GI) not in sys.path:
    sys.path.insert(0, str(_GI))

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from multicopter_jax.loader import copter_params, print_summary
from dynamics.jax_copter import run as run_mission, build_mission

DESIGN_ROOT = Path("/home/kaiser/projects/gi/multicopter_design/resources/copter")

CONFIGS = [
    {"name": "Quadcopter",   "xml": "quadcopter.xml",   "mass": 1.0,  "color": "#00d4ff"},
    {"name": "Pentacopter",  "xml": "pentacopter.xml",  "mass": 1.2,  "color": "#ff6b35"},
    {"name": "Y6 Coaxial",   "xml": "y6copter.xml",     "mass": 1.5,  "color": "#7fff00"},
]

T_END  = 20.0
DT     = 1.0 / 500.0
OUT    = REPO / "assets"
OUT.mkdir(exist_ok=True)


def energy_estimate(log_m: np.ndarray, params: dict, dt: float) -> float:
    """Rough electrical energy (J): sum(T_i * sqrt(T_i/kT) * kQ/kT * omega_i) * dt."""
    kT = params["kT"]
    kQ = params["kQ"]
    omega = np.sqrt(np.maximum(log_m / kT, 0.))   # (N_steps, N_rotors) rad/s
    torque = kQ * omega**2                          # N·m
    power  = torque * omega                         # W (mechanical)
    return float(np.sum(power) * dt)                # J


def run_config(cfg: dict) -> dict:
    xml   = DESIGN_ROOT / cfg["xml"]
    mass  = cfg["mass"]
    name  = cfg["name"]

    print(f"\n{'='*60}")
    print(f"  {name}  ({xml.name}, {mass:.1f} kg)")
    print(f"{'='*60}")

    params = copter_params(xml, total_mass=mass)
    print_summary(params)

    # Hover trim (min-norm)
    T_trim = params["T_hover_vec"]
    T_eq   = params["T_hover"]
    print(f"\n  Hover trim per motor: {T_trim.round(3)}")
    print(f"  Equal-thrust baseline: {T_eq:.3f} N/motor")
    print(f"  Trim spread:           {T_trim.max()-T_trim.min():.3f} N  "
          f"(σ={T_trim.std():.3f})")

    A = params["A"]
    cond = float(np.linalg.cond(A @ A.T))
    print(f"  A condition number:    {cond:.2f}")

    t0 = time.time()
    result = run_mission(
        xml_file=xml, total_mass=mass,
        T_end=T_END, dt=DT,
        out_fig=OUT / f"sweep_{cfg['xml'].replace('.xml','')}.png",
        verbose=True,
    )
    elapsed = time.time() - t0

    log_s  = result["log_states"]
    log_m  = result["log_motors"]
    euler  = result["euler_deg"]
    RPM    = result["RPM"]
    t_arr  = result["log_t"]

    # Metrics
    peak_roll  = float(np.abs(euler[:, 0]).max())
    peak_pitch = float(np.abs(euler[:, 1]).max())
    peak_yaw   = float(np.abs(euler[:, 2]).max())
    hover_rpm  = result["hover_rpm"]
    settle_z   = result["settle_time_s"]
    energy_J   = energy_estimate(log_m, params, DT)

    # Motor load balance at steady hover (last 100 steps)
    T_steady = log_m[-100:].mean(axis=0)
    load_spread = float(T_steady.max() - T_steady.min())

    print(f"\n  ── Results ──────────────────────────────────")
    print(f"  Sim time:         {elapsed:.1f} s")
    print(f"  Hover RPM:        {hover_rpm:.0f}")
    print(f"  Settle time (z):  {settle_z:.2f} s")
    print(f"  Peak roll:        {peak_roll:.2f}°")
    print(f"  Peak pitch:       {peak_pitch:.2f}°")
    print(f"  Motor spread (ss):{load_spread:.4f} N")
    print(f"  Energy est.:      {energy_J/1000:.2f} kJ  ({energy_J/3600:.3f} Wh)")

    return {
        "name":       name,
        "color":      cfg["color"],
        "params":     params,
        "t":          t_arr,
        "pos":        log_s[:, :3],
        "euler_deg":  euler,
        "RPM":        RPM,
        "log_m":      log_m,
        "hover_rpm":  hover_rpm,
        "settle_z":   settle_z,
        "peak_roll":  peak_roll,
        "peak_pitch": peak_pitch,
        "peak_yaw":   peak_yaw,
        "energy_kJ":  energy_J / 1000,
        "T_trim":     T_trim,
        "load_spread":load_spread,
        "N":          params["N"],
        "mass":       mass,
    }


def make_comparison_figure(results: list, setpoints: np.ndarray) -> None:
    """6-panel comparison: altitude, x-position, roll, yaw, RPM-M1, energy."""
    fig = plt.figure(figsize=(18, 12), facecolor="#0a0a14")
    fig.suptitle(
        "Design Sweep — Quad vs Pentacopter vs Y6  (multicopter_jax physics)",
        color="white", fontsize=14, y=0.98,
    )
    gs = gridspec.GridSpec(3, 3, hspace=0.4, wspace=0.32)
    gs.update(left=0.06, right=0.97, top=0.93, bottom=0.07)

    # Reference setpoints (from last result's t array)
    t_ref = results[0]["t"]
    sp_z  = setpoints[:len(t_ref), 2]
    sp_psi_deg = np.degrees(setpoints[:len(t_ref), 3])

    def styled(ax, ylabel):
        ax.set_facecolor("#0a0a14")
        ax.set_xlabel("t (s)", color="white", fontsize=9)
        ax.set_ylabel(ylabel, color="white", fontsize=9)
        ax.tick_params(colors="white", labelsize=8)
        ax.spines[:].set_color("#555")
        ax.grid(alpha=0.12, color="white")

    panels = [
        (0, 0, "Altitude z (m)",       lambda r: r["pos"][:, 2],           sp_z),
        (0, 1, "Position x (m)",        lambda r: r["pos"][:, 0],           None),
        (0, 2, "Position y (m)",        lambda r: r["pos"][:, 1],           None),
        (1, 0, "Roll φ (°)",            lambda r: r["euler_deg"][:, 0],     None),
        (1, 1, "Yaw ψ (°)",             lambda r: r["euler_deg"][:, 2],     sp_psi_deg),
        (1, 2, "Airspeed |V| (m/s)",    lambda r: np.sqrt(np.gradient(r["pos"][:, 0], r["t"])**2 +
                                                                   np.gradient(r["pos"][:, 1], r["t"])**2 +
                                                                   np.gradient(r["pos"][:, 2], r["t"])**2), None),
        (2, 0, "Motor M1 RPM",          lambda r: r["RPM"][:, 0],           None),
        (2, 1, "Peak motor RPM",         lambda r: r["RPM"].max(axis=1),     None),
        (2, 2, "Cum. energy (J)",        None,                               None),
    ]

    for row, col, ylabel, getter, sp_ov in panels:
        ax = fig.add_subplot(gs[row, col])
        if getter is None:
            # Special: cumulative energy
            for r in results:
                kT = r["params"]["kT"]
                kQ = r["params"]["kQ"]
                omega = np.sqrt(np.maximum(r["log_m"] / kT, 0.))
                pwr   = (kQ * omega**3).sum(axis=1)         # W
                energy_cum = np.cumsum(pwr) * DT
                ax.plot(r["t"], energy_cum, color=r["color"], lw=1.4, label=r["name"])
            ax.legend(facecolor="#0a0a14", edgecolor="#555", labelcolor="white", fontsize=8)
        else:
            for r in results:
                ax.plot(r["t"], getter(r), color=r["color"], lw=1.4, label=r["name"])
            if sp_ov is not None:
                ax.plot(t_ref, sp_ov, "--", color="white", alpha=0.3, lw=1., label="setpoint")
            ax.legend(facecolor="#0a0a14", edgecolor="#555", labelcolor="white", fontsize=8)
        styled(ax, ylabel)

    out = OUT / "design_sweep_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0a0a14")
    print(f"\n  Saved: {out}")
    plt.close(fig)


def make_stats_figure(results: list) -> None:
    """Bar chart summary of key metrics."""
    fig, axes = plt.subplots(1, 5, figsize=(16, 4), facecolor="#0a0a14")
    fig.suptitle("Design Sweep — Key Metrics", color="white", fontsize=12, y=1.02)

    names  = [r["name"] for r in results]
    colors = [r["color"] for r in results]
    x      = np.arange(len(results))

    metrics = [
        ("Settle z (s)",     [r["settle_z"]    for r in results]),
        ("Peak roll (°)",    [r["peak_roll"]    for r in results]),
        ("Hover RPM",        [r["hover_rpm"]    for r in results]),
        ("Energy (kJ)",      [r["energy_kJ"]    for r in results]),
        ("Trim spread (N)",  [r["load_spread"]  for r in results]),
    ]
    for ax, (ylabel, vals) in zip(axes, metrics):
        bars = ax.bar(x, vals, color=colors, edgecolor="#333", width=0.55)
        ax.set_xticks(x); ax.set_xticklabels([r["name"].split()[0] for r in results],
                                              color="white", fontsize=9)
        ax.set_ylabel(ylabel, color="white", fontsize=9)
        ax.set_facecolor("#0a0a14"); ax.tick_params(colors="white", labelsize=8)
        ax.spines[:].set_color("#555"); ax.grid(axis="y", alpha=0.15, color="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01,
                    f"{v:.2f}", ha="center", va="bottom", color="white", fontsize=8)

    plt.tight_layout()
    out = OUT / "design_sweep_stats.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0a0a14")
    print(f"  Saved: {out}")
    plt.close(fig)


def print_table(results: list) -> None:
    """Print markdown-compatible comparison table."""
    print("\n" + "="*70)
    print("  Design Sweep Results")
    print("="*70)
    hdr = f"{'Config':<14}  {'N':<4}  {'Mass':>6}  {'Hover RPM':>10}  {'Settle':>8}  {'Peak roll':>10}  {'Energy':>9}  {'Spread':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"  {r['name']:<12}  {r['N']:<4}  {r['mass']:>5.1f}kg  "
              f"{r['hover_rpm']:>10.0f}  {r['settle_z']:>7.2f}s  "
              f"{r['peak_roll']:>9.2f}°  {r['energy_kJ']:>8.3f}kJ  "
              f"{r['load_spread']:>7.4f}N")
    print("="*70)


if __name__ == "__main__":
    print("🚁  drone_flow — Design Sweep Experiment")
    print(f"    Configs: {[c['name'] for c in CONFIGS]}")
    print(f"    Mission: {T_END}s  |  dt={DT*1000:.1f}ms (500 Hz)\n")

    # Build setpoints array for all configs (same mission)
    _, setpoints = build_mission(T_END, DT)

    results = []
    for cfg in CONFIGS:
        try:
            r = run_config(cfg)
            results.append(r)
        except Exception as e:
            print(f"  ERROR running {cfg['name']}: {e}")

    print_table(results)
    make_comparison_figure(results, setpoints)
    make_stats_figure(results)

    print("\n✅  Design sweep complete")
    print(f"    Figures → {OUT}/design_sweep_*.png")
