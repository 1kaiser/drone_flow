"""
Drone fuselage CFD — JAXFLUIDS compressible Navier-Stokes.

Geometry:  2D top-view ellipse (a=0.706, b=0.5) matching 240×170 mm fuselage.
Physics:   Compressible NS, Re=200, Ma≈0.3, end_time=60.
Outputs:   figures/drone_*.png  +  data/drone_arm_loads.json
"""
import os, json, sys
os.environ["JAX_PLATFORMS"] = "cpu"

from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
CFD  = REPO / "cfd"
sys.path.insert(0, str(REPO))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run(out_json=None, out_fig_prefix=None):
    """Run drone fuselage CFD and return arm loads dict."""
    if out_json is None:
        out_json = REPO / "data" / "drone_arm_loads.json"
    if out_fig_prefix is None:
        out_fig_prefix = REPO / "assets" / "drone"

    os.chdir(CFD)

    from jaxfluids import InputManager, InitializationManager, SimulationManager

    input_manager          = InputManager("drone_flow.json", "numerical_setup.json")
    initialization_manager = InitializationManager(input_manager)
    sim_manager            = SimulationManager(input_manager)

    jxf_buffers = initialization_manager.initialization()
    sim_manager.simulate(jxf_buffers)

    # ── Load results ──────────────────────────────────────────────────────────
    try:
        from jaxfluids_postprocess import load_data
        path     = sim_manager.output_writer.save_path_domain
        jxf_data = load_data(path, ["velocity", "pressure", "vorticity", "levelset"])
        x  = jxf_data.cell_centers[0][:, 0, 0]
        y  = jxf_data.cell_centers[1][0, :, 0]
        ls   = jxf_data.data["levelset"][-1, :, :, 0]
        pres = jxf_data.data["pressure"][-1, :, :, 0]
        vort = jxf_data.data["vorticity"][-1, 2, :, :, 0]
        t_end = float(jxf_data.times[-1])
    except Exception as e:
        print(f"[warn] postprocess fallback ({e}) — loading h5 directly")
        import h5py
        h5_dir = Path(sim_manager.output_writer.save_path_domain)
        h5_files = sorted(h5_dir.glob("*.h5"))
        with h5py.File(h5_files[-1]) as h:
            t_end = float(h["time"][()])
            x     = h["domain/gridX"][:]
            y     = h["domain/gridY"][:]
            pres  = h["primitives/pressure"][0]
            vort  = h["miscellaneous/vorticity"][0, :, :, 0]
            ls    = h["levelset/levelset"][0]

    XX, YY = np.meshgrid(x, y)
    a_ell, b_ell = 0.706, 0.5
    solid = ls < 0
    mf = lambda arr: np.ma.masked_where(solid, arr)

    # ── Figures ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0a0a14")
    fig.suptitle(f"Drone fuselage Re≈200  t={t_end:.1f}", color="white", fontsize=13)
    th = np.linspace(0, 2 * np.pi, 200)
    vmax = float(np.percentile(np.abs(mf(vort).compressed()), 97))

    axes[0].pcolormesh(XX, YY, mf(vort), cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    axes[0].set_title("Vorticity  ω_z", color="white")
    axes[1].pcolormesh(XX, YY, mf(pres), cmap="plasma", shading="auto")
    axes[1].set_title("Pressure  p", color="white")
    for ax in axes:
        ax.plot(a_ell * np.cos(th), b_ell * np.sin(th), "w-", lw=1.2, alpha=0.7)
        ax.set_xlim(-3, 6); ax.set_ylim(-3, 3)
        ax.set_facecolor("#0a0a14")
        ax.tick_params(colors="white"); ax.spines[:].set_color("#444")

    plt.tight_layout()
    out_v = str(out_fig_prefix) + "_vorticity_full.png"
    plt.savefig(out_v, dpi=130, bbox_inches="tight", facecolor="#0a0a14")
    plt.close()
    print(f"  Saved: {out_v}")

    # Final frame
    fig2, ax2 = plt.subplots(figsize=(8, 5), facecolor="#0a0a14")
    ax2.pcolormesh(XX, YY, mf(vort), cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    ax2.plot(a_ell * np.cos(th), b_ell * np.sin(th), "w-", lw=1.2, alpha=0.7)
    ax2.set_xlim(-3, 6); ax2.set_ylim(-3, 3)
    ax2.set_facecolor("#0a0a14"); ax2.set_title(f"Vorticity  t={t_end:.1f}", color="white")
    ax2.tick_params(colors="white"); ax2.spines[:].set_color("#444")
    out_f = str(out_fig_prefix) + "_final_frame.png"
    plt.savefig(out_f, dpi=130, bbox_inches="tight", facecolor="#0a0a14")
    plt.close()
    print(f"  Saved: {out_f}")

    # Surface Cp
    dx = (x[-1] - x[0]) / len(x)
    surface_mask = np.abs(ls) < 1.5 * dx
    surf_x = XX[surface_mask]; surf_y = YY[surface_mask]; surf_p = pres[surface_mask]

    fig3, ax3 = plt.subplots(figsize=(8, 5), facecolor="#0a0a14")
    sc = ax3.scatter(surf_x, surf_y, c=surf_p, cmap="plasma", s=3)
    ax3.plot(a_ell * np.cos(th), b_ell * np.sin(th), "w-", lw=1.5, alpha=0.8)
    ax3.set_facecolor("#0a0a14"); ax3.set_title("Surface pressure  p", color="white")
    ax3.tick_params(colors="white"); ax3.spines[:].set_color("#444")
    plt.colorbar(sc, ax=ax3).ax.yaxis.set_tick_params(color="white")
    out_s = str(out_fig_prefix) + "_surface_Cp.png"
    plt.savefig(out_s, dpi=130, bbox_inches="tight", facecolor="#0a0a14")
    plt.close()
    print(f"  Saved: {out_s}")

    # ── Arm root pressures ────────────────────────────────────────────────────
    arm_roots = {
        "FL": ( a_ell * np.cos(np.radians( 45)),  b_ell * np.sin(np.radians( 45))),
        "FR": ( a_ell * np.cos(np.radians(-45)),  b_ell * np.sin(np.radians(-45))),
        "RL": (-a_ell * np.cos(np.radians( 45)),  b_ell * np.sin(np.radians( 45))),
        "RR": (-a_ell * np.cos(np.radians(-45)),  b_ell * np.sin(np.radians(-45))),
    }
    arm_loads = {}
    for arm, (ax_, ay_) in arm_roots.items():
        if len(surf_x) == 0:
            arm_loads[arm] = float(np.mean(pres))
        else:
            dist = np.sqrt((surf_x - ax_) ** 2 + (surf_y - ay_) ** 2)
            arm_loads[arm] = float(surf_p[np.argmin(dist)])

    p_inf = 1.0
    q_inf = 0.5 * 1.0 * 0.354964787 ** 2
    arm_Cp = {k: (v - p_inf) / q_inf for k, v in arm_loads.items()}

    result = {
        "description": "Surface pressure at drone arm roots (nondim). Cp = (p - p_inf) / q_inf",
        "Re": 200,
        "t_end": t_end,
        "p_inf": p_inf,
        "q_inf": q_inf,
        "arm_pressure": arm_loads,
        "arm_Cp": arm_Cp,
    }
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_json}")
    return result


if __name__ == "__main__":
    result = run()
    print("\nArm Cp:", result["arm_Cp"])
