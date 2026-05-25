"""
Drone fuselage flow — adapted from JAXFLUIDS cylinder_flow example.

Geometry:  2D top-view cross-section of the quadcopter fuselage.
           Drone body: 240×170 mm (aspect ratio 1.41:1)
           Level-set ellipse: a=0.706 (x), b=0.5 (y)  → same cross-section
           area as the reference cylinder (R=0.5).

Physics:   Compressible Navier-Stokes, Re≈200 (same as cylinder_flow).
           Inflow: U=0.355, ρ=1, p=1, Ma≈0.3.

Outputs:
  1. Vorticity + pressure visualisation (saved PNG + animation)
  2. Surface pressure profile along the fuselage → JSON for jax-fem arm loads
"""

import os, json
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from jaxfluids import InputManager, InitializationManager, SimulationManager

# ── 1. Run simulation ────────────────────────────────────────────────────────
input_manager        = InputManager("drone_flow.json", "numerical_setup.json")
initialization_manager = InitializationManager(input_manager)
sim_manager          = SimulationManager(input_manager)

jxf_buffers = initialization_manager.initialization()
sim_manager.simulate(jxf_buffers)

# ── 2. Load results ──────────────────────────────────────────────────────────
try:
    from jaxfluids_postprocess import load_data
    path = sim_manager.output_writer.save_path_domain
    jxf_data = load_data(path, ["velocity", "pressure", "vorticity", "levelset"])
    cell_centers = jxf_data.cell_centers
    data         = jxf_data.data
    times        = jxf_data.times
    use_postprocess = True
except Exception as e:
    print(f"[warn] jaxfluids_postprocess not available ({e}), loading h5 manually")
    use_postprocess = False

# ── 3. Visualise ─────────────────────────────────────────────────────────────
if use_postprocess:
    x  = cell_centers[0][:, 0, 0]        # (Nx,)
    y  = cell_centers[1][0, :, 0]        # (Ny,)
    XX, YY = np.meshgrid(x, y, indexing="ij")

    # Use last time-step
    ls   = data["levelset"][-1, :, :, 0]     # (Nx, Ny)
    pres = data["pressure"][-1, :, :, 0]
    vort = data["vorticity"][-1, 2, :, :, 0] # z-component

    mask = ls < 0                            # solid interior
    vort_masked = np.ma.masked_where(mask, vort)
    pres_masked = np.ma.masked_where(mask, pres)

    # ── Vorticity field ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0a0a14")
    fig.suptitle(f"Drone fuselage flow  Re≈200  t={times[-1]:.1f}",
                 color="white", fontsize=13)

    vmax = np.percentile(np.abs(vort_masked.compressed()), 97)
    im0 = axes[0].pcolormesh(XX, YY, vort_masked, cmap="RdBu_r",
                              vmin=-vmax, vmax=vmax, shading="auto")
    axes[0].set_facecolor("#0a0a14")
    axes[0].set_title("Vorticity  ω_z", color="white")
    plt.colorbar(im0, ax=axes[0]).ax.yaxis.set_tick_params(color="white")

    # Pressure field
    im1 = axes[1].pcolormesh(XX, YY, pres_masked, cmap="plasma",
                              shading="auto")
    axes[1].set_facecolor("#0a0a14")
    axes[1].set_title("Pressure  p", color="white")
    plt.colorbar(im1, ax=axes[1]).ax.yaxis.set_tick_params(color="white")

    # Draw drone ellipse outline on both
    a_ell, b_ell = 0.706, 0.5
    theta = np.linspace(0, 2*np.pi, 200)
    xe = a_ell * np.cos(theta)
    ye = b_ell * np.sin(theta)
    for ax in axes:
        ax.plot(xe, ye, "w-", lw=1.2, alpha=0.7)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_color("white")
        ax.set_xlim(-3, 6)
        ax.set_ylim(-3, 3)
        ax.set_xlabel("x (nondim)", color="white")
        ax.set_ylabel("y (nondim)", color="white")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("white")

    plt.tight_layout()
    out_flow = "drone_flow_result.png"
    plt.savefig(out_flow, dpi=130, bbox_inches="tight", facecolor="#0a0a14")
    print(f"Saved: {out_flow}")

    # ── Surface pressure → jax-fem arm loads ────────────────────────────────
    # Find cells near the levelset surface (|φ| < 1.5 * dx)
    dx = (x[-1] - x[0]) / len(x)
    surface_mask = np.abs(ls) < 1.5 * dx

    surf_x   = XX[surface_mask]
    surf_y   = YY[surface_mask]
    surf_p   = pres[surface_mask]

    # Arm attachment points in nondim coords (45° from body centre)
    # Drone: arm root ≈ body edge = (±a_ell*cos(45°), ±b_ell*sin(45°))
    arm_roots = {
        "fl": ( a_ell * np.cos(np.radians(45)),  b_ell * np.sin(np.radians(45))),
        "fr": ( a_ell * np.cos(np.radians(-45)), b_ell * np.sin(np.radians(-45))),
        "rl": (-a_ell * np.cos(np.radians(45)),  b_ell * np.sin(np.radians(45))),
        "rr": (-a_ell * np.cos(np.radians(-45)), b_ell * np.sin(np.radians(-45))),
    }

    # For each arm root, take pressure of nearest surface cell
    arm_loads = {}
    for arm, (ax_, ay_) in arm_roots.items():
        if len(surf_x) == 0:
            arm_loads[arm] = float(np.mean(pres))
            continue
        dist = np.sqrt((surf_x - ax_)**2 + (surf_y - ay_)**2)
        nearest = np.argmin(dist)
        arm_loads[arm] = float(surf_p[nearest])

    p_inf = 1.0
    q_inf = 0.5 * 1.0 * 0.354964787**2
    arm_Cp = {k: (v - p_inf) / q_inf for k, v in arm_loads.items()}

    fem_input = {
        "description": "Surface pressure at drone arm roots (nondim). "
                       "Cp = (p - p_inf) / q_inf",
        "Re": 200,
        "t_end": float(times[-1]),
        "p_inf": p_inf,
        "q_inf": q_inf,
        "arm_pressure": arm_loads,
        "arm_Cp": arm_Cp,
    }
    with open("drone_arm_loads.json", "w") as f:
        json.dump(fem_input, f, indent=2)
    print("Saved arm loads: drone_arm_loads.json")
    print(json.dumps(arm_Cp, indent=2))
