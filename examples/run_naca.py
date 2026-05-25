"""
NACA 0012 at Mach 2 — adapted from JAXFLUIDS examples/examples_2D/11_NACA
Runs on CPU, saves every 0.5 time units (10 snapshots total).
"""
import os, sys
os.environ["JAX_PLATFORMS"] = "cpu"

# Point at the NACA case files
NACA_DIR = "/home/kaiser/projects/gi/JAXFLUIDS/examples/examples_2D/11_NACA"
os.chdir(NACA_DIR)

import json, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jaxfluids import InputManager, InitializationManager, SimulationManager

# ── Patch case JSON: reduce save_dt so we don't get 500 files ───────────────
with open("NACA.json") as f:
    case = json.load(f)

case["general"]["save_dt"]   = 0.5   # 10 snapshots instead of 500
case["general"]["save_path"] = "/tmp/naca_results"
case["general"]["end_time"]  = 5.0

patched = "/tmp/NACA_patched.json"
with open(patched, "w") as f:
    json.dump(case, f, indent=2)

# ── Run ──────────────────────────────────────────────────────────────────────
input_manager        = InputManager(patched, "numerical_setup.json")
initialization_manager = InitializationManager(input_manager)
sim_manager          = SimulationManager(input_manager)

jxf_buffers = initialization_manager.initialization()
sim_manager.simulate(jxf_buffers)

# ── Load & plot ──────────────────────────────────────────────────────────────
try:
    from jaxfluids_postprocess import load_data
    path = sim_manager.output_writer.save_path_domain
    jxf_data = load_data(path, ["density", "pressure", "mach_number",
                                 "schlieren", "levelset", "volume_fraction"])
    cell_centers = jxf_data.cell_centers
    data         = jxf_data.data
    times        = jxf_data.times

    x  = cell_centers[0][:, 0, 0]
    y  = cell_centers[1][0, :, 0]
    XX, YY = np.meshgrid(x, y, indexing="ij")

    mask_solid = data["volume_fraction"][-1] <= 0.0

    def mfield(key, clip=None):
        f = np.ma.masked_where(mask_solid[..., 0], data[key][-1, :, :, 0])
        if clip:
            f = np.clip(f, *clip)
        return f

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor="#0a0a14")
    fig.suptitle(f"NACA 0012  Ma≈2  t={times[-1]:.2f}", color="white", fontsize=13)

    panels = [
        ("density",    mfield("density"),              "inferno",  "Density  ρ"),
        ("pressure",   mfield("pressure"),             "plasma",   "Pressure  p"),
        ("mach",       mfield("mach_number",(0,3)),    "RdBu_r",   "Mach number (clipped 0–3)"),
        ("schlieren",  mfield("schlieren",(1,500)),    "Greys_r",  "Numerical schlieren"),
    ]

    for ax, (_, field, cmap, title) in zip(axes.flat, panels):
        im = ax.pcolormesh(XX, YY, field, cmap=cmap, shading="auto")
        ax.set_facecolor("#0a0a14")
        ax.set_title(title, color="white", fontsize=10)
        plt.colorbar(im, ax=ax).ax.yaxis.set_tick_params(color="white")
        ax.set_xlim(-0.3, 1.7); ax.set_ylim(-0.5, 0.5)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_color("white")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("white")
        ax.set_xlabel("x", color="white"); ax.set_ylabel("y", color="white")

    plt.tight_layout()
    out = "/home/kaiser/projects/gi/drone_flow/assets/naca_result.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0a0a14")
    print(f"Saved: {out}")

except Exception as e:
    print(f"[warn] postprocess failed: {e}")
    print("Raw HDF5 files are in /tmp/naca_results/")
