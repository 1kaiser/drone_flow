"""
Rayleigh-Taylor instability — JAXFLUIDS examples_2D/04_rayleigh_taylor_instability
64x256 viscous, gravity forcing, heavy-over-light fluid, end_time=1.95
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

CASE_DIR = "/home/kaiser/projects/gi/JAXFLUIDS/examples/examples_2D/04_rayleigh_taylor_instability"
os.chdir(CASE_DIR)

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py

from jaxfluids import InputManager, InitializationManager, SimulationManager

with open("rti.json") as f:
    case = json.load(f)
case["general"]["save_path"] = "/tmp/rti_results"
case["general"]["save_dt"]   = 0.65   # 3 snapshots

patched = "/tmp/rti_patched.json"
with open(patched, "w") as f:
    json.dump(case, f, indent=2)

input_manager          = InputManager(patched, "numerical_setup.json")
initialization_manager = InitializationManager(input_manager)
sim_manager            = SimulationManager(input_manager)

jxf_buffers = initialization_manager.initialization()
sim_manager.simulate(jxf_buffers)

# Load all snapshots for a multi-panel time evolution
domain = sim_manager.output_writer.save_path_domain
files  = sorted(os.listdir(domain))
print(f"Snapshots: {len(files)}")

snapshots = []
for fn in files:
    with h5py.File(f"{domain}/{fn}") as h:
        snapshots.append({
            "t":   float(h["time"][()]),
            "x":   h["domain/gridX"][:],
            "y":   h["domain/gridY"][:],
            "rho": h["primitives/density"][0],
            "vort":h["miscellaneous/vorticity"][0, 2],   # z-component
        })

# Plot density evolution across all times
n = len(snapshots)
fig, axes = plt.subplots(2, n, figsize=(5*n, 10), facecolor="#0a0a14")
fig.suptitle("Rayleigh-Taylor instability  ρ_heavy/ρ_light=2  viscous", color="white", fontsize=13)

for col, snap in enumerate(snapshots):
    XX, YY = np.meshgrid(snap["x"], snap["y"])
    vmax_v = np.percentile(np.abs(snap["vort"]), 98)

    for row, (field, cmap, title) in enumerate([
        (snap["rho"],  "RdBu_r", f"Density  t={snap['t']:.2f}"),
        (snap["vort"], "seismic", f"Vorticity  t={snap['t']:.2f}"),
    ]):
        ax = axes[row, col]
        if row == 1:
            im = ax.pcolormesh(XX, YY, field, cmap=cmap, shading="auto",
                               vmin=-vmax_v, vmax=vmax_v)
        else:
            im = ax.pcolormesh(XX, YY, field, cmap=cmap, shading="auto")
        ax.set_facecolor("#0a0a14")
        ax.set_title(title, color="white", fontsize=9)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white")
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_color("white")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("white")
        ax.set_xlabel("x", color="white")
        ax.set_ylabel("y", color="white")
        ax.set_aspect("equal")

plt.tight_layout()
out = "/home/kaiser/projects/gi/drone_flow/rti_result.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0a0a14")
print(f"Saved: {out}")
