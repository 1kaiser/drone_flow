"""
Diamond airfoil at Mach 2 — adapted from JAXFLUIDS examples/examples_2D/12_diamond_airfoil
Viscous (mu=0.0005), 400x400 grid, end_time=1.0
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

DIAMOND_DIR = "/home/kaiser/projects/gi/JAXFLUIDS/examples/examples_2D/12_diamond_airfoil"
os.chdir(DIAMOND_DIR)

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jaxfluids import InputManager, InitializationManager, SimulationManager

# Patch save path
with open("diamond_airfoil.json") as f:
    case = json.load(f)
case["general"]["save_path"] = "/tmp/diamond_results"

patched = "/tmp/diamond_patched.json"
with open(patched, "w") as f:
    json.dump(case, f, indent=2)

# Run
input_manager        = InputManager(patched, "numerical_setup.json")
initialization_manager = InitializationManager(input_manager)
sim_manager          = SimulationManager(input_manager)

jxf_buffers = initialization_manager.initialization()
sim_manager.simulate(jxf_buffers)

# Load last snapshot from HDF5
import h5py

domain = sim_manager.output_writer.save_path_domain
files  = sorted(os.listdir(domain))
print(f"Snapshots: {len(files)}  — using last: {files[-1]}")

with h5py.File(f"{domain}/{files[-1]}") as h:
    t   = float(h["time"][()])
    x   = h["domain/gridX"][:]
    y   = h["domain/gridY"][:]
    rho = h["primitives/density"][0]
    p   = h["primitives/pressure"][0]
    T   = h["primitives/temperature"][0]
    Ma  = h["miscellaneous/mach_number"][0]
    sch = h["miscellaneous/schlieren"][0]
    vf  = h["levelset/volume_fraction"][0]

XX, YY = np.meshgrid(x, y)
solid  = vf <= 0.0

def mfield(arr, clip=None):
    f = np.ma.masked_where(solid, arr)
    if clip:
        f = np.clip(f, *clip)
    return f

fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="#0a0a14")
fig.suptitle(f"Diamond airfoil  Ma≈2  viscous  t={t:.2f}", color="white", fontsize=13)

panels = [
    (mfield(rho),          "inferno",  "Density  ρ"),
    (mfield(p),            "plasma",   "Pressure  p"),
    (mfield(Ma, (0, 3)),   "RdBu_r",   "Mach  (0–3)"),
    (mfield(sch, (1, 500)),"Greys_r",  "Schlieren"),
]

for ax, (field, cmap, title) in zip(axes, panels):
    im = ax.pcolormesh(XX, YY, field, cmap=cmap, shading="auto")
    ax.set_facecolor("#0a0a14")
    ax.set_title(title, color="white", fontsize=10)
    cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.08)
    cb.ax.xaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cb.ax.axes, "xticklabels"), color="white")
    ax.set_xlim(-1.0, 3.0); ax.set_ylim(-2.0, 2.0)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_color("white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("white")
    ax.set_xlabel("x", color="white"); ax.set_ylabel("y", color="white")

plt.tight_layout()
out = "/home/kaiser/projects/gi/drone_flow/diamond_result.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0a0a14")
print(f"Saved: {out}")
