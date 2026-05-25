"""
Double Mach reflection — JAXFLUIDS examples_2D/08_double_mach_reflection
256x256 inviscid, Mach 10 shock at 60°, end_time=0.28
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

CASE_DIR = "/home/kaiser/projects/gi/JAXFLUIDS/examples/examples_2D/08_double_mach_reflection"
os.chdir(CASE_DIR)

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py

from jaxfluids import InputManager, InitializationManager, SimulationManager

with open("double_mach_reflection.json") as f:
    case = json.load(f)
case["general"]["save_path"] = "/tmp/dmr_results"
case["general"]["save_dt"]   = 0.14   # just 2 snapshots + t=0

patched = "/tmp/dmr_patched.json"
with open(patched, "w") as f:
    json.dump(case, f, indent=2)

input_manager          = InputManager(patched, "numerical_setup.json")
initialization_manager = InitializationManager(input_manager)
sim_manager            = SimulationManager(input_manager)

jxf_buffers = initialization_manager.initialization()
sim_manager.simulate(jxf_buffers)

# Load last snapshot
domain = sim_manager.output_writer.save_path_domain
files  = sorted(os.listdir(domain))
print(f"Snapshots: {len(files)}  — using: {files[-1]}")

with h5py.File(f"{domain}/{files[-1]}") as h:
    t   = float(h["time"][()])
    x   = h["domain/gridX"][:]
    y   = h["domain/gridY"][:]
    rho = h["primitives/density"][0]
    p   = h["primitives/pressure"][0]
    Ma  = h["miscellaneous/mach_number"][0]
    sch = h["miscellaneous/schlieren"][0]

XX, YY = np.meshgrid(x, y)

fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="#0a0a14")
fig.suptitle(f"Double Mach reflection  Ma=10  60°  t={t:.3f}", color="white", fontsize=13)

panels = [
    (rho,              "inferno",  "Density  ρ",              None),
    (p,                "plasma",   "Pressure  p",             None),
    (np.clip(Ma,0,12), "RdBu_r",   "Mach  (0–12)",            None),
    (np.clip(sch,1,1e4),"Greys_r", "Schlieren",               None),
]

for ax, (field, cmap, title, _) in zip(axes.flat, panels):
    im = ax.pcolormesh(XX, YY, field, cmap=cmap, shading="auto")
    ax.set_facecolor("#0a0a14")
    ax.set_title(title, color="white", fontsize=10)
    cb = plt.colorbar(im, ax=ax)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white")
    ax.set_xlim(0, 4); ax.set_ylim(0, 2)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_color("white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("white")
    ax.set_xlabel("x", color="white")
    ax.set_ylabel("y", color="white")

plt.tight_layout()
out = "/home/kaiser/projects/gi/drone_flow/assets/double_mach_result.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0a0a14")
print(f"Saved: {out}")
