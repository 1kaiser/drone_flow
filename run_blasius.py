"""
Laminar supersonic boundary layer (Blasius) — JAXFLUIDS notebooks/simulations/laminar_boundarylayer
200x100 grid, Sutherland viscosity, Ma≈2.25, isothermal wall
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

# Must run from the case dir so blasius_inlet.h5 is found
CASE_DIR = "/home/kaiser/projects/gi/JAXFLUIDS/notebooks/simulations/laminar_boundarylayer"
NUM_SETUP = "/home/kaiser/projects/gi/JAXFLUIDS/notebooks/simulations/numerical_setup_files/numerical_setup_singlephase.json"
os.chdir(CASE_DIR)

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py

from jaxfluids import InputManager, InitializationManager, SimulationManager

with open("case_setup.json") as f:
    case = json.load(f)
case["general"]["save_path"] = "/tmp/blasius_results"

patched = "/tmp/blasius_patched.json"
with open(patched, "w") as f:
    json.dump(case, f, indent=2)

# Patch numerical setup: old notebooks use underscore (CENTRAL6_ADAP) but
# installed build expects hyphen (CENTRAL6-ADAP)
with open(NUM_SETUP) as f:
    num_raw = f.read().replace("_ADAP", "-ADAP")
num_patched = "/tmp/blasius_numerical_setup.json"
with open(num_patched, "w") as f:
    f.write(num_raw)

input_manager          = InputManager(patched, num_patched)
initialization_manager = InitializationManager(input_manager)
sim_manager            = SimulationManager(input_manager)

jxf_buffers = initialization_manager.initialization()
sim_manager.simulate(jxf_buffers)

# Load last snapshot
domain = sim_manager.output_writer.save_path_domain
files  = sorted(os.listdir(domain))
print(f"Snapshots: {len(files)}  — using: {files[-1]}")

with h5py.File(f"{domain}/{files[-1]}") as h:
    t    = float(h["time"][()])
    x    = h["domain/gridX"][:]
    y    = h["domain/gridY"][:]
    rho  = h["primitives/density"][0]
    ux   = h["primitives/velocity"][0, :, :, 0]
    uy   = h["primitives/velocity"][0, :, :, 1]
    p    = h["primitives/pressure"][0]
    T    = h["primitives/temperature"][0]

XX, YY = np.meshgrid(x, y)

fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor="#0a0a14")
fig.suptitle(f"Laminar supersonic boundary layer (Blasius)  Ma≈2.25  t={t:.2f}",
             color="white", fontsize=13)

panels = [
    (ux,  "plasma",   "Streamwise velocity  u"),
    (T,   "inferno",  "Temperature  T"),
    (uy,  "RdBu_r",   "Wall-normal velocity  v"),
    (p,   "viridis",  "Pressure  p"),
]

for ax, (field, cmap, title) in zip(axes.flat, panels):
    im = ax.pcolormesh(XX, YY, field, cmap=cmap, shading="auto")
    ax.set_facecolor("#0a0a14")
    ax.set_title(title, color="white", fontsize=10)
    cb = plt.colorbar(im, ax=ax)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white")
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_color("white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("white")
    ax.set_xlabel("x", color="white")
    ax.set_ylabel("y", color="white")

plt.tight_layout()
out = "/home/kaiser/projects/gi/drone_flow/blasius_result.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0a0a14")
print(f"Saved: {out}")

# --- Blasius profile comparison at outlet ---
# Theoretical Blasius: u/u_inf = f'(eta), eta = y * sqrt(u_inf / (nu * x))
# Extract wall-normal profile at last x station
u_inf = float(ux[0, -1])   # freestream at outlet top
x_out = x[-1]
# rough estimate of BL thickness where u > 0.99 u_inf
bl_idx = np.where(ux[:, -1] < 0.99 * u_inf)[0]
delta  = float(y[bl_idx[-1]]) if len(bl_idx) else float(y[-1])
print(f"Outlet BL thickness δ ≈ {delta:.4f}  (x={x_out:.2f}, u_inf={u_inf:.4f})")
