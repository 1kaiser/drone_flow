# %% [markdown]
# # 🚁 drone_flow — Multi-Physics Drone Simulation Notebook
#
# End-to-end pipeline:
# **CFD (JAXFLUIDS)** → **FEM (HEX8)** → **6-DOF Dynamics** → **Three.js Viewer**
#
# ---
# Convert to notebook and execute:
# ```bash
# conda run -n num_python jupytext --to notebook notebooks/drone_pipeline.py --set-kernel python3
# conda run -n num_python papermill notebooks/drone_pipeline.ipynb notebooks/drone_pipeline_out.ipynb
# ```

# %% tags=["parameters"]
# --- Papermill parameters (override from CLI with -p key value) ---
SKIP_CFD    = True    # True = use cached data/drone_arm_loads.json
RUN_FEM     = True
RUN_DYNAMICS = True
RUN_EXPORT  = False   # export_all.py needs JAXFLUIDS h5 results directory

# %% [markdown]
# ## 📦 Setup

# %%
import os, sys, json
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

REPO = Path("..").resolve()   # notebooks/ → drone_flow/
sys.path.insert(0, str(REPO))
print(f"REPO: {REPO}")
print(f"Python: {sys.version.split()[0]}")

# %% [markdown]
# ---
# ## 💨 Step 1 — CFD: Drone Fuselage Flow (JAXFLUIDS)
#
# **Geometry:** 2D ellipse level-set (a=0.706, b=0.5) matching 240×170 mm fuselage cross-section.
# **Physics:** Compressible Navier-Stokes, Re=200, Ma≈0.3, end_time=60.
# **Output:** Surface Cp at 4 arm roots → `data/drone_arm_loads.json`

# %%
cfd_loads_path = REPO / "data" / "drone_arm_loads.json"

if not SKIP_CFD:
    print("Running JAXFLUIDS CFD simulation (~60 s sim time)...")
    from cfd.run_cfd import run as run_cfd
    cfd_result = run_cfd(
        out_json=cfd_loads_path,
        out_fig_prefix=REPO / "assets" / "drone",
    )
else:
    print(f"Loading cached: {cfd_loads_path}")
    with open(cfd_loads_path) as f:
        cfd_result = json.load(f)

print("\nArm Cp values:")
for arm, cp in cfd_result["arm_Cp"].items():
    print(f"  {arm}: Cp = {cp:.4f}")

# %% [markdown]
# ### 🌀 CFD Flow Visualisation

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="#0a0a14")
fig.suptitle("JAXFLUIDS CFD — Drone Fuselage Re=200", color="white", fontsize=14)

imgs = [
    ("Vorticity field",      REPO / "assets" / "drone_vorticity_full.png"),
    ("Surface pressure Cp",  REPO / "assets" / "drone_surface_Cp.png"),
    ("Final vortex frame",   REPO / "assets" / "drone_final_frame.png"),
]
for ax, (title, path) in zip(axes, imgs):
    if path.exists():
        ax.imshow(mpimg.imread(str(path)))
    else:
        ax.text(0.5, 0.5, f"Run CFD to generate:\n{path.name}",
                ha="center", va="center", color="white", transform=ax.transAxes)
    ax.set_title(title, color="white", fontsize=10)
    ax.axis("off"); ax.set_facecolor("#0a0a14")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 📊 Arm Load Summary

# %%
import pandas as pd
rows = []
for arm in ["FL", "FR", "RL", "RR"]:
    cp = cfd_result["arm_Cp"][arm]
    p  = cfd_result["arm_pressure"][arm]
    q_phys = 0.5 * 1.225 * 10.0**2
    f_aero = abs(cp) * q_phys * 0.032 * 0.262
    rows.append({"Arm": arm, "Cp": f"{cp:.4f}", "p (nondim)": f"{p:.4f}",
                 "F_aero @10m/s (N)": f"{f_aero:.4f}"})
df_cfd = pd.DataFrame(rows)
print(df_cfd.to_string(index=False))

# %% [markdown]
# ---
# ## 🏗️ Step 2 — FEM: Drone Arm Structural Analysis

# %%
fem_result_path = REPO / "data" / "drone_arm_fem_result.json"

if RUN_FEM:
    print("Running HEX8 FEM...")
    from fem.drone_arm_fem import run as run_fem
    fem_result = run_fem(
        in_json  = cfd_loads_path,
        out_json = fem_result_path,
        out_fig  = REPO / "assets" / "drone_arm_fem.png",
    )
else:
    with open(fem_result_path) as f:
        fem_result = json.load(f)
    print("Loaded cached FEM result")

# %% [markdown]
# ### 📊 FEM Results

# %%
fem_fig = REPO / "assets" / "drone_arm_fem.png"
if fem_fig.exists():
    fig, ax = plt.subplots(figsize=(16, 5), facecolor="#0a0a14")
    ax.imshow(mpimg.imread(str(fem_fig))); ax.axis("off")
    ax.set_facecolor("#0a0a14")
    fig.suptitle("HEX8 FEM — FR Arm Structural Analysis", color="white", fontsize=13)
    plt.tight_layout(); plt.show()

rows_fem = [
    ["Arm",              "FR"],
    ["Material",         "Al 6061  (E=70 GPa, σ_yield=276 MPa)"],
    ["Mesh",             "20×2×2 HEX8  (189 nodes, 567 DOFs)"],
    ["F_aero",           f"{fem_result['F_aero_N']:.4f} N"],
    ["F_motor (gravity)",f"{abs(fem_result['F_motor_N']):.3f} N"],
    ["Tip Δy (aero)",    f"{fem_result['tip_displacement_mm']['y']:.4f} mm"],
    ["Tip Δz (gravity)", f"{fem_result['tip_displacement_mm']['z']:.4f} mm"],
    ["Max von Mises",    f"{fem_result['max_vonMises_MPa']:.4f} MPa"],
    ["Safety factor",    f"{fem_result['yield_safety_factor']:.0f}×  ✅"],
]
df_fem = pd.DataFrame(rows_fem, columns=["Parameter", "Value"])
print(df_fem.to_string(index=False))

# %% [markdown]
# ---
# ## ✈️ Step 3 — 6-DOF Flight Dynamics (500 Hz)

# %%
if RUN_DYNAMICS:
    print("Running 6-DOF dynamics (20 s at 500 Hz)...")
    import jax, jax.numpy as jnp
    from dynamics.propeller  import hover_omega, omega_to_rpm
    from dynamics.dynamics   import MASS, G, rk4
    from dynamics.controller import control

    rk4_jit  = jax.jit(rk4)
    ctrl_jit = jax.jit(control)

    DT = 1/500.; T_END = 20.; N = int(T_END / DT)
    state = jnp.zeros(12)

    def ramp(t, t0, z0, z1, v=0.8):
        dt = jnp.maximum(t - t0, 0.)
        return z0 + jnp.clip(dt*v/jnp.maximum(jnp.abs(z1-z0),1e-3), 0., 1.)*(z1-z0)

    def setpoint(t):
        z_d   = jnp.where(t<1., 0., jnp.where(t<10., ramp(t,1.,0.,2.), ramp(t,10.,2.,3.)))
        px_d  = jnp.where(t<9., 0., ramp(t,9.,0.,4.,0.4))
        py_d  = jnp.where(t<13., 0., ramp(t,13.,0.,2.,0.3))
        psi_d = jnp.where(t<6., 0., ramp(t,6.,0.,jnp.radians(60.),0.25))
        return jnp.array([px_d, py_d, z_d, psi_d])

    log_s = np.zeros((N+1,12)); log_m = np.zeros((N+1,4)); log_t = np.zeros(N+1)
    log_s[0] = np.array(state)
    for i in range(N):
        t = i*DT; sp = setpoint(t); mot = ctrl_jit(state, sp)
        state = rk4_jit(state, mot, DT)
        log_s[i+1] = np.array(state); log_m[i+1] = np.array(mot); log_t[i+1] = (i+1)*DT

    RPM = omega_to_rpm(log_m)
    print(f"Hover RPM: {float(omega_to_rpm(hover_omega(MASS))):.0f}")

# %% [markdown]
# ### 📊 Step Response Summary

# %%
def settle_time(t_arr, y_arr, t_step, target, band=0.05):
    mask = (t_arr > t_step) & (np.abs(y_arr - target) < band * max(abs(target), 0.1))
    return float(t_arr[mask][0] - t_step) if mask.any() else float("nan")

if RUN_DYNAMICS:
    rows_dyn = [
        ["Altitude 0→2 m  (5%)", f"{settle_time(log_t, log_s[:,2], 3.5, 2.0):.2f} s"],
        ["Altitude 2→3 m  (5%)", f"{settle_time(log_t, log_s[:,2], 11.25, 3.0):.2f} s"],
        ["Yaw 0→60°  (5%)",      f"{settle_time(log_t, np.degrees(log_s[:,5]), 7.5, 60.0):.2f} s"],
        ["Hover RPM",            f"{float(omega_to_rpm(hover_omega(MASS))):.0f}"],
    ]
    df_dyn = pd.DataFrame(rows_dyn, columns=["Manoeuvre", "Result"])
    print(df_dyn.to_string(index=False))

dyn_fig = REPO / "assets" / "dynamics_result.png"
if dyn_fig.exists():
    fig, ax = plt.subplots(figsize=(15, 8), facecolor="#0a0a14")
    ax.imshow(mpimg.imread(str(dyn_fig))); ax.axis("off"); ax.set_facecolor("#0a0a14")
    fig.suptitle("6-DOF State Response — 20 s Mission", color="white", fontsize=13)
    plt.tight_layout(); plt.show()

# %% [markdown]
# ---
# ## 🔬 JAXFLUIDS Validation Gallery

# %%
gallery = [
    ("NACA 0012 Ma≈2",            "naca_result.png"),
    ("Diamond Airfoil Ma≈2",      "diamond_result.png"),
    ("Bow Shock Ma≈2",            "bowshock_result.png"),
    ("Double Mach Reflection Ma=10","double_mach_result.png"),
    ("Rayleigh-Taylor Instability","rti_result.png"),
    ("Blasius Boundary Layer",    "blasius_result.png"),
]
fig, axes = plt.subplots(2, 3, figsize=(18, 9), facecolor="#0a0a14")
fig.suptitle("JAXFLUIDS 2D Example Gallery", color="white", fontsize=14)
for ax, (title, fname) in zip(axes.flat, gallery):
    p = REPO / "assets" / fname
    if p.exists():
        ax.imshow(mpimg.imread(str(p)))
    else:
        ax.text(0.5, 0.5, f"Run examples/run_{fname.split('_')[0]}.py",
                ha="center", va="center", color="white", transform=ax.transAxes, fontsize=8)
    ax.set_title(title, color="white", fontsize=9); ax.axis("off"); ax.set_facecolor("#0a0a14")
plt.tight_layout(); plt.show()

# %% [markdown]
# ---
# ## 🌐 Step 4 — Browser Viewer
#
# After running `export_all.py`, serve from the parent directory:
#
# ```bash
# python3 -m http.server 7800
# # open http://localhost:7800/drone_flow/viewer/
# ```
#
# The viewer shows:
# - 🚁 Drone GLB with 6-DOF attitude animation
# - 💨 CFD vorticity/pressure panel (31 frames, t=0→60)
# - 🏗️ FEM arm stress (von Mises, tip deflection, safety factor)
# - ⏱️ Timeline scrubber + play/pause

# %%
if RUN_EXPORT:
    import importlib.util, importlib
    spec = importlib.util.spec_from_file_location("export_all", REPO / "export_all.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print("Export complete → viewer/ ready")
else:
    print("Export skipped (set RUN_EXPORT=True to regenerate viewer data)")

# %% [markdown]
# ---
# ## ✅ Pipeline Summary

# %%
print("🚁 drone_flow — Pipeline Complete")
print(f"   REPO: {REPO}")
print()
print("  Outputs:")
print(f"    data/drone_arm_loads.json    — CFD arm Cp")
print(f"    data/drone_arm_fem_result.json — HEX8 FEM displacements + von Mises")
print(f"    assets/                      — all result figures")
print(f"    viewer/                      — Three.js animation data")
