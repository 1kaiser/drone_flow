"""
Pre-compute all data for the drone viewer animation.
Outputs to drone_flow/viewer/:
  trajectory.json   — 6-DOF states at 30 fps
  fem_data.json     — arm stress sampled along trajectory
  cfd_frames/       — PNG pressure+vorticity frames from JAXFLUIDS h5 files
  manifest.json     — frame count, FPS, metadata
"""
import os, sys, json
os.environ["JAX_PLATFORMS"] = "cpu"
from pathlib import Path
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import jax, jax.numpy as jnp
import numpy as np
import h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = REPO / "viewer"
CFD = OUT / "cfd_frames"
CFD.mkdir(parents=True, exist_ok=True)

# ─── 1. TRAJECTORY (multicopter_jax: kT/kQ measurement fit, quaternion dynamics)
print("=== 1. Running 6-DOF trajectory (multicopter_jax) ===")
from dynamics.jax_copter import run as run_jax_copter
_dyn = run_jax_copter(
    T_end=20.0, dt=1.0 / 500.,
    out_json=OUT / "trajectory.json",
    verbose=True,
)
print(f"  hover_rpm={_dyn['hover_rpm']:.0f}  settle={_dyn['settle_time_s']:.2f}s")


# ─── 2. FEM ARM STRESS along trajectory (sampled every 10 output frames) ─────
print("=== 2. FEM arm stress ===")

# Load trajectory frames from JSON written by jax_copter
with open(OUT / "trajectory.json") as f_traj:
    traj_data = json.load(f_traj)
frames = traj_data["frames"]

# kT from measurement-fit motor model
_kT = float(_dyn["params"]["kT"][0])

# Physical scaling
RHO_PHYS = 1.225        # kg/m³

with open(REPO / "data" / "drone_arm_loads.json") as f:
    cfd_loads = json.load(f)

Cp_FR = cfd_loads["arm_Cp"]["FR"]   # -0.175 at hover

# Simple beam formula: tip deflection = FL³/(3EI)
ARM_L = 0.262   # m
ARM_D = 0.032   # m
E_AL  = 70e9    # Pa
I_sec = (ARM_D**4) / 64 * 3.14159  # circular tube approx

fem_samples = []
for i, fr in enumerate(frames):
    if i % 10 != 0:
        continue
    # Motor thrust at this frame (average of 4 motors)
    rpm_avg = sum(fr["rpm"]) / 4
    omega_avg = rpm_avg * 2 * 3.14159 / 60
    T_motor = float(_kT * omega_avg**2)

    # Airspeed magnitude
    airspeed = float(np.linalg.norm(fr["vel"]))
    q_local  = max(0.5 * RHO_PHYS * airspeed**2, 0.01)
    F_aero   = abs(Cp_FR) * q_local * ARM_D * ARM_L

    # Gravity load (motor weight ~80g)
    F_grav = 0.08 * 9.81

    # Tip deflection (bending)
    F_total = F_aero + F_grav
    delta_tip = F_total * ARM_L**3 / (3 * E_AL * I_sec)

    # Von Mises at root (bending stress = M*c/I, M = F*L, c = D/2)
    M_root = F_total * ARM_L
    sigma  = M_root * (ARM_D/2) / I_sec
    sigma_vm = sigma / 1e6   # MPa

    fem_samples.append({
        "frame_idx": i,
        "t": fr["t"],
        "T_motor_N": round(T_motor, 3),
        "F_aero_N":  round(F_aero, 4),
        "delta_tip_mm": round(delta_tip*1000, 4),
        "vonMises_MPa": round(sigma_vm, 4),
        "safety_factor": round(276.0 / max(sigma_vm, 0.001), 1),
    })

fem_out = {"arm_length_m": ARM_L, "material": "Al6061", "yield_MPa": 276, "samples": fem_samples}
with open(OUT / "fem_data.json", "w") as f:
    json.dump(fem_out, f, indent=2)
print(f"  fem_data.json: {len(fem_samples)} samples")


# ─── 3. CFD FRAMES from JAXFLUIDS h5 files ───────────────────────────────────
print("=== 3. Exporting CFD frames ===")
H5_DIR = REPO / "results" / "drone_fuselage_Re200" / "domain"
h5_files = sorted(H5_DIR.glob("*.h5"))
print(f"  Found {len(h5_files)} h5 snapshots")

cfd_meta = []
for idx, h5path in enumerate(h5_files):
    with h5py.File(h5path) as h:
        t_cfd = float(h["time"][()])
        x     = h["domain/gridX"][:]
        y     = h["domain/gridY"][:]
        pres  = h["primitives/pressure"][0]          # (Ny, Nx)
        vort  = h["miscellaneous/vorticity"][0, :, :, 0]  # (Ny, Nx)
        ls    = h["levelset/levelset"][0]           # (Ny, Nx)

    XX, YY = np.meshgrid(x, y)
    solid  = ls < 0

    def mf(arr, clip=None):
        f = np.ma.masked_where(solid, arr)
        if clip: f = np.clip(f, *clip)
        return f

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#0a0a14")
    vmax = float(np.percentile(np.abs(mf(vort).compressed()), 97)) if mf(vort).count() > 0 else 1.0

    im0 = axes[0].pcolormesh(XX, YY, mf(vort), cmap="RdBu_r",
                              vmin=-vmax, vmax=vmax, shading="auto")
    axes[0].set_facecolor("#0a0a14"); axes[0].set_title(f"Vorticity  t={t_cfd:.1f}", color="white", fontsize=9)

    im1 = axes[1].pcolormesh(XX, YY, mf(pres), cmap="plasma", shading="auto")
    axes[1].set_facecolor("#0a0a14"); axes[1].set_title("Pressure  p", color="white", fontsize=9)

    # Drone ellipse outline
    th = np.linspace(0, 2*np.pi, 200)
    for ax in axes:
        ax.plot(0.706*np.cos(th), 0.5*np.sin(th), "w-", lw=1, alpha=0.6)
        ax.set_xlim(-3, 6); ax.set_ylim(-3, 3)
        ax.set_facecolor("#0a0a14")
        ax.tick_params(colors="white"); ax.spines[:].set_color("#444")
        for tl in ax.get_xticklabels()+ax.get_yticklabels(): tl.set_color("white")

    plt.tight_layout(pad=0.5)
    fname = f"frame_{idx:03d}.png"
    plt.savefig(CFD / fname, dpi=90, bbox_inches="tight", facecolor="#0a0a14")
    plt.close()
    cfd_meta.append({"idx": idx, "t_cfd": round(t_cfd, 3), "file": fname})
    if idx % 5 == 0:
        print(f"  frame {idx:02d}/{len(h5_files)}  t={t_cfd:.1f}")

print(f"  Saved {len(h5_files)} CFD frames to viewer/cfd_frames/")


# ─── 4. MANIFEST ─────────────────────────────────────────────────────────────
manifest = {
    "trajectory": {"fps": FPS, "n_frames": len(frames), "t_end": T_END},
    "cfd": {"n_frames": len(h5_files), "frames": cfd_meta},
    "fem": {"n_samples": len(fem_samples)},
    "drone_glb": "../../../cad-power-animations/models/drone/.drone.step.glb",
}
with open(OUT / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print(f"\nAll outputs in: {OUT}")
print("  trajectory.json, fem_data.json, cfd_frames/, manifest.json")
