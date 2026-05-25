# 🚁 drone_flow — Multi-Physics Drone Simulation

> **Fluid dynamics → Structural analysis → Flight dynamics → 3D browser visualization**

A complete end-to-end simulation of a quadcopter drone built on fixed, reproducible library versions:

| Library | Version | Role |
|---|---|---|
| [JAXFLUIDS](https://github.com/tumaer/JAXFLUIDS) | `2.0.0` | Compressible Navier-Stokes CFD (level-set immersed boundary) |
| [JAX](https://github.com/google/jax) | `0.9.2` | JIT-compiled 6-DOF dynamics + BEM propeller model |
| [scipy](https://scipy.org/) | `≥1.11` | Sparse HEX8 FEM stiffness assembly and direct solve |
| [Three.js](https://threejs.org/) | `0.165.0` | Browser-based 3D animation (CDN, no build step) |
| [jupytext](https://jupytext.readthedocs.io/) | `≥1.16` | `.py` ↔ `.ipynb` conversion for reproducible notebooks |
| [papermill](https://papermill.readthedocs.io/) | `≥2.6` | Parameterised notebook execution |

---

## 🗺️ Pipeline

```
🔧 Step 1  Drone Geometry
   build123d CAD  (240×170 mm fuselage, 4×262 mm arms, X-config)
   └─→ STEP → GLB  (Three.js model)

💨 Step 2  Fuselage CFD                          cfd/run_cfd.py
   JAXFLUIDS compressible NS, ellipse level-set
   Re=200, Ma≈0.3, end_time=60 s
   └─→ surface Cp at 4 arm roots  →  data/drone_arm_loads.json

🏗️ Step 3  Structural FEM                        fem/drone_arm_fem.py
   HEX8 cantilever (FR arm, Al 6061, 20×2×2 mesh)
   Loads: Cp × q_phys + motor weight
   └─→ tip deflection, von Mises, safety factor  →  data/drone_arm_fem_result.json

✈️  Step 4  6-DOF Flight Dynamics                 dynamics/ + run_dynamics.py
   Newton-Euler + BEM propeller (500 Hz, RK4)
   Cascaded PID — hover → climb → yaw → translate
   └─→ 20 s trajectory  →  assets/dynamics_result.png

🌐 Step 5  Browser Viewer                         viewer/index.html
   Pre-computed JSON + PNG frames  →  Three.js animation
   └─→ drone GLB · CFD frames · FEM stress · RPM panel
```

**Run the full pipeline:**
```bash
JAX_PLATFORMS=cpu conda run -n num_python python3 pipeline.py
JAX_PLATFORMS=cpu conda run -n num_python python3 pipeline.py --skip-cfd   # reuse cached CFD
JAX_PLATFORMS=cpu conda run -n num_python python3 pipeline.py --only fem   # single step
```

**Run as a notebook (jupytext + papermill):**
```bash
conda run -n num_python jupytext --to notebook notebooks/drone_pipeline.py --set-kernel python3
conda run -n num_python papermill notebooks/drone_pipeline.ipynb notebooks/drone_pipeline_out.ipynb
# override parameters:
conda run -n num_python papermill notebooks/drone_pipeline.ipynb out.ipynb -p SKIP_CFD True
```

---

## 🏁 Multi-Drone Model Comparison

Three airframes, same 20 s mission, same cascaded PID structure (attitude gains scaled by √(I/I_ref) to equalise closed-loop bandwidth).

```bash
JAX_PLATFORMS=cpu conda run -n num_python python3 run_comparison.py
```

### Dynamics Traces

![comparison dynamics](assets/comparison_dynamics.png)

### Motor RPM Profiles

![comparison rpm](assets/comparison_rpm.png)

### Step Response Summary

![comparison barplot](assets/comparison_barplot.png)

<details>
<summary>📊 Full Comparison Table</summary>

| Metric | Photography Drone | 5" Racing Drone | Heavy Lifter |
|---|---|---|---|
| **Mass** | 1.50 kg | 0.45 kg | 3.50 kg |
| **Prop size** | 8.0" (203 mm) | 5.0" (127 mm) | 13.0" (330 mm) |
| **CT / CQ** | 0.109 / 0.0095 | 0.105 / 0.0088 | 0.115 / 0.011 |
| **Arm span** | 254 mm | 120 mm | 380 mm |
| **Hover RPM** | 7 642 | 10 897 | 4 301 |
| **Max RPM (mission)** | 8 367 | 11 917 | 4 684 |
| **KP_RP (scaled)** | 8.00 | 2.55 | 18.26 |
| **KP_YAW (scaled)** | 3.00 | 1.09 | 5.54 |
| **Alt 0→2 m settle** | 4.11 s | 4.13 s | 4.00 s |
| **Alt 2→3 m settle** | 2.95 s | 2.95 s | 2.93 s |
| **Yaw 0→60° settle** | 3.44 s | 3.45 s | 3.43 s |
| **Max tilt** | 24.2° | 23.9° | 24.2° |
| **Max airspeed** | 3.86 m/s | 3.61 m/s | 5.26 m/s |

**Key findings:**
- Altitude and yaw settle times are nearly identical across all three airframes — the mass-compensated PID with inertia-scaled attitude gains achieves the same closed-loop bandwidth regardless of vehicle size.
- The racer operates at 43% higher RPM than the photography drone at hover; the lifter at 44% lower. Transient RPM swings are also 3× wider on the racer (narrow arms = smaller moment arm, so more RPM delta needed per unit of angular acceleration).
- Maximum airspeed differs: the heavier lifter's larger translational inertia produces more velocity overshoot during the horizontal translation phase.
- Without inertia scaling (same raw gains for all), the racer saturated its motor RPM limits during attitude corrections, causing unstable altitude tracking — demonstrating why gain scheduling or bandwidth matching is essential when porting a controller to a different airframe.

</details>

---

## 💨 Step 2 — CFD: Fuselage Flow

| Vorticity field | Surface pressure Cp |
|:---:|:---:|
| ![vorticity](assets/drone_vorticity_full.png) | ![surface Cp](assets/drone_surface_Cp.png) |

**Final flow frame — von Kármán vortex shedding, Re=200:**

![fuselage](assets/drone_final_frame.png)

<details>
<summary>📐 Geometry & Grid Setup</summary>

**Level-set ellipse** (matches 240×170 mm fuselage cross-section):
```python
"levelset": "lambda x,y: jnp.sqrt((x/0.706)**2 + (y/0.5)**2) - 1.0"
```

**Grid:** 300×180 cells, CONSTANT region 200×100 cells — **Δx=Δy=0.04 required** (square cells for level-set model).

**Physics:** Compressible Navier-Stokes, Re=200, Ma≈0.3, `end_time=60`.

**Arm root locations** (45° on ellipse surface):
```python
arm_roots = {
    "FL": ( 0.499,  0.354),   # (a·cos45°, b·sin45°)
    "FR": ( 0.499, -0.354),
    "RL": (-0.499,  0.354),
    "RR": (-0.499, -0.354),
}
```

**Output:**
```json
{
  "arm_Cp":      {"FL": -0.077, "FR": -0.175, "RL": -0.077, "RR": -0.175},
  "arm_pressure":{"FL":  0.9952,"FR":  0.9890,"RL":  0.9952,"RR":  0.9890}
}
```
</details>

---

## 🏗️ Step 3 — Structural FEM

![fem](assets/drone_arm_fem.png)

| Metric | Value |
|---|---|
| Tip deflection Δz | −0.0009 mm |
| Max von Mises stress | 0.022 MPa |
| Safety factor (Al 6061 σ_y=276 MPa) | **12,574×** ✅ |

<details>
<summary>🔩 Mesh, Material & Load Details</summary>

**Mesh:** 20×2×2 HEX8 elements → 189 nodes, 567 DOFs.

**Material:** Al 6061 — E=70 GPa, ν=0.3, σ_yield=276 MPa.

**Physical load scaling:**
```python
U_inf  = 10 m/s   →   q_phys = 0.5 × 1.225 × 10² = 61.25 Pa
F_aero = |Cp_FR| × q_phys × D × L = 0.175 × 61.25 × 0.032 × 0.262 = 0.090 N
F_grav = 0.08 × 9.81 = 0.785 N   (motor + ESC weight)
```

**Key finding:** At U=10 m/s gravity dominates; arm is extremely stiff (safety factor >12 000×). Aero loads become significant above ~40 m/s.
</details>

---

## ✈️ Step 4 — 6-DOF Flight Dynamics

![dynamics](assets/dynamics_result.png)

| Manoeuvre | Settle time (5%) |
|---|---|
| Altitude 0 → 2 m | 4.11 s |
| Altitude 2 → 3 m | 3.03 s |
| Yaw 0 → 60° | 2.17 s |
| Hover RPM | 7 642 RPM |

<details>
<summary>⚙️ State Vector, BEM Model & Mixer</summary>

**State (12):** `[px, py, pz, φ, θ, ψ, u, v, w, p, q, r]`

**BEM propeller:**
```python
T = CT · ρ · n² · D⁴    (CT=0.109, D=0.203 m)
Q = CQ · ρ · n² · D⁵    (CQ=0.0095)
```

**Motor layout (X-config, top view):**
```
FL (CCW) ─── FR (CW)
     \           /
      \         /
  RL (CW) ─── RR (CCW)
```

**Mixer — positive Mz (CCW yaw) requires speeding up CW motors:**
```python
T_FL = T/4 - Mx/(4L) + My/(4L) - Mz/(4r)   # CCW motor
T_FR = T/4 + Mx/(4L) + My/(4L) + Mz/(4r)   # CW  motor
T_RL = T/4 - Mx/(4L) - My/(4L) + Mz/(4r)   # CW  motor
T_RR = T/4 + Mx/(4L) - My/(4L) - Mz/(4r)   # CCW motor
```

**Controller gains:**
| Loop | Kp | Kd |
|---|---|---|
| Altitude | 2.5 | 4.5 |
| Roll/Pitch | 8.0 | 3.5 |
| Yaw | 3.0 | 2.0 |
| Position (→ tilt) | 0.25 | — |
</details>

---

## 🔬 JAXFLUIDS Validation Gallery

<table>
<tr>
  <td align="center"><b>NACA 0012 — Ma≈2, Re=∞</b><br><img src="assets/naca_result.png" width="340"/></td>
  <td align="center"><b>Diamond Airfoil — Ma≈2, viscous</b><br><img src="assets/diamond_result.png" width="340"/></td>
</tr>
<tr>
  <td align="center"><b>Bow Shock — Ma≈2, viscous</b><br><img src="assets/bowshock_result.png" width="340"/></td>
  <td align="center"><b>Double Mach Reflection — Ma=10</b><br><img src="assets/double_mach_result.png" width="340"/></td>
</tr>
<tr>
  <td align="center"><b>Rayleigh-Taylor Instability</b><br><img src="assets/rti_result.png" width="340"/></td>
  <td align="center"><b>Blasius Boundary Layer — Ma≈2.25</b><br><img src="assets/blasius_result.png" width="340"/></td>
</tr>
</table>

<details>
<summary>📋 Example Parameters Table</summary>

| Example | Physics | Grid | Wall time |
|---|---|---|---|
| `cylinder_flow` (drone) | Von Kármán shedding Re=200 | 300×180 | ~60 s sim |
| `NACA 0012` | Supersonic inviscid Ma≈2 | 256×128 | < 1 min |
| `diamond_airfoil` | Supersonic viscous Ma≈2 | 400×400 | 393 s |
| `bowshock` | Detached shock Ma≈2, viscous | 120×480 | 652 s |
| `double_mach_reflection` | Ma=10 shock at 60°, inviscid | 256×256 | 95 s |
| `rayleigh_taylor` | Heavy-over-light gravity instability | 64×256 | 48 s |
| `laminar_boundarylayer` | Blasius BL Ma≈2.25 | 200×100 | 534 s |

All run with `JAX_PLATFORMS=cpu conda run -n num_python python3 examples/run_<name>.py`
</details>

---

## 🌐 Step 5 — Browser Viewer

Three.js animation — drone GLB attitude, CFD vorticity panel, FEM stress bar, 4 motor RPMs.

```bash
# Export viewer data
JAX_PLATFORMS=cpu conda run -n num_python python3 export_all.py

# Serve from parent directory (so /cad-power-animations/ is reachable)
cd /path/to/gi
python3 -m http.server 7800
# open http://localhost:7800/drone_flow/viewer/
```

---

## 📂 Repository Structure

```
drone_flow/
│
├── 🗒️  pipeline.py               Master script — runs all 4 steps
├── 🗒️  export_all.py             Pre-compute viewer data (trajectory + CFD frames + FEM)
├── 📒  run_dynamics.py           Standalone dynamics test + 6-panel plot
│
├── 💨  cfd/
│   ├── drone_flow.json           JAXFLUIDS case config (ellipse level-set)
│   ├── numerical_setup.json      JAXFLUIDS numerical schemes
│   └── run_cfd.py                CFD runner — exposes run() for pipeline
│
├── 🏗️  fem/
│   └── drone_arm_fem.py          HEX8 FEM runner — exposes run() for pipeline
│
├── ✈️  dynamics/
│   ├── propeller.py              BEM: T=CT·ρ·n²·D⁴, Q=CQ·ρ·n²·D⁵
│   ├── dynamics.py               Newton-Euler 6-DOF + RK4
│   └── controller.py             Cascaded PD: altitude + attitude + mixer
│
├── 🔬  examples/
│   ├── run_naca.py               NACA 0012 Ma≈2
│   ├── run_diamond.py            Diamond airfoil Ma≈2
│   ├── run_bowshock.py           Detached bow shock
│   ├── run_double_mach.py        Double Mach reflection Ma=10
│   ├── run_blasius.py            Laminar boundary layer
│   └── run_rti.py                Rayleigh-Taylor instability
│
├── 📓  notebooks/
│   └── drone_pipeline.py         Jupytext percent-format notebook (papermill-ready)
│
├── 🌐  viewer/
│   ├── index.html                Three.js animation viewer
│   ├── trajectory.json           6-DOF states @ 30 fps (625 frames)
│   ├── fem_data.json             Arm stress samples (63 points)
│   ├── manifest.json             Frame metadata
│   └── cfd_frames/               PNG sequence ¹
│
├── 🖼️  assets/                   All result figures (scripts write here)
│   ├── drone_final_frame.png
│   ├── drone_vorticity_full.png
│   ├── drone_surface_Cp.png
│   ├── drone_arm_fem.png
│   ├── dynamics_result.png
│   ├── naca_result.png  · diamond_result.png  · bowshock_result.png
│   ├── double_mach_result.png  · rti_result.png  · blasius_result.png
│
├── 📁  data/                     Small JSON outputs (committed)
│   ├── drone_arm_loads.json      CFD → Cp at 4 arm roots
│   └── drone_arm_fem_result.json FEM → displacements + von Mises
│
└── 📁  results/                  JAXFLUIDS HDF5 output ¹ (gitignored, ~93 MB)

¹ gitignored — regenerated by running the pipeline
```

---

## ⚙️ Installation & Reproducibility

```bash
# Fixed-version install for reproducible results
conda activate num_python    # Python 3.13

pip install "jax[cpu]==0.9.2" jaxfluids==2.0.0
pip install "scipy>=1.11" h5py matplotlib
pip install "jupytext>=1.16" "papermill>=2.6"
```

Three.js `0.165.0` is pinned via CDN importmap in `viewer/index.html` — no npm required.

---

## 🐛 Key Bugs Fixed

<details>
<summary>Show all 5 fixes</summary>

| Bug | Root cause | Fix |
|---|---|---|
| **JAXFLUIDS cell aspect ratio** | Level-set model requires Δx=Δy (square cells) in the fine region. | Computed CONSTANT region cell count so 200 cells / 8 units = 100 cells / 4 units = 0.04. |
| **Yaw mixer sign inversion** | Speeding up CCW motors (FL/RR) for positive Mz was wrong. CCW props create CW reaction torque on airframe — Newton's 3rd law. | Speed up CW motors (FR/RL) for CCW airframe rotation. |
| **JAXFLUIDS stencil names** | Old example configs use `CENTRAL6_ADAP` (underscore); installed v2 requires `CENTRAL6-ADAP` (hyphen). | Patch config JSON at runtime: `raw.replace("_ADAP", "-ADAP")`. |
| **HDF5 vorticity shape** | 2D field stored as `(1, Ny, Nx, 1)` — z-vorticity is the last index, not index 2. | Index as `h["miscellaneous/vorticity"][0, :, :, 0]`. |
| **Altitude runaway** | Hard step setpoints → massive integral windup, z overshoot to 10 m. | Clamp altitude error to ±1 m; replace steps with smooth velocity ramps. |

</details>
