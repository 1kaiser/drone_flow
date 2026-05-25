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
SKIP_CFD      = True    # True = use cached data/drone_arm_loads.json
RUN_FEM       = True
RUN_DYNAMICS  = True
RUN_COMPARISON = True   # multi-drone comparison (fast, ~18 s)
RUN_EXPORT    = False   # export_all.py needs JAXFLUIDS h5 results directory

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
# ## 🏁 Multi-Drone Model Comparison
#
# Same 20 s mission on three airframes.
# Attitude gains are scaled by **√(I / I_ref)** so closed-loop bandwidth is equal,
# making the settle-time comparison fair across very different sizes.

# %%
# ── Drone configs ─────────────────────────────────────────────────────────────
RHO_CMP = 1.225; G_CMP = 9.81; PI2_CMP = 2.0 * np.pi

DRONES_CMP = [
    dict(name="Photography Drone", label="photo",  color="#00d4ff",
         mass=1.5,  D=0.203, CT=0.109, CQ=0.0095, L=0.2545,
         Ixx=0.0196, Iyy=0.0196, Izz=0.0264, Kd=0.25),
    dict(name='5" Racing Drone',   label="racer",  color="#ff6b35",
         mass=0.45, D=0.127, CT=0.105, CQ=0.0088, L=0.120,
         Ixx=0.0020, Iyy=0.0020, Izz=0.0035, Kd=0.08),
    dict(name="Heavy Lifter",      label="lifter", color="#7fff00",
         mass=3.5,  D=0.330, CT=0.115, CQ=0.011,  L=0.380,
         Ixx=0.0650, Iyy=0.0650, Izz=0.0900, Kd=0.30),
]

# Reference gains (Photography Drone)
IXX_REF_CMP = 0.0196; IZZ_REF_CMP = 0.0264
KP_Z_CMP = 2.5; KD_Z_CMP = 4.5; KP_XY_CMP = 0.25
MAX_TILT_CMP = np.radians(25.0)
KP_RP_REF = 8.0; KD_RP_REF = 3.5; KP_YAW_REF = 3.0; KD_YAW_REF = 2.0

DT_CMP = 1/500.; T_END_CMP = 20.; N_CMP = int(T_END_CMP / DT_CMP)

# %%
import jax, jax.numpy as jnp

def _ramp_cmp(t, t0, z0, z1, v=0.8):
    dt = jnp.maximum(t - t0, 0.)
    return z0 + jnp.clip(dt * v / jnp.maximum(jnp.abs(z1-z0), 1e-3), 0., 1.) * (z1-z0)

def _setpoint_cmp(t):
    z_d   = jnp.where(t<1., 0., jnp.where(t<10., _ramp_cmp(t,1.,0.,2.), _ramp_cmp(t,10.,2.,3.)))
    px_d  = jnp.where(t<9.,  0., _ramp_cmp(t,9.,0.,4.,0.4))
    py_d  = jnp.where(t<13., 0., _ramp_cmp(t,13.,0.,2.,0.3))
    psi_d = jnp.where(t<6.,  0., _ramp_cmp(t,6.,0.,jnp.radians(60.),0.25))
    return jnp.array([px_d, py_d, z_d, psi_d])

def _make_sim_cmp(cfg):
    mass=cfg["mass"]; D=cfg["D"]; CT=cfg["CT"]; CQ=cfg["CQ"]
    L=cfg["L"]; Ixx=cfg["Ixx"]; Iyy=cfg["Iyy"]; Izz=cfg["Izz"]; Kd=cfg["Kd"]
    kT = CT*RHO_CMP*D**4/PI2_CMP**2; kQ = CQ*RHO_CMP*D**5/PI2_CMP**2
    r_ = kQ/kT; I_inv = jnp.array([1/Ixx,1/Iyy,1/Izz])
    att = np.sqrt(Ixx/IXX_REF_CMP); yaw = np.sqrt(Izz/IZZ_REF_CMP)
    KP_RP=KP_RP_REF*att; KD_RP=KD_RP_REF*att
    KP_YAW=KP_YAW_REF*yaw; KD_YAW=KD_YAW_REF*yaw
    OMAX = 3000.0

    def _T2w(T): return jnp.sqrt(jnp.maximum(T,0.)/kT)
    def _mix(Tt,Mx,My,Mz):
        a=Tt/4; b=Mx/(4*L); c=My/(4*L); d=Mz/(4*r_)
        return jnp.clip(jnp.array([_T2w(a-b+c-d),_T2w(a+b+c+d),
                                    _T2w(a-b-c+d),_T2w(a+b-c-d)]),50.,OMAX)
    def _R(phi,th,psi):
        cp,sp=jnp.cos(phi),jnp.sin(phi); ct,st=jnp.cos(th),jnp.sin(th)
        cy,sy=jnp.cos(psi),jnp.sin(psi)
        return jnp.array([[cy*ct,cy*st*sp-sy*cp,cy*st*cp+sy*sp],
                           [sy*ct,sy*st*sp+cy*cp,sy*st*cp-cy*sp],
                           [-st,  ct*sp,          ct*cp         ]])
    def _ek(phi,th,p,q,r):
        cp,sp=jnp.cos(phi),jnp.sin(phi); ct,st=jnp.cos(th),jnp.sin(th); tt=st/ct
        return p+sp*tt*q+cp*tt*r, cp*q-sp*r, (sp/ct)*q+(cp/ct)*r

    def ctrl(state,sp):
        px,py,pz,phi,th,psi,u,v,w,p,q,r=state; px_d,py_d,pz_d,psi_d=sp
        R=_R(phi,th,psi); vel_w=R@jnp.array([u,v,w]); vz_w=vel_w[2]
        ez=jnp.clip(pz_d-pz,-1.,1.)
        Tt=jnp.clip(mass*(G_CMP+KP_Z_CMP*ez-KD_Z_CMP*vz_w),
                    0.3*mass*G_CMP,2.5*mass*G_CMP)
        cy,sy_=jnp.cos(psi),jnp.sin(psi)
        ex_b=(px_d-px)*cy+(py_d-py)*sy_; ey_b=-(px_d-px)*sy_+(py_d-py)*cy
        th_d=jnp.clip(KP_XY_CMP*ex_b,-MAX_TILT_CMP,MAX_TILT_CMP)
        phi_d=jnp.clip(-KP_XY_CMP*ey_b,-MAX_TILT_CMP,MAX_TILT_CMP)
        Mx=KP_RP*(phi_d-phi)-KD_RP*p; My=KP_RP*(th_d-th)-KD_RP*q
        Mz=KP_YAW*(psi_d-psi)-KD_YAW*r
        return _mix(Tt,Mx,My,Mz)

    def _f(state,motors):
        px,py,pz,phi,th,psi,u,v,w,p,q,r=state
        T=kT*motors**2; Q=kQ*motors**2; Tt=jnp.sum(T)
        Mx=L*(T[1]+T[3]-T[0]-T[2]); My=L*(T[0]+T[1]-T[2]-T[3])
        Mz=-Q[0]+Q[1]+Q[2]-Q[3]
        R=_R(phi,th,psi); vel_b=jnp.array([u,v,w])
        F_b=jnp.array([0.,0.,Tt])+R.T@jnp.array([0.,0.,-mass*G_CMP])-Kd*vel_b
        phi_d,th_d,psi_d=_ek(phi,th,p,q,r); ob=jnp.array([p,q,r])
        Io=jnp.array([Ixx*p,Iyy*q,Izz*r])
        return jnp.concatenate([R@vel_b,jnp.array([phi_d,th_d,psi_d]),
                                 F_b/mass-jnp.cross(ob,vel_b),
                                 I_inv*(jnp.array([Mx,My,Mz])-jnp.cross(ob,Io))])
    def rk4_(state,motors):
        k1=_f(state,motors); k2=_f(state+.5*DT_CMP*k1,motors)
        k3=_f(state+.5*DT_CMP*k2,motors); k4=_f(state+DT_CMP*k3,motors)
        return state+(DT_CMP/6.)*(k1+2*k2+2*k3+k4)

    omega_h=float(jnp.sqrt(mass*G_CMP/(4.*kT)))
    return jax.jit(ctrl), jax.jit(rk4_), omega_h, omega_h*60./PI2_CMP, kT

# %%
def _settle(t, y, t0, target, band=0.05):
    mask = (t > t0) & (np.abs(y - target) < band * max(abs(target), 0.1))
    return float(t[mask][0] - t0) if mask.any() else float("nan")

cmp_logs = []; cmp_stats = []

if RUN_COMPARISON:
    import time as _time
    for cfg in DRONES_CMP:
        ctrl_j, rk4_j, omega_h, rpm_h, kT = _make_sim_cmp(cfg)
        state = jnp.zeros(12)
        ls = np.zeros((N_CMP+1,12)); lm = np.zeros((N_CMP+1,4)); lt = np.zeros(N_CMP+1)
        ls[0] = np.array(state)
        t0w = _time.time()
        for i in range(N_CMP):
            t = i*DT_CMP; mot = ctrl_j(state, _setpoint_cmp(t))
            state = rk4_j(state, mot)
            ls[i+1]=np.array(state); lm[i+1]=np.array(mot); lt[i+1]=(i+1)*DT_CMP
        rpm = lm * 60. / PI2_CMP
        st1 = _settle(lt, ls[:,2],   3.5,  2.0)
        st2 = _settle(lt, ls[:,2],  11.25, 3.0)
        sty = _settle(lt, np.degrees(ls[:,5]), 7.5, 60.0)
        att_scale = float(np.sqrt(cfg["Ixx"]/IXX_REF_CMP))
        yaw_scale = float(np.sqrt(cfg["Izz"]/IZZ_REF_CMP))
        cmp_stats.append({
            "name": cfg["name"], "mass_kg": cfg["mass"],
            "prop_in": round(cfg["D"]*39.37,1),
            "hover_rpm": round(rpm_h,0), "max_rpm": round(float(rpm.max()),0),
            "KP_RP": round(KP_RP_REF*att_scale,2), "KP_YAW": round(KP_YAW_REF*yaw_scale,2),
            "settle_alt1_s": round(st1,2) if not np.isnan(st1) else "—",
            "settle_alt2_s": round(st2,2) if not np.isnan(st2) else "—",
            "settle_yaw_s":  round(sty,2) if not np.isnan(sty) else "—",
            "max_tilt_deg":  round(float(np.degrees(np.abs(ls[:,3:5])).max()),2),
            "max_speed_ms":  round(float(np.linalg.norm(ls[:,6:9],axis=1).max()),2),
        })
        cmp_logs.append((lt, ls, rpm, cfg))
        print(f"  {cfg['name']}: hover {rpm_h:.0f} RPM  "
              f"alt {st1:.2f}s  yaw {sty:.2f}s  ({_time.time()-t0w:.1f}s wall)")
else:
    # Load pre-computed stats
    stats_path = REPO / "data" / "comparison_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            cmp_stats = json.load(f)
        print(f"Loaded cached comparison stats ({len(cmp_stats)} drones)")

# %% [markdown]
# ### 📊 Comparison Stats Table

# %%
if cmp_stats:
    rows_cmp = []
    metrics = [
        ("Mass (kg)",           "mass_kg"),
        ("Prop size (in)",      "prop_in"),
        ("Hover RPM",           "hover_rpm"),
        ("Max RPM (mission)",   "max_rpm"),
        ("KP_RP (scaled)",      "KP_RP"),
        ("KP_YAW (scaled)",     "KP_YAW"),
        ("Alt 0→2 m settle (s)","settle_alt1_s"),
        ("Alt 2→3 m settle (s)","settle_alt2_s"),
        ("Yaw 0→60° settle (s)","settle_yaw_s"),
        ("Max tilt (°)",        "max_tilt_deg"),
        ("Max airspeed (m/s)",  "max_speed_ms"),
    ]
    for label, key in metrics:
        row = {"Metric": label}
        for s in cmp_stats:
            row[s["name"]] = s[key]
        rows_cmp.append(row)
    df_cmp = pd.DataFrame(rows_cmp).set_index("Metric")
    print(df_cmp.to_string())

# %% [markdown]
# ### 📈 Dynamics Comparison Plots

# %%
if cmp_logs:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor="#0a0a14")
    fig.suptitle("Drone Model Comparison — Same 20 s Mission, Same PID Structure",
                 color="white", fontsize=14)
    panels_c = [
        ("Altitude z (m)",      lambda s: s[:,2]),
        ("Yaw ψ (°)",           lambda s: np.degrees(s[:,5])),
        ("Airspeed |V| (m/s)",  lambda s: np.linalg.norm(s[:,6:9],axis=1)),
        ("Roll φ (°)",          lambda s: np.degrees(s[:,3])),
        ("Motor RPM (FL)",      None),
        ("Position x (m)",      lambda s: s[:,0]),
    ]
    lt_ref = cmp_logs[0][0]
    def _np_ramp(t,t0,z0,z1,v=0.8):
        return z0+np.clip(np.maximum(t-t0,0.)*v/max(abs(z1-z0),1e-3),0.,1.)*(z1-z0)
    sp_z_c   = np.where(lt_ref<1.,0.,np.where(lt_ref<10.,
                _np_ramp(lt_ref,1.,0.,2.),_np_ramp(lt_ref,10.,2.,3.)))
    sp_psi_c = np.where(lt_ref<6.,0.,_np_ramp(lt_ref,6.,0.,60.,.3/np.pi*180))

    for ai,(ylabel,fn) in enumerate(panels_c):
        ax=axes.flat[ai]
        ax.set_facecolor("#0a0a14"); ax.set_xlabel("Time (s)",color="white")
        ax.set_ylabel(ylabel,color="white"); ax.tick_params(colors="white")
        ax.spines[:].set_color("#555"); ax.grid(alpha=0.15,color="white")
        ax.set_title(ylabel,color="white",fontsize=10)
        for lt_,ls_,rpm_,cfg_ in cmp_logs:
            c=cfg_["color"]; lbl=cfg_["name"]
            if ai==4: ax.plot(lt_,rpm_[:,0],color=c,lw=1.,alpha=.9,label=lbl)
            else:     ax.plot(lt_,fn(ls_),  color=c,lw=1.2,alpha=.9,label=lbl)
        if ai==0: ax.plot(lt_ref,sp_z_c,  "--",color="white",alpha=.3,lw=.8,label="setpoint")
        if ai==1: ax.plot(lt_ref,sp_psi_c,"--",color="white",alpha=.3,lw=.8,label="setpoint")
        ax.legend(facecolor="#0a0a14",edgecolor="#444",labelcolor="white",fontsize=8)
    plt.tight_layout(); plt.show()

# %%
# Motor RPM profiles — one subplot per drone
if cmp_logs:
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5), facecolor="#0a0a14")
    fig2.suptitle("Motor RPM Profiles — All 4 Motors", color="white", fontsize=13)
    m_colors = ["#00d4ff","#ff6b35","#7fff00","#c77dff"]
    m_labels = ["FL (CCW)","FR (CW)","RL (CW)","RR (CCW)"]
    for ax2,(lt_,ls_,rpm_,cfg_) in zip(axes2, cmp_logs):
        ax2.set_facecolor("#0a0a14"); ax2.set_title(cfg_["name"],color="white",fontsize=10)
        ax2.set_xlabel("Time (s)",color="white"); ax2.set_ylabel("RPM",color="white")
        ax2.tick_params(colors="white"); ax2.spines[:].set_color("#555")
        ax2.grid(alpha=0.15,color="white")
        for mi,(mc,ml) in enumerate(zip(m_colors,m_labels)):
            ax2.plot(lt_,rpm_[:,mi],color=mc,lw=.9,alpha=.85,label=ml)
        ax2.legend(facecolor="#0a0a14",edgecolor="#444",labelcolor="white",fontsize=8,ncol=2)
    plt.tight_layout(); plt.show()

# %%
# Step response bar chart
if cmp_stats:
    fig3, axes3 = plt.subplots(1, 3, figsize=(14, 5), facecolor="#0a0a14")
    fig3.suptitle("Step Response Comparison", color="white", fontsize=13)
    bar_metrics = [
        ("settle_alt1_s", "Alt 0→2 m settle (s)"),
        ("settle_yaw_s",  "Yaw 0→60° settle (s)"),
        ("max_tilt_deg",  "Max tilt angle (°)"),
    ]
    colors_bar = [cfg["color"] for cfg in DRONES_CMP]
    for ax3,(key,title) in zip(axes3,bar_metrics):
        vals  = [s[key] if isinstance(s[key],float) else float("nan") for s in cmp_stats]
        names = [s["name"].replace(" ","\n") for s in cmp_stats]
        bars  = ax3.bar(names,vals,color=colors_bar,alpha=.85,edgecolor="#555")
        for bar,v in zip(bars,vals):
            if not np.isnan(v):
                ax3.text(bar.get_x()+bar.get_width()/2,
                         bar.get_height()+0.02*max(v for v in vals if not np.isnan(v)),
                         f"{v:.2f}",ha="center",color="white",fontsize=10,fontweight="bold")
        ax3.set_facecolor("#0a0a14"); ax3.set_title(title,color="white",fontsize=10)
        ax3.tick_params(colors="white"); ax3.spines[:].set_color("#555")
        ax3.grid(alpha=0.15,color="white",axis="y")
    plt.tight_layout(); plt.show()

# %% [markdown]
# ### 💡 Key Findings
#
# - **Altitude and yaw settle times are nearly identical** across all three airframes —
#   the mass-compensated PID with inertia-scaled attitude gains achieves the same
#   closed-loop bandwidth regardless of vehicle size.
# - The **racer operates 43% higher RPM** at hover; the lifter **44% lower**.
#   Transient swings are wider on the racer (short arm → more Δω per unit moment).
# - Without inertia scaling (same raw gains), the racer saturates its motor RPM
#   limits during attitude corrections, losing altitude tracking entirely —
#   demonstrating why **gain scheduling is essential** when porting a controller
#   across airframes.

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
