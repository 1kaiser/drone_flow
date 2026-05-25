#!/usr/bin/env python3
"""
Multi-drone model comparison — same 20 s mission, three airframes.

Models:
  A  Photography Drone  (baseline, 1.5 kg, 8" props)
  B  5" Racing Drone    (lightweight, 0.45 kg, 5" props)
  C  Heavy Lifter       (3.5 kg, 13" props)

All run the same cascaded PID controller (gains auto-scaled by mass).
Output: assets/comparison_*.png  +  data/comparison_stats.json
"""
import os, sys, json, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from pathlib import Path
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RHO   = 1.225      # kg/m³
G_ACC = 9.81       # m/s²
PI2   = 2.0 * np.pi

# ── Drone configs ─────────────────────────────────────────────────────────────
DRONES = [
    dict(
        name="Photography Drone",  label="photo",  color="#00d4ff",
        mass=1.5,
        D=0.203,   CT=0.109,  CQ=0.0095,   # 8" DJI-style prop
        L=0.2545,
        Ixx=0.0196, Iyy=0.0196, Izz=0.0264,
        Kd=0.25,
    ),
    dict(
        name='5" Racing Drone',  label="racer",  color="#ff6b35",
        mass=0.45,
        D=0.127,   CT=0.105,  CQ=0.0088,   # 5" race prop
        L=0.120,
        Ixx=0.0020, Iyy=0.0020, Izz=0.0035,
        Kd=0.08,
    ),
    dict(
        name="Heavy Lifter",  label="lifter",  color="#7fff00",
        mass=3.5,
        D=0.330,   CT=0.115,  CQ=0.011,    # 13" industrial prop
        L=0.380,
        Ixx=0.0650, Iyy=0.0650, Izz=0.0900,
        Kd=0.30,
    ),
]

# ── Controller gains (reference drone) ────────────────────────────────────────
# Altitude / position: mass-compensated so no scaling needed
KP_Z,  KD_Z  = 2.5,  4.5
KP_XY        = 0.25
MAX_TILT     = np.radians(25.0)
# Attitude / yaw: scaled per drone by sqrt(I_drone / I_ref) to equalize bandwidth
# Reference inertias (Photography Drone)
IXX_REF = 0.0196; IZZ_REF = 0.0264
KP_RP_REF, KD_RP_REF   = 8.0, 3.5
KP_YAW_REF, KD_YAW_REF = 3.0, 2.0

# ── Mission setpoint (same for all drones) ────────────────────────────────────
DT    = 1.0 / 500.0
T_END = 20.0
N_SIM = int(T_END / DT)
FPS   = 30
RATIO = int(1.0 / (FPS * DT))

def ramp(t, t0, z0, z1, v=0.8):
    dt = jnp.maximum(t - t0, 0.)
    return z0 + jnp.clip(dt * v / jnp.maximum(jnp.abs(z1-z0), 1e-3), 0., 1.) * (z1-z0)

def setpoint(t):
    z_d   = jnp.where(t < 1., 0.,
              jnp.where(t < 10., ramp(t,1.,0.,2.), ramp(t,10.,2.,3.)))
    px_d  = jnp.where(t < 9., 0., ramp(t, 9., 0., 4., 0.4))
    py_d  = jnp.where(t < 13., 0., ramp(t,13., 0., 2., 0.3))
    psi_d = jnp.where(t < 6., 0., ramp(t, 6., 0., jnp.radians(60.), 0.25))
    return jnp.array([px_d, py_d, z_d, psi_d])


# ── Parametric simulation (closures capture drone-specific constants) ─────────
def make_sim(cfg):
    mass = cfg["mass"]; D = cfg["D"]; CT = cfg["CT"]; CQ = cfg["CQ"]
    L = cfg["L"]; Ixx = cfg["Ixx"]; Iyy = cfg["Iyy"]; Izz = cfg["Izz"]
    Kd = cfg["Kd"]
    KP_RP = cfg.get("KP_RP", KP_RP_REF); KD_RP = cfg.get("KD_RP", KD_RP_REF)
    KP_YAW = cfg.get("KP_YAW", KP_YAW_REF); KD_YAW = cfg.get("KD_YAW", KD_YAW_REF)

    kT  = CT * RHO * D**4 / PI2**2
    kQ  = CQ * RHO * D**5 / PI2**2
    r_  = kQ / kT
    I_inv = jnp.array([1.0/Ixx, 1.0/Iyy, 1.0/Izz])
    OMEGA_MIN = 50.0; OMEGA_MAX = 3000.0

    # hover RPM
    omega_h = float(jnp.sqrt(mass * G_ACC / (4.0 * kT)))
    rpm_h   = omega_h * 60.0 / PI2
    twr     = (4.0 * kT * omega_h**2) / (mass * G_ACC)  # should be ≈1.0

    def _T_to_omega(T):
        return jnp.sqrt(jnp.maximum(T, 0.0) / kT)

    def mixer(T_total, Mx, My, Mz):
        a = T_total / 4.0; b = Mx / (4.0*L); c = My / (4.0*L); d = Mz / (4.0*r_)
        omegas = jnp.array([
            _T_to_omega(a - b + c - d),  # FL CCW
            _T_to_omega(a + b + c + d),  # FR CW
            _T_to_omega(a - b - c + d),  # RL CW
            _T_to_omega(a + b - c - d),  # RR CCW
        ])
        return jnp.clip(omegas, OMEGA_MIN, OMEGA_MAX)

    def R_bw(phi, theta, psi):
        cp,sp = jnp.cos(phi),   jnp.sin(phi)
        ct,st = jnp.cos(theta), jnp.sin(theta)
        cy,sy = jnp.cos(psi),   jnp.sin(psi)
        return jnp.array([
            [cy*ct, cy*st*sp-sy*cp, cy*st*cp+sy*sp],
            [sy*ct, sy*st*sp+cy*cp, sy*st*cp-cy*sp],
            [-st,   ct*sp,          ct*cp          ],
        ])

    def euler_kin(phi, theta, p, q, r):
        cp,sp = jnp.cos(phi), jnp.sin(phi)
        ct,st = jnp.cos(theta), jnp.sin(theta)
        tt = st / ct
        return (p + sp*tt*q + cp*tt*r,
                cp*q - sp*r,
                (sp/ct)*q + (cp/ct)*r)

    def control(state, sp):
        px,py,pz,phi,theta,psi,u,v,w,p,q,r = state
        px_d,py_d,pz_d,psi_d = sp
        R = R_bw(phi, theta, psi)
        vel_w = R @ jnp.array([u,v,w])
        vz_w  = vel_w[2]; vx_w = vel_w[0]; vy_w = vel_w[1]
        ez    = jnp.clip(pz_d - pz, -1.0, 1.0)
        T_tot = mass * (G_ACC + KP_Z*ez - KD_Z*vz_w)
        T_tot = jnp.clip(T_tot, 0.3*mass*G_ACC, 2.5*mass*G_ACC)
        ex,ey = px_d - px, py_d - py
        cy,sy_ = jnp.cos(psi), jnp.sin(psi)
        ex_b =  cy*ex + sy_*ey; ey_b = -sy_*ex + cy*ey
        theta_d = jnp.clip( KP_XY*ex_b, -MAX_TILT, MAX_TILT)
        phi_d   = jnp.clip(-KP_XY*ey_b, -MAX_TILT, MAX_TILT)
        Mx = KP_RP*(phi_d   - phi)   - KD_RP*p
        My = KP_RP*(theta_d - theta) - KD_RP*q
        Mz = KP_YAW*(psi_d  - psi)  - KD_YAW*r
        # (KP_RP, KD_RP, KP_YAW, KD_YAW captured from make_sim closure)
        return mixer(T_tot, Mx, My, Mz)

    def step(state, motors):
        px,py,pz,phi,theta,psi,u,v,w,p,q,r = state
        oFL,oFR,oRL,oRR = motors
        T  = kT * jnp.array([oFL,oFR,oRL,oRR])**2
        Q  = kQ * jnp.array([oFL,oFR,oRL,oRR])**2
        T_tot = jnp.sum(T)
        Mx = L*(T[1]+T[3]-T[0]-T[2])
        My = L*(T[0]+T[1]-T[2]-T[3])
        Mz = -Q[0]+Q[1]+Q[2]-Q[3]
        R  = R_bw(phi, theta, psi)
        vel_b = jnp.array([u,v,w])
        grav_b = R.T @ jnp.array([0.,0.,-mass*G_ACC])
        drag_b = -Kd*vel_b
        F_b = jnp.array([0.,0.,T_tot]) + grav_b + drag_b
        pos_dot = R @ vel_b
        phi_d,theta_d,psi_d = euler_kin(phi,theta,p,q,r)
        omega_b = jnp.array([p,q,r])
        vel_dot = F_b/mass - jnp.cross(omega_b,vel_b)
        I_omega = jnp.array([Ixx*p,Iyy*q,Izz*r])
        rates_dot = I_inv*(jnp.array([Mx,My,Mz]) - jnp.cross(omega_b,I_omega))
        return jnp.concatenate([pos_dot,
                                 jnp.array([phi_d,theta_d,psi_d]),
                                 vel_dot, rates_dot])

    def rk4(state, motors):
        k1 = step(state, motors)
        k2 = step(state+0.5*DT*k1, motors)
        k3 = step(state+0.5*DT*k2, motors)
        k4 = step(state+DT*k3, motors)
        return state + (DT/6.)*(k1+2*k2+2*k3+k4)

    return jax.jit(control), jax.jit(rk4), omega_h, rpm_h, twr, kT, kQ


# ── Run all drones ────────────────────────────────────────────────────────────
def settle_time(t_arr, y_arr, t_step, target, band=0.05):
    mask = (t_arr > t_step) & (np.abs(y_arr - target) < band * max(abs(target), 0.1))
    return float(t_arr[mask][0] - t_step) if mask.any() else float("nan")

def omega_to_rpm(omega):
    return omega * 60.0 / PI2

all_logs = []
all_stats = []

for cfg in DRONES:
    print(f"\n{'='*55}")
    print(f"  Simulating: {cfg['name']}")
    print(f"{'='*55}")
    # Attitude gains scaled by sqrt(I/I_ref) so closed-loop bandwidth matches
    att_scale = np.sqrt(cfg["Ixx"] / IXX_REF)
    yaw_scale = np.sqrt(cfg["Izz"] / IZZ_REF)
    cfg["KP_RP"]  = KP_RP_REF  * att_scale
    cfg["KD_RP"]  = KD_RP_REF  * att_scale
    cfg["KP_YAW"] = KP_YAW_REF * yaw_scale
    cfg["KD_YAW"] = KD_YAW_REF * yaw_scale
    ctrl_jit, rk4_jit, omega_h, rpm_h, twr, kT, kQ = make_sim(cfg)

    state = jnp.zeros(12)
    log_s = np.zeros((N_SIM+1, 12))
    log_m = np.zeros((N_SIM+1, 4))
    log_t = np.zeros(N_SIM+1)
    log_s[0] = np.array(state)

    t0 = time.time()
    for i in range(N_SIM):
        t = i * DT
        sp = setpoint(t)
        mot = ctrl_jit(state, sp)
        state = rk4_jit(state, mot)
        log_s[i+1] = np.array(state)
        log_m[i+1] = np.array(mot)
        log_t[i+1] = (i+1) * DT

    wall = time.time() - t0
    RPM = omega_to_rpm(log_m)

    st_alt1 = settle_time(log_t, log_s[:,2], 3.5,  2.0)
    st_alt2 = settle_time(log_t, log_s[:,2], 11.25, 3.0)
    st_yaw  = settle_time(log_t, np.degrees(log_s[:,5]), 7.5, 60.0)
    max_rpm = float(RPM.max())
    max_tilt= float(np.degrees(np.abs(log_s[:,3:5])).max())
    max_spd = float(np.linalg.norm(log_s[:,6:9], axis=1).max())

    stats = {
        "name":           cfg["name"],
        "label":          cfg["label"],
        "mass_kg":        cfg["mass"],
        "prop_in":        round(cfg["D"] * 39.37, 1),
        "hover_rpm":      round(rpm_h, 0),
        "max_rpm":        round(max_rpm, 0),
        "twr":            round(twr, 3),
        "KP_RP":          round(float(cfg.get("KP_RP", KP_RP_REF)), 2),
        "KP_YAW":         round(float(cfg.get("KP_YAW", KP_YAW_REF)), 2),
        "settle_alt1_s":  round(st_alt1, 2) if not np.isnan(st_alt1) else "—",
        "settle_alt2_s":  round(st_alt2, 2) if not np.isnan(st_alt2) else "—",
        "settle_yaw_s":   round(st_yaw,  2) if not np.isnan(st_yaw)  else "—",
        "max_tilt_deg":   round(max_tilt, 2),
        "max_speed_ms":   round(max_spd, 2),
        "wall_time_s":    round(wall, 1),
    }
    all_stats.append(stats)
    all_logs.append((log_t, log_s, RPM, cfg))

    print(f"  Hover {rpm_h:.0f} RPM  T/W={twr:.3f}  ({wall:.1f}s wall)")
    print(f"  Alt  0→2m: {st_alt1:.2f}s  |  Alt 2→3m: {st_alt2:.2f}s")
    print(f"  Yaw 0→60°: {st_yaw:.2f}s  |  Max tilt: {max_tilt:.1f}°")


# ── Comparison figure ─────────────────────────────────────────────────────────
print("\nGenerating comparison figures...")
(REPO / "assets").mkdir(exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor="#0a0a14")
fig.suptitle("Drone Model Comparison — Same 20 s Mission, Same PID Gains",
             color="white", fontsize=14)

panels = [
    (0, "Altitude z (m)",       lambda s: s[:,2]),
    (1, "Yaw ψ (°)",            lambda s: np.degrees(s[:,5])),
    (2, "Airspeed |V| (m/s)",   lambda s: np.linalg.norm(s[:,6:9], axis=1)),
    (3, "Roll φ (°)",           lambda s: np.degrees(s[:,3])),
    (4, "Motor RPM (FL)",       None),
    (5, "Position x (m)",       lambda s: s[:,0]),
]

# Setpoint overlays
log_t_ref = all_logs[0][0]
def np_ramp(t, t0, z0, z1, v=0.8):
    dt = np.maximum(t - t0, 0.); frac = np.clip(dt*v/max(abs(z1-z0),1e-3),0.,1.)
    return z0 + frac*(z1-z0)

sp_z = np.where(log_t_ref<1.,0.,
        np.where(log_t_ref<10.,np_ramp(log_t_ref,1.,0.,2.),np_ramp(log_t_ref,10.,2.,3.)))
sp_psi = np.where(log_t_ref<6.,0.,np_ramp(log_t_ref,6.,0.,60.,.3/np.pi*180))

for ai, (_, ylabel, fn) in enumerate(panels):
    ax = axes.flat[ai]
    ax.set_facecolor("#0a0a14"); ax.set_xlabel("Time (s)", color="white")
    ax.set_ylabel(ylabel, color="white"); ax.tick_params(colors="white")
    ax.spines[:].set_color("#555"); ax.grid(alpha=0.15, color="white")
    ax.set_title(ylabel, color="white", fontsize=10)

    for log_t_, log_s_, RPM_, cfg_ in all_logs:
        c = cfg_["color"]; lbl = cfg_["name"]
        if ai == 4:
            ax.plot(log_t_, RPM_[:,0], color=c, lw=1.0, alpha=0.9, label=lbl)
        else:
            ax.plot(log_t_, fn(log_s_), color=c, lw=1.2, alpha=0.9, label=lbl)

    if ai == 0:
        ax.plot(log_t_ref, sp_z, "--", color="white", alpha=0.3, lw=0.8, label="setpoint")
    if ai == 1:
        ax.plot(log_t_ref, sp_psi, "--", color="white", alpha=0.3, lw=0.8, label="setpoint")
    ax.legend(facecolor="#0a0a14", edgecolor="#444", labelcolor="white", fontsize=8)

plt.tight_layout()
out_cmp = REPO / "assets" / "comparison_dynamics.png"
plt.savefig(out_cmp, dpi=150, bbox_inches="tight", facecolor="#0a0a14")
plt.close()
print(f"  Saved: {out_cmp}")

# ── RPM profile figure ────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5), facecolor="#0a0a14")
fig2.suptitle("Motor RPM Profiles — All 4 Motors", color="white", fontsize=13)
m_labels = ["FL (CCW)","FR (CW)","RL (CW)","RR (CCW)"]
m_colors = ["#00d4ff","#ff6b35","#7fff00","#c77dff"]

for ax2, (log_t_, log_s_, RPM_, cfg_) in zip(axes2, all_logs):
    ax2.set_facecolor("#0a0a14")
    ax2.set_title(cfg_["name"], color="white", fontsize=10)
    ax2.set_xlabel("Time (s)", color="white"); ax2.set_ylabel("RPM", color="white")
    ax2.tick_params(colors="white"); ax2.spines[:].set_color("#555")
    ax2.grid(alpha=0.15, color="white")
    for mi, (mc, ml) in enumerate(zip(m_colors, m_labels)):
        ax2.plot(log_t_, RPM_[:,mi], color=mc, lw=0.9, alpha=0.85, label=ml)
    ax2.legend(facecolor="#0a0a14", edgecolor="#444", labelcolor="white",
               fontsize=8, ncol=2)

plt.tight_layout()
out_rpm = REPO / "assets" / "comparison_rpm.png"
plt.savefig(out_rpm, dpi=150, bbox_inches="tight", facecolor="#0a0a14")
plt.close()
print(f"  Saved: {out_rpm}")

# ── Bar chart — settle times ──────────────────────────────────────────────────
fig3, axes3 = plt.subplots(1, 3, figsize=(14, 5), facecolor="#0a0a14")
fig3.suptitle("Step Response Comparison", color="white", fontsize=13)
metrics = [
    ("settle_alt1_s", "Alt 0→2 m settle (s)"),
    ("settle_yaw_s",  "Yaw 0→60° settle (s)"),
    ("max_tilt_deg",  "Max tilt angle (°)"),
]
for ax3, (key, title) in zip(axes3, metrics):
    vals  = [s[key] for s in all_stats]
    names = [s["name"].replace(" ", "\n") for s in all_stats]
    cols  = [cfg["color"] for cfg in DRONES]
    bars  = ax3.bar(names, vals, color=cols, alpha=0.85, edgecolor="#555")
    for bar, v in zip(bars, vals):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02*max(vals),
                 f"{v:.2f}", ha="center", color="white", fontsize=10, fontweight="bold")
    ax3.set_facecolor("#0a0a14"); ax3.set_title(title, color="white", fontsize=10)
    ax3.tick_params(colors="white"); ax3.spines[:].set_color("#555")
    ax3.grid(alpha=0.15, color="white", axis="y")

plt.tight_layout()
out_bar = REPO / "assets" / "comparison_barplot.png"
plt.savefig(out_bar, dpi=150, bbox_inches="tight", facecolor="#0a0a14")
plt.close()
print(f"  Saved: {out_bar}")

# ── Save stats JSON ───────────────────────────────────────────────────────────
out_stats = REPO / "data" / "comparison_stats.json"
with open(out_stats, "w") as f:
    json.dump(all_stats, f, indent=2)
print(f"  Saved: {out_stats}")

# ── Print summary table ───────────────────────────────────────────────────────
print(f"\n{'='*75}")
print(f"{'Metric':<28}{'Photography':>14}{'Racing':>14}{'Lifter':>14}")
print(f"{'='*75}")
rows = [
    ("Mass (kg)",          "mass_kg",       "{:.2f}"),
    ("Prop size (in)",     "prop_in",       '{:.1f}"'),
    ("Hover RPM",          "hover_rpm",     "{:.0f}"),
    ("Max RPM (mission)",  "max_rpm",       "{:.0f}"),
    ("Thrust/Weight",      "twr",           "{:.3f}"),
    ("Alt 0→2m settle (s)","settle_alt1_s", "{:.2f}"),
    ("Alt 2→3m settle (s)","settle_alt2_s", "{:.2f}"),
    ("Yaw 0→60° settle (s)","settle_yaw_s", "{:.2f}"),
    ("Max tilt (°)",       "max_tilt_deg",  "{:.1f}"),
    ("Max airspeed (m/s)", "max_speed_ms",  "{:.2f}"),
]
for label, key, fmt in rows:
    vals = [fmt.format(s[key]) for s in all_stats]
    print(f"  {label:<26}{vals[0]:>14}{vals[1]:>14}{vals[2]:>14}")
print(f"{'='*75}")
