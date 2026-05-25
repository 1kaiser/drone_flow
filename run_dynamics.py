"""
Phase 1 test: hover trim + altitude step + yaw step + horizontal position step.
Runs 6-DOF dynamics at 500 Hz, plots 6-panel state response.
"""
import os, sys
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "/home/kaiser/projects/gi/drone_flow")

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drone_dynamics.propeller import hover_omega, D, CT, CQ, RHO, omega_to_rpm
from drone_dynamics.dynamics   import MASS, G, rk4
from drone_dynamics.controller import control

# ── JIT compile ──────────────────────────────────────────────────────────────
rk4_jit  = jax.jit(rk4)
ctrl_jit = jax.jit(control)

# ── Simulation parameters ────────────────────────────────────────────────────
DT    = 1.0 / 500.0   # 500 Hz
T_END = 15.0
N     = int(T_END / DT)

# Initial state: at origin, stationary, z=0
state = jnp.zeros(12)

# Smooth ramp helper: ramp from z0→z1 at v_max m/s starting at t0
def ramp(t, t0, z0, z1, v_max=0.8):
    dt   = jnp.maximum(t - t0, 0.0)
    frac = jnp.clip(dt * v_max / jnp.maximum(jnp.abs(z1 - z0), 1e-3), 0.0, 1.0)
    return z0 + frac * (z1 - z0)

# Setpoint schedule (smooth ramps, not hard steps)
def setpoint(t):
    z_d   = jnp.where(t < 1.0,
                0.0,                            # hold at ground 1s
                jnp.where(t < 10.0,
                    ramp(t, 1.0, 0.0, 2.0),    # climb to 2m @0.8m/s
                    ramp(t, 10.0, 2.0, 3.0)))   # step to 3m @0.8m/s
    px_d  = jnp.where(t < 9.0, 0.0,
                ramp(t, 9.0, 0.0, 3.0, 0.5))   # move 3m fwd @0.5m/s
    py_d  = 0.0
    psi_d = jnp.where(t < 6.0, 0.0,
                ramp(t, 6.0, 0.0, jnp.radians(45.0), 0.3))  # yaw to 45°
    return jnp.array([px_d, py_d, z_d, psi_d])

# ── Run ──────────────────────────────────────────────────────────────────────
print("Running 6-DOF simulation... ", end="", flush=True)

log_states = np.zeros((N+1, 12))
log_motors = np.zeros((N+1, 4))
log_t      = np.zeros(N+1)
log_states[0] = np.array(state)

for i in range(N):
    t   = i * DT
    sp  = setpoint(t)
    mot = ctrl_jit(state, sp)
    state = rk4_jit(state, mot, DT)
    log_states[i+1] = np.array(state)
    log_motors[i+1] = np.array(mot)
    log_t[i+1]      = (i+1) * DT

print("done.")

RPM = omega_to_rpm(log_motors)   # (N+1, 4)

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(15, 11), facecolor="#0a0a14")
fig.suptitle("Quadcopter 6-DOF dynamics  (500 Hz)  15 s mission", color="white", fontsize=13)

# Setpoint traces for overlay
def np_ramp(t, t0, z0, z1, v_max=0.8):
    dt   = np.maximum(t - t0, 0.0)
    frac = np.clip(dt * v_max / max(abs(z1 - z0), 1e-3), 0.0, 1.0)
    return z0 + frac * (z1 - z0)

sp_z   = np.where(log_t < 1.0, 0.0,
          np.where(log_t < 10.0, np_ramp(log_t,1.0,0.0,2.0), np_ramp(log_t,10.0,2.0,3.0)))
sp_px  = np.where(log_t < 9.0, 0.0, np_ramp(log_t,9.0,0.0,3.0,0.5))
sp_psi = np.where(log_t < 6.0, 0.0, np_ramp(log_t,6.0,0.0,45.0,0.3))

panels = [
    # (ax_idx, x, y, ylabel, color, setpoint or None)
    (0, log_t, log_states[:,2],              "Altitude z (m)",      "#00d4ff", sp_z),
    (1, log_t, log_states[:,0],              "Position x (m)",      "#00d4ff", sp_px),
    (2, log_t, np.degrees(log_states[:,3]),  "Roll φ (°)",          "#ff6b35", None),
    (3, log_t, np.degrees(log_states[:,5]),  "Yaw ψ (°)",           "#ff6b35", sp_psi),
    (4, log_t, RPM[:,0],                     "Motor RPM (FL)",       "#7fff00", None),
    (5, log_t, np.linalg.norm(log_states[:,6:9], axis=1), "Airspeed |V| (m/s)", "#c77dff", None),
]

for idx, (ai, t, y, ylabel, color, sp) in enumerate(panels):
    ax = axes.flat[ai]
    ax.plot(t, y, color=color, lw=1.2)
    if sp is not None:
        ax.plot(t, sp, "--", color="white", alpha=0.35, lw=1.0, label="setpoint")
        ax.legend(facecolor="#0a0a14", edgecolor="#555", labelcolor="white", fontsize=8)
    if "RPM" in ylabel:
        colors_m = ["#00d4ff","#ff6b35","#7fff00","#c77dff"]
        labels_m = ["FL","FR","RL","RR"]
        for mi, (mc, ml) in enumerate(zip(colors_m, labels_m)):
            ax.plot(t, RPM[:,mi], color=mc, lw=0.9, alpha=0.85, label=ml)
        ax.legend(facecolor="#0a0a14", edgecolor="#555", labelcolor="white", fontsize=8,
                  ncol=2)
    ax.set_facecolor("#0a0a14")
    ax.set_xlabel("Time (s)", color="white")
    ax.set_ylabel(ylabel, color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#555")
    ax.grid(alpha=0.15, color="white")

plt.tight_layout()
out = "/home/kaiser/projects/gi/drone_flow/dynamics_result.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0a0a14")
print(f"Saved: {out}")

# ── Summary ──────────────────────────────────────────────────────────────────
omega_h = hover_omega(MASS)
print(f"\nDrone trim:")
print(f"  Mass {MASS} kg  |  Prop {D*100:.0f} cm  |  Hover {omega_to_rpm(omega_h):.0f} RPM")
print(f"  Hover thrust: {MASS*G:.2f} N  ({MASS*G/4:.2f} N per motor)")

def settle_time(t_arr, y_arr, t_step, target, band=0.05):
    mask = (t_arr > t_step) & (np.abs(y_arr - target) < band * max(abs(target), 0.1))
    return (t_arr[mask][0] - t_step) if mask.any() else float("nan")

print(f"\nStep responses:")
print(f"  Altitude 0→2 m  settle (5%): {settle_time(log_t, log_states[:,2],  3.5, 2.0):.2f} s")
print(f"  Altitude 2→3 m  settle (5%): {settle_time(log_t, log_states[:,2], 11.25, 3.0):.2f} s")
print(f"  Yaw 0→45° settle (5%):       {settle_time(log_t, np.degrees(log_states[:,5]), 7.5, 45.0):.2f} s")
print(f"  Position 0→3 m  settle (5%): {settle_time(log_t, log_states[:,0], 15.0, 3.0):.2f} s")
