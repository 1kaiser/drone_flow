"""
Cascaded PID controller.

Outer loop: altitude + heading  → desired attitude (phi_d, theta_d)
Inner loop: attitude            → moments (Mx, My, Mz)
Mixer:      moments + T_total   → motor omega [FL, FR, RL, RR]
"""
import jax.numpy as jnp
from drone_dynamics.propeller import CT, CQ, RHO, D, G
from drone_dynamics.dynamics  import MASS, L, R_body_to_world

# ── Gain tables ───────────────────────────────────────────────────────────────
KP_Z,  KD_Z  = 2.5,  4.5     # altitude  (ωn≈1.6, ζ≈1.4 → overdamped)
KP_RP, KD_RP = 8.0,  3.5     # roll/pitch
KP_YAW,KD_YAW= 3.0,  2.0     # yaw
KP_XY        = 0.25           # position → desired tilt (rad per m error)
MAX_TILT     = jnp.radians(25.0)   # safety clamp

# ── Motor mixing constants ────────────────────────────────────────────────────
_kT = CT * RHO * D**4 / (2*jnp.pi)**2   # T = _kT * omega^2
_kQ = CQ * RHO * D**5 / (2*jnp.pi)**2   # Q = _kQ * omega^2
_r  = _kQ / _kT                          # torque-to-thrust ratio

OMEGA_MIN = 100.0    # rad/s  (~955 RPM)
OMEGA_MAX = 2000.0   # rad/s  (~19099 RPM)


def _T_to_omega(T):
    return jnp.sqrt(jnp.maximum(T, 0.0) / _kT)

def mixer(T_total, Mx, My, Mz):
    """
    Allocate [T_total, Mx, My, Mz] → [omega_FL, omega_FR, omega_RL, omega_RR]
    X-config allocation matrix (inverse):
      T_FL =  T/4 - Mx/(4L) + My/(4L) + Mz/(4r)
      T_FR =  T/4 + Mx/(4L) + My/(4L) - Mz/(4r)
      T_RL =  T/4 - Mx/(4L) - My/(4L) - Mz/(4r)
      T_RR =  T/4 + Mx/(4L) - My/(4L) + Mz/(4r)
    """
    a = T_total / 4.0
    b = Mx / (4.0 * L)
    c = My / (4.0 * L)
    d = Mz / (4.0 * _r)
    # To yaw CCW (+Mz): speed up CW motors FR/RL, slow down CCW motors FL/RR
    T_FL = a - b + c - d
    T_FR = a + b + c + d
    T_RL = a - b - c + d
    T_RR = a + b - c - d
    omegas = jnp.array([_T_to_omega(T_FL), _T_to_omega(T_FR),
                         _T_to_omega(T_RL), _T_to_omega(T_RR)])
    return jnp.clip(omegas, OMEGA_MIN, OMEGA_MAX)


def control(state, setpoint):
    """
    Compute motor omega commands.
    state:    [px,py,pz, phi,theta,psi, u,v,w, p,q,r]
    setpoint: [px_d, py_d, pz_d, psi_d]
    """
    px, py, pz, phi, theta, psi, u, v, w, p, q, r = state
    px_d, py_d, pz_d, psi_d = setpoint

    # ── World-frame vertical velocity ──────────────────────────────────────
    R  = R_body_to_world(phi, theta, psi)
    vel_w = R @ jnp.array([u, v, w])
    vz_w  = vel_w[2]
    vx_w, vy_w = vel_w[0], vel_w[1]

    # ── Altitude controller → total thrust ────────────────────────────────
    # Cap error to ±1 m so a large step doesn't produce runaway thrust
    ez   = jnp.clip(pz_d - pz, -1.0, 1.0)
    T_total = MASS * (G + KP_Z * ez - KD_Z * vz_w)
    T_total = jnp.clip(T_total, 0.3 * MASS * G, 2.5 * MASS * G)

    # ── Position controller → desired tilt ────────────────────────────────
    ex, ey = px_d - px, py_d - py
    # Rotate error to body horizontal frame (yaw-compensated)
    cy, sy = jnp.cos(psi), jnp.sin(psi)
    ex_b =  cy*ex + sy*ey
    ey_b = -sy*ex + cy*ey

    theta_d = jnp.clip( KP_XY * ex_b, -MAX_TILT, MAX_TILT)
    phi_d   = jnp.clip(-KP_XY * ey_b, -MAX_TILT, MAX_TILT)

    # ── Attitude controller → moments ─────────────────────────────────────
    Mx = KP_RP  * (phi_d   - phi)   - KD_RP  * p
    My = KP_RP  * (theta_d - theta) - KD_RP  * q
    Mz = KP_YAW * (psi_d   - psi)   - KD_YAW * r

    return mixer(T_total, Mx, My, Mz)
