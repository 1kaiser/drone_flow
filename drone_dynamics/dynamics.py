"""
6-DOF quadcopter rigid-body dynamics (Newton-Euler, body frame).

State x (12,):  [px, py, pz,  phi, theta, psi,  u, v, w,  p, q, r]
                 world pos     ZYX Euler          body vel   body rates

Motors u (4,):  [omega_FL, omega_FR, omega_RL, omega_RR]  rad/s
Motor layout (top view, X-config):
    FL(CCW) ---+--- FR(CW)
               |
    RL(CW)  ---+--- RR(CCW)
Sign convention: CCW = positive yaw torque contribution.
"""
import jax.numpy as jnp
from drone_dynamics.propeller import thrust, torque, G

# ── Physical parameters ───────────────────────────────────────────────────────
MASS = 1.5      # kg
IXX  = 0.0196   # kg·m²  (estimated for 360mm arm span, 1.5kg)
IYY  = 0.0196
IZZ  = 0.0264
L    = 0.2545   # m  moment arm = 360mm * sin(45°)
KD   = 0.25     # linear translational drag  [N/(m/s)]

I_INV = jnp.array([1/IXX, 1/IYY, 1/IZZ])


# ── Kinematics ────────────────────────────────────────────────────────────────
def R_body_to_world(phi, theta, psi):
    """ZYX Euler → DCM (columns = body axes in world frame)"""
    cp, sp = jnp.cos(phi),   jnp.sin(phi)
    ct, st = jnp.cos(theta), jnp.sin(theta)
    cy, sy = jnp.cos(psi),   jnp.sin(psi)
    return jnp.array([
        [cy*ct,  cy*st*sp - sy*cp,  cy*st*cp + sy*sp],
        [sy*ct,  sy*st*sp + cy*cp,  sy*st*cp - cy*sp],
        [-st,    ct*sp,             ct*cp            ]
    ])

def euler_kinematics(phi, theta, p, q, r):
    """Body rates → Euler angle rates  (avoids singularity check for now)"""
    cp, sp = jnp.cos(phi), jnp.sin(phi)
    ct, st = jnp.cos(theta), jnp.sin(theta)
    tt = st / ct
    phi_dot   = p + sp*tt*q + cp*tt*r
    theta_dot =     cp*q    - sp*r
    psi_dot   =    (sp/ct)*q + (cp/ct)*r
    return phi_dot, theta_dot, psi_dot


# ── Dynamics ─────────────────────────────────────────────────────────────────
def f(state, motors):
    """
    Continuous-time EOM.
    Returns dstate/dt  (12,)
    """
    px, py, pz, phi, theta, psi, u, v, w, p, q, r = state
    oFL, oFR, oRL, oRR = motors

    # --- propeller forces & torques ---
    TFL, TFR, TRL, TRR = thrust(oFL), thrust(oFR), thrust(oRL), thrust(oRR)
    QFL, QFR, QRL, QRR = torque(oFL), torque(oFR), torque(oRL), torque(oRR)
    T_tot = TFL + TFR + TRL + TRR

    # --- moments in body frame ---
    Mx = L * ( TFR + TRR - TFL - TRL)   # roll  (+right up)
    My = L * ( TFL + TFR - TRL - TRR)   # pitch (+nose up)
    Mz = (-QFL + QFR + QRL - QRR)       # yaw   (+CCW from above)

    # --- body-frame force ---
    R   = R_body_to_world(phi, theta, psi)
    vel_b = jnp.array([u, v, w])
    grav_b = R.T @ jnp.array([0.0, 0.0, -MASS * G])
    drag_b = -KD * vel_b
    F_b = jnp.array([0.0, 0.0, T_tot]) + grav_b + drag_b

    # --- translational kinematics (world frame) ---
    pos_dot = R @ vel_b                 # [dpx, dpy, dpz]

    # --- Euler angle kinematics ---
    phi_dot, theta_dot, psi_dot = euler_kinematics(phi, theta, p, q, r)

    # --- translational dynamics (body frame) ---
    omega_b = jnp.array([p, q, r])
    vel_dot = F_b / MASS - jnp.cross(omega_b, vel_b)

    # --- rotational dynamics (Euler equations) ---
    M_b    = jnp.array([Mx, My, Mz])
    I_omega = jnp.array([IXX*p, IYY*q, IZZ*r])
    rates_dot = I_INV * (M_b - jnp.cross(omega_b, I_omega))

    return jnp.concatenate([
        pos_dot,
        jnp.array([phi_dot, theta_dot, psi_dot]),
        vel_dot,
        rates_dot,
    ])


def rk4(state, motors, dt):
    """Single RK4 step."""
    k1 = f(state, motors)
    k2 = f(state + 0.5*dt*k1, motors)
    k3 = f(state + 0.5*dt*k2, motors)
    k4 = f(state + dt*k3, motors)
    return state + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
