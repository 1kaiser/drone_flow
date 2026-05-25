"""
Propeller actuator-disc model.
T = CT * rho * n^2 * D^4   (thrust)
Q = CQ * rho * n^2 * D^5   (torque)
n = omega / (2*pi)  [rev/s]
"""
import jax.numpy as jnp

RHO   = 1.225    # kg/m³ air density at sea level
D     = 0.203    # m   8-inch prop diameter
CT    = 0.109    # thrust coefficient
CQ    = 0.0095   # torque coefficient
G     = 9.81     # m/s²

# Precomputed per-rev² factors
_kT = CT * RHO * D**4 / (2 * jnp.pi)**2   # T = _kT * omega^2
_kQ = CQ * RHO * D**5 / (2 * jnp.pi)**2   # Q = _kQ * omega^2


def thrust(omega):
    return _kT * omega**2

def torque(omega):
    return _kQ * omega**2

def hover_omega(mass):
    """omega per motor for steady hover"""
    T_each = mass * G / 4.0
    return jnp.sqrt(T_each / _kT)

def omega_to_rpm(omega):
    return omega * 60.0 / (2.0 * jnp.pi)
