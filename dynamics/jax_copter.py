"""Physics-grounded flight dynamics using multicopter_jax.

Replaces BEM + Euler-angle PID with:
  - kT / kQ fitted from RCbenchmark measurement data (motor_10inch_prop.txt)
  - Quaternion 13-state rigid-body dynamics (no gimbal lock)
  - N-rotor allocation matrix via pseudo-inverse (quad, Y6, vtail, ...)
  - Cascaded PD position + attitude controller with yaw tracking

The `run()` function output is backward-compatible with the old step_dynamics()
return dict AND writes trajectory.json in the same format as export_all.py.
"""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from pathlib import Path

# Add /projects/gi to sys.path so that multicopter_jax is importable as a package
# (avoids name collision with drone_flow/dynamics/)
_GI = Path(__file__).resolve().parents[2]
if str(_GI) not in sys.path:
    sys.path.insert(0, str(_GI))

import jax
import jax.numpy as jnp
import numpy as np

from multicopter_jax.loader     import copter_params
from multicopter_jax.dynamics   import rk4_step, euler_from_quat
from multicopter_jax.controller import default_gains

# ── Default configuration ────────────────────────────────────────────────────
DESIGN_ROOT = Path("/home/kaiser/projects/gi/multicopter_design")
QUAD_XML    = DESIGN_ROOT / "resources" / "copter" / "quadcopter.xml"
DEFAULT_MASS = 1.0   # kg (frame + motors + battery + electronics)


# ── Controller ───────────────────────────────────────────────────────────────
def pd_ctrl_yaw(state, target_pos, target_yaw, int_err, dt, gains, params):
    """Cascaded PD with yaw tracking.

    Extends controller.pd_ctrl to track a desired yaw angle (rad) by adding
    a proportional Mz term on top of the yaw-rate damping.
    """
    from multicopter_jax.controller import pd_ctrl as _pd_ctrl
    # Run standard PD (no yaw setpoint)
    thrust_raw, new_int_err = _pd_ctrl(state, target_pos, int_err, dt, gains, params)

    # Add yaw correction:  ΔMz = Kp_yaw * (ψ_des - ψ) - Kd_yaw * r
    q     = state[3:7]
    omega = state[10:13]
    roll, pitch, yaw = euler_from_quat(q)
    yaw_err = _wrap_angle(target_yaw - yaw)

    A_pinv = jnp.array(params["A_pinv"])
    Kp_yaw = float(gains["Kp_att"][2]) * 0.5
    Kd_yaw = float(gains["Kd_att"][2])
    dMz    = Kp_yaw * yaw_err - Kd_yaw * omega[2]

    # Build pure-Mz correction via allocation
    cmd_yaw = jnp.array([0., 0., 0., dMz])
    thrust_yaw = A_pinv @ cmd_yaw
    thrust = jnp.clip(thrust_raw + thrust_yaw, 0., params["T_max"])
    return thrust, new_int_err


def _wrap_angle(a):
    """Wrap angle to [-π, π]."""
    return (a + jnp.pi) % (2 * jnp.pi) - jnp.pi


# ── Mission schedule ─────────────────────────────────────────────────────────
def build_mission(T_end: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Returns (t_arr, setpoints) where setpoints[:, :] = [x, y, z, yaw_rad]."""
    t = np.arange(0, T_end + dt, dt)

    def np_ramp(arr, t0, z0, z1, v=0.8):
        d = np.maximum(arr - t0, 0.)
        return z0 + np.clip(d * v / max(abs(z1 - z0), 1e-3), 0., 1.) * (z1 - z0)

    sp_x   = np.where(t < 9.,  0., np_ramp(t, 9.,  0., 4.,  0.4))
    sp_y   = np.where(t < 13., 0., np_ramp(t, 13., 0., 2.,  0.3))
    sp_z   = np.where(t < 1.,  0.,
             np.where(t < 10., np_ramp(t, 1., 0., 2.),
                                np_ramp(t, 10., 2., 3.)))
    sp_yaw = np.where(t < 6.,  0., np_ramp(t, 6., 0., np.radians(60.), 0.25))

    return t, np.stack([sp_x, sp_y, sp_z, sp_yaw], axis=1)


# ── Main entry point ─────────────────────────────────────────────────────────
def run(xml_file=None, total_mass: float = DEFAULT_MASS,
        T_end: float = 20.0, dt: float = 1.0 / 500.,
        out_fig=None, out_json=None, verbose: bool = True) -> dict:
    """Run physics-based waypoint mission, output figure + trajectory.json.

    Args:
        xml_file:    copter XML (default: quadcopter.xml from multicopter_design)
        total_mass:  drone mass in kg
        T_end:       simulation duration (s)
        dt:          integration timestep (s), default 1/500
        out_fig:     path for the dark-theme 6-panel plot PNG
        out_json:    path for trajectory.json (viewer compatible)
        verbose:     print progress

    Returns:
        dict with log_states (N+1, 13), log_motors (N+1, N), log_t (N+1,),
        RPM, euler_deg, params, hover_rpm, settle_time_s
    """
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xml = Path(xml_file) if xml_file else QUAD_XML
    params = copter_params(xml, total_mass=total_mass)
    N = params["N"]
    gains = default_gains(params)

    if verbose:
        kT0 = float(params["kT"][0])
        T_h = params["T_hover"]
        hover_rpm = float(np.sqrt(T_h / kT0)) * (60 / (2 * np.pi))
        print(f"  Copter: {N} rotors  |  {total_mass:.2f} kg  |  "
              f"kT={kT0:.3e}  |  hover {hover_rpm:.0f} RPM")
    else:
        kT0 = float(params["kT"][0])
        T_h = params["T_hover"]
        hover_rpm = float(np.sqrt(T_h / kT0)) * (60 / (2 * np.pi))

    t_arr, setpoints = build_mission(T_end, dt)
    n_steps = len(t_arr) - 1

    # JIT-compiled step
    @jax.jit
    def _step(state, int_err, target_pos, target_yaw):
        thrust, new_ie = pd_ctrl_yaw(
            state, target_pos, target_yaw, int_err, dt, gains, params)
        return rk4_step(state, thrust, dt, params), thrust, new_ie

    # Storage
    log_s = np.zeros((n_steps + 1, 13))
    log_m = np.zeros((n_steps + 1, N))

    state   = jnp.concatenate([
        jnp.zeros(3),
        jnp.array([1., 0., 0., 0.]),
        jnp.zeros(6),
    ])
    int_err = jnp.zeros(3)
    log_s[0] = np.array(state)

    for i in range(n_steps):
        sp      = setpoints[i]
        target  = jnp.array([sp[0], sp[1], sp[2]])
        state, thrust, int_err = _step(state, int_err, target, float(sp[3]))
        log_s[i + 1] = np.array(state)
        log_m[i + 1] = np.array(thrust)

    # Derived quantities
    kT = params["kT"]
    RPM       = np.sqrt(np.maximum(log_m / kT, 0.)) * (60. / (2 * np.pi))
    euler_all = np.array([euler_from_quat(jnp.array(log_s[i, 3:7]))
                          for i in range(n_steps + 1)])

    # Settle time: first time |z - z_final| < 5 cm after z ramp complete
    z_final = setpoints[-1, 2]
    sp_z    = setpoints[:n_steps + 1, 2]
    err_z   = np.abs(log_s[:, 2] - sp_z)
    mask    = (t_arr[:n_steps + 1] > 3.5) & (err_z < 0.05)
    settle  = float(t_arr[np.where(mask)[0][0]]) if mask.any() else float(T_end)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(15, 11), facecolor="#0a0a14")
    fig.suptitle(
        f"Quadcopter  6-DOF quaternion dynamics  (kT/kQ measurement fit)  {T_end:.0f} s mission",
        color="white", fontsize=13,
    )
    t_plot = t_arr[:n_steps + 1]
    sp_psi_deg = np.degrees(setpoints[:n_steps + 1, 3])

    panels = [
        (0, log_s[:, 2],                      "Altitude z (m)",      "#00d4ff", setpoints[:n_steps + 1, 2]),
        (1, log_s[:, 0],                       "Position x (m)",      "#00d4ff", None),
        (2, np.degrees(euler_all[:, 0]),       "Roll φ (°)",          "#ff6b35", None),
        (3, np.degrees(euler_all[:, 2]),       "Yaw ψ (°)",           "#ff6b35", sp_psi_deg),
        (4, RPM[:, 0],                         f"Motor M1 RPM",       "#7fff00", None),
        (5, np.linalg.norm(log_s[:, 7:10], axis=1), "Airspeed |V| (m/s)", "#c77dff", None),
    ]
    colors_m = ["#00d4ff", "#ff6b35", "#7fff00", "#c77dff", "#e040fb", "#ff9800"]
    for ai, (idx, y, ylabel, color, sp_ov) in enumerate(panels):
        ax = axes.flat[idx]
        ax.plot(t_plot, y, color=color, lw=1.2)
        if sp_ov is not None:
            ax.plot(t_plot, sp_ov, "--", color="white", alpha=0.35, lw=1., label="setpoint")
            ax.legend(facecolor="#0a0a14", edgecolor="#555", labelcolor="white", fontsize=8)
        if "RPM" in ylabel:
            for mi in range(N):
                ax.plot(t_plot, RPM[:, mi], color=colors_m[mi % len(colors_m)],
                        lw=0.9, alpha=0.85, label=f"M{mi+1}")
            ax.legend(facecolor="#0a0a14", edgecolor="#555", labelcolor="white",
                      fontsize=8, ncol=min(N, 3))
        ax.set_facecolor("#0a0a14"); ax.set_xlabel("Time (s)", color="white")
        ax.set_ylabel(ylabel, color="white"); ax.tick_params(colors="white")
        ax.spines[:].set_color("#555"); ax.grid(alpha=0.15, color="white")

    plt.tight_layout()
    if out_fig:
        Path(out_fig).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_fig, dpi=150, bbox_inches="tight", facecolor="#0a0a14")
        if verbose:
            print(f"  Saved: {out_fig}")
    plt.close()

    # ── trajectory.json ──────────────────────────────────────────────────────
    if out_json:
        FPS   = 30
        RATIO = max(1, int(round(1.0 / (FPS * dt))))
        frames = []
        for i in range(0, n_steps + 1, RATIO):
            s   = log_s[i]
            e   = euler_all[i]
            rpm_row = RPM[i]
            frames.append({
                "t":   round(float(t_arr[i]), 4),
                "pos": [round(float(s[j]), 4) for j in range(3)],
                "att": [round(float(np.degrees(e[j])), 3) for j in range(3)],
                "vel": [round(float(s[j]), 4) for j in range(7, 10)],
                "rpm": [round(float(rpm_row[mi]), 1) for mi in range(N)],
                "sp":  {
                    "z":       round(float(setpoints[i, 2]), 3),
                    "psi_deg": round(float(np.degrees(setpoints[i, 3])), 1),
                },
            })
        traj = {
            "fps": FPS, "dt": round(1. / FPS, 4),
            "t_end": T_end, "n_frames": len(frames),
            "hover_rpm": round(hover_rpm, 1),
            "n_rotors": N,
            "frames": frames,
        }
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(traj, f, separators=(",", ":"))
        if verbose:
            print(f"  trajectory.json: {len(frames)} frames @ {FPS} fps")

    return {
        "log_states":   log_s,
        "log_motors":   log_m,
        "log_t":        t_arr[:n_steps + 1],
        "RPM":          RPM,
        "euler_deg":    np.degrees(euler_all),
        "params":       params,
        "hover_rpm":    hover_rpm,
        "settle_time_s": settle,
    }
