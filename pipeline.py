#!/usr/bin/env python3
"""
🚁 drone_flow — Complete Multi-Physics Simulation Pipeline

Runs all four physics stages in sequence:

  Step 1 — CFD     JAXFLUIDS compressible NS (ellipse fuselage, Re=200)
  Step 2 — FEM     HEX8 structural analysis of drone arm (Al 6061)
  Step 3 — Dynamics  6-DOF Newton-Euler + BEM propeller + cascaded PID
  Step 4 — Export  Pre-compute trajectory / CFD frames / FEM stress for viewer

Usage:
  JAX_PLATFORMS=cpu conda run -n num_python python3 pipeline.py
  JAX_PLATFORMS=cpu conda run -n num_python python3 pipeline.py --skip-cfd
  JAX_PLATFORMS=cpu conda run -n num_python python3 pipeline.py --only dynamics
"""
import os, sys, argparse, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from pathlib import Path
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

STEPS = ["cfd", "fem", "dynamics", "export"]


def step_cfd():
    print("\n" + "="*60)
    print("🌀  Step 1/4 — CFD (JAXFLUIDS compressible NS)")
    print("="*60)
    t0 = time.time()
    from cfd.run_cfd import run
    result = run(
        out_json=REPO / "data"    / "drone_arm_loads.json",
        out_fig_prefix=REPO / "assets" / "drone",
    )
    print(f"  ✅  CFD done in {time.time()-t0:.0f}s  |  Cp FR = {result['arm_Cp']['FR']:.3f}")
    return result


def step_fem():
    print("\n" + "="*60)
    print("🏗️   Step 2/4 — FEM (HEX8 structural, Al 6061)")
    print("="*60)
    t0 = time.time()
    from fem.drone_arm_fem import run
    result = run(
        in_json  = REPO / "data"    / "drone_arm_loads.json",
        out_json = REPO / "data"    / "drone_arm_fem_result.json",
        out_fig  = REPO / "assets" / "drone_arm_fem.png",
    )
    sf = result["yield_safety_factor"]
    vm = result["max_vonMises_MPa"]
    print(f"  ✅  FEM done in {time.time()-t0:.1f}s  |  σ_VM={vm:.4f} MPa  SF={sf:.0f}×")
    return result


def step_dynamics():
    print("\n" + "="*60)
    print("✈️   Step 3/4 — 6-DOF Flight Dynamics (500 Hz, 20 s)")
    print("="*60)
    t0 = time.time()
    import jax, jax.numpy as jnp, numpy as np
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from dynamics.propeller  import hover_omega, omega_to_rpm
    from dynamics.dynamics   import MASS, G, rk4
    from dynamics.controller import control

    rk4_jit  = jax.jit(rk4)
    ctrl_jit = jax.jit(control)

    DT = 1.0 / 500.0; T_END = 20.0; N = int(T_END / DT)
    state = jnp.zeros(12)

    def ramp(t, t0_, z0, z1, v=0.8):
        dt = jnp.maximum(t - t0_, 0.)
        return z0 + jnp.clip(dt * v / jnp.maximum(jnp.abs(z1-z0), 1e-3), 0., 1.) * (z1-z0)

    def setpoint(t):
        z_d   = jnp.where(t < 1., 0.,
                  jnp.where(t < 10., ramp(t,1.,0.,2.), ramp(t,10.,2.,3.)))
        px_d  = jnp.where(t < 9., 0., ramp(t,9.,0.,4.,0.4))
        py_d  = jnp.where(t < 13., 0., ramp(t,13.,0.,2.,0.3))
        psi_d = jnp.where(t < 6., 0., ramp(t,6.,0.,jnp.radians(60.),0.25))
        return jnp.array([px_d, py_d, z_d, psi_d])

    log_s = np.zeros((N+1, 12)); log_m = np.zeros((N+1, 4)); log_t = np.zeros(N+1)
    log_s[0] = np.array(state)

    for i in range(N):
        t = i * DT; sp = setpoint(t); mot = ctrl_jit(state, sp)
        state = rk4_jit(state, mot, DT)
        log_s[i+1] = np.array(state); log_m[i+1] = np.array(mot); log_t[i+1] = (i+1)*DT

    RPM = omega_to_rpm(log_m)

    def np_ramp(t, t0_, z0, z1, v=0.8):
        dt = np.maximum(t - t0_, 0.)
        return z0 + np.clip(dt * v / max(abs(z1-z0), 1e-3), 0., 1.) * (z1-z0)

    sp_z   = np.where(log_t < 1., 0., np.where(log_t < 10.,
                np_ramp(log_t,1.,0.,2.), np_ramp(log_t,10.,2.,3.)))
    sp_psi = np.where(log_t < 6., 0., np_ramp(log_t,6.,0.,45.,.3/np.pi*180))

    fig, axes = plt.subplots(3, 2, figsize=(15, 11), facecolor="#0a0a14")
    fig.suptitle("Quadcopter 6-DOF dynamics  (500 Hz)  20 s mission", color="white", fontsize=13)
    panels = [
        (0, log_t, log_s[:,2],                "Altitude z (m)",      "#00d4ff", sp_z),
        (1, log_t, log_s[:,0],                "Position x (m)",      "#00d4ff", None),
        (2, log_t, np.degrees(log_s[:,3]),    "Roll φ (°)",          "#ff6b35", None),
        (3, log_t, np.degrees(log_s[:,5]),    "Yaw ψ (°)",           "#ff6b35", sp_psi),
        (4, log_t, RPM[:,0],                  "Motor RPM (FL)",      "#7fff00", None),
        (5, log_t, np.linalg.norm(log_s[:,6:9], axis=1), "Airspeed |V| (m/s)", "#c77dff", None),
    ]
    for ai, tx, y, ylabel, color, sp in panels:
        ax = axes.flat[ai]
        ax.plot(tx, y, color=color, lw=1.2)
        if sp is not None:
            ax.plot(tx, sp, "--", color="white", alpha=0.35, lw=1., label="setpoint")
            ax.legend(facecolor="#0a0a14", edgecolor="#555", labelcolor="white", fontsize=8)
        if "RPM" in ylabel:
            for mi, (mc, ml) in enumerate(zip(["#00d4ff","#ff6b35","#7fff00","#c77dff"],
                                               ["FL","FR","RL","RR"])):
                ax.plot(tx, RPM[:,mi], color=mc, lw=0.9, alpha=0.85, label=ml)
            ax.legend(facecolor="#0a0a14", edgecolor="#555", labelcolor="white",
                      fontsize=8, ncol=2)
        ax.set_facecolor("#0a0a14"); ax.set_xlabel("Time (s)", color="white")
        ax.set_ylabel(ylabel, color="white"); ax.tick_params(colors="white")
        ax.spines[:].set_color("#555"); ax.grid(alpha=0.15, color="white")
    plt.tight_layout()
    out_fig = REPO / "assets" / "dynamics_result.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_fig, dpi=150, bbox_inches="tight", facecolor="#0a0a14")
    plt.close()
    print(f"  Saved: {out_fig}")
    print(f"  ✅  Dynamics done in {time.time()-t0:.1f}s  |  hover {omega_to_rpm(hover_omega(MASS)):.0f} RPM")
    return {"log_states": log_s, "log_motors": log_m, "log_t": log_t}


def step_export():
    print("\n" + "="*60)
    print("🌐  Step 4/4 — Export viewer data (trajectory + CFD frames + FEM)")
    print("="*60)
    t0 = time.time()
    import importlib, importlib.util
    spec = importlib.util.spec_from_file_location("export_all", REPO / "export_all.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print(f"  ✅  Export done in {time.time()-t0:.0f}s")


def main():
    parser = argparse.ArgumentParser(description="drone_flow master pipeline")
    parser.add_argument("--skip-cfd",  action="store_true",
                        help="skip CFD step (use existing data/drone_arm_loads.json)")
    parser.add_argument("--only", choices=STEPS, default=None,
                        help="run a single step only")
    args = parser.parse_args()

    t_start = time.time()
    print("🚁  drone_flow — Multi-Physics Simulation Pipeline")
    print(f"    REPO: {REPO}")

    if args.only:
        {"cfd": step_cfd, "fem": step_fem,
         "dynamics": step_dynamics, "export": step_export}[args.only]()
    else:
        if not args.skip_cfd:
            step_cfd()
        else:
            print("\n⏭️   Skipping CFD — using existing data/drone_arm_loads.json")
        step_fem()
        step_dynamics()
        step_export()

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"✅  Pipeline complete in {elapsed/60:.1f} min")
    print(f"    Viewer: cd {REPO.parent} && python3 -m http.server 7800")
    print(f"    Open:   http://localhost:7800/drone_flow/viewer/")
    print("="*60)


if __name__ == "__main__":
    main()
