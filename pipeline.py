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
    print("✈️   Step 3/4 — Flight Dynamics (multicopter_jax, measurement-fit kT/kQ)")
    print("="*60)
    t0 = time.time()
    from dynamics.jax_copter import run
    result = run(
        T_end=20.0, dt=1.0 / 500.,
        out_fig  = REPO / "assets"  / "dynamics_result.png",
        out_json = REPO / "viewer"  / "trajectory.json",
        verbose=True,
    )
    hr = result["hover_rpm"]
    st = result["settle_time_s"]
    print(f"  ✅  Dynamics done in {time.time()-t0:.1f}s  |  hover {hr:.0f} RPM  |  settle {st:.2f}s")
    return result


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
