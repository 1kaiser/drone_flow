"""
Drone arm structural analysis — HEX8 FEM.

Pipeline:
  data/drone_arm_loads.json  (Cp from JAXFLUIDS CFD)
      ↓  scale to physical units
  HEX8 FEM cantilever beam  (FR arm, 20×2×2 elements, Al 6061)
      ↓  assemble K, apply BCs + aero loads + motor weight
  Sparse solve → u (nodal displacements)
      ↓
  Von Mises stress at element centroids
      ↓
  figures/drone_arm_fem.png  +  data/drone_arm_fem_result.json
"""
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import json, time, sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def run(in_json=None, out_json=None, out_fig=None):
    """Run HEX8 FEM on drone FR arm. Returns result dict."""
    if in_json  is None: in_json  = REPO / "data"    / "drone_arm_loads.json"
    if out_json is None: out_json = REPO / "data"    / "drone_arm_fem_result.json"
    if out_fig  is None: out_fig  = REPO / "assets" / "drone_arm_fem.png"

    # ── 1. Load CFD arm pressures ─────────────────────────────────────────────
    with open(in_json) as f:
        cfd = json.load(f)

    rho_air = 1.225; U_inf = 10.0
    q_phys  = 0.5 * rho_air * U_inf ** 2
    arm_Cp  = cfd["arm_Cp"]

    # ── 2. Arm geometry ───────────────────────────────────────────────────────
    ARM_D = 0.032; ARM_LENGTH = 0.262; MOTOR_MASS = 0.100; g = 9.81
    Nx, Ny, Nz = 20, 2, 2
    Lx, Ly, Lz = ARM_LENGTH, ARM_D, ARM_D

    # ── 3. Mesh ───────────────────────────────────────────────────────────────
    def box_mesh(Nx, Ny, Nz, Lx, Ly, Lz):
        nx, ny, nz = Nx+1, Ny+1, Nz+1
        xs = np.linspace(0, Lx, nx)
        ys = np.linspace(0, Ly, ny)
        zs = np.linspace(0, Lz, nz)
        pts = np.array([[xs[i], ys[j], zs[k]]
                        for k in range(nz) for j in range(ny) for i in range(nx)])
        def nid(i,j,k): return k*ny*nx + j*nx + i
        cells = np.array([
            [nid(i,j,k), nid(i+1,j,k), nid(i+1,j+1,k), nid(i,j+1,k),
             nid(i,j,k+1), nid(i+1,j,k+1), nid(i+1,j+1,k+1), nid(i,j+1,k+1)]
            for k in range(Nz) for j in range(Ny) for i in range(Nx)
        ])
        return pts, cells

    pts, cells = box_mesh(Nx, Ny, Nz, Lx, Ly, Lz)
    n_nodes = len(pts); n_elems = len(cells); n_dofs = n_nodes * 3
    print(f"  Mesh: {n_nodes} nodes, {n_elems} elements, {n_dofs} DOFs")

    # ── 4. Material (Al 6061) ─────────────────────────────────────────────────
    E, nu_m = 70e9, 0.3
    lam = E * nu_m / ((1+nu_m) * (1-2*nu_m))
    mu  = E / (2 * (1+nu_m))
    C = np.diag([lam+2*mu, lam+2*mu, lam+2*mu, mu, mu, mu])
    C[:3, :3] = lam; C[0,0] += 2*mu; C[1,1] += 2*mu; C[2,2] += 2*mu

    # ── 5. HEX8 shape functions ───────────────────────────────────────────────
    _gp = 1 / np.sqrt(3)
    QPTS = np.array([[s*_gp, t*_gp, u*_gp]
                     for s in [-1,1] for t in [-1,1] for u in [-1,1]])
    QWTS = np.ones(8)
    _c   = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                     [-1,-1, 1],[1,-1, 1],[1,1, 1],[-1,1, 1]], dtype=float)

    def shape_grads_ref(xi, eta, zeta):
        dN = np.zeros((8, 3))
        dN[:,0] = 0.125 * _c[:,0] * (1+_c[:,1]*eta)  * (1+_c[:,2]*zeta)
        dN[:,1] = 0.125 * (1+_c[:,0]*xi) * _c[:,1]   * (1+_c[:,2]*zeta)
        dN[:,2] = 0.125 * (1+_c[:,0]*xi) * (1+_c[:,1]*eta) * _c[:,2]
        return dN

    # ── 6. Assemble K ─────────────────────────────────────────────────────────
    K_sp = lil_matrix((n_dofs, n_dofs))
    for el in cells:
        xe = pts[el]; ke = np.zeros((24, 24))
        for (xi, eta, zeta), w in zip(QPTS, QWTS):
            dN   = shape_grads_ref(xi, eta, zeta)
            J    = xe.T @ dN; detJ = np.linalg.det(J)
            dNdx = dN @ np.linalg.inv(J)
            B = np.zeros((6, 24))
            for a in range(8):
                B[0,a*3+0]=dNdx[a,0]; B[1,a*3+1]=dNdx[a,1]; B[2,a*3+2]=dNdx[a,2]
                B[3,a*3+0]=dNdx[a,1]; B[3,a*3+1]=dNdx[a,0]
                B[4,a*3+1]=dNdx[a,2]; B[4,a*3+2]=dNdx[a,1]
                B[5,a*3+0]=dNdx[a,2]; B[5,a*3+2]=dNdx[a,0]
            ke += abs(detJ) * w * (B.T @ C @ B)
        dofs = np.array([[n*3, n*3+1, n*3+2] for n in el]).flatten()
        for i, di in enumerate(dofs):
            for j, dj in enumerate(dofs):
                K_sp[di, dj] += ke[i, j]

    K_sp  = csr_matrix(K_sp)
    f_vec = np.zeros(n_dofs)
    print("  K assembled")

    # ── 7. Loads ──────────────────────────────────────────────────────────────
    Cp_arm = arm_Cp["FR"]
    F_aero_total = Cp_arm * q_phys * ARM_D * ARM_LENGTH
    tip_nodes    = np.where(pts[:,0] > Lx - 1e-9)[0]
    F_motor = -MOTOR_MASS * g
    for n in tip_nodes:
        f_vec[n*3+2] += F_motor / len(tip_nodes)
        f_vec[n*3+1] += F_aero_total / len(tip_nodes)

    # ── 8. Dirichlet BCs (fixed root) ─────────────────────────────────────────
    fixed_dofs = np.array([[n*3, n*3+1, n*3+2]
                           for n in np.where(pts[:,0] < 1e-9)[0]]).flatten()
    K_mod = K_sp.tolil(); f_mod = f_vec.copy()
    for d in fixed_dofs:
        K_mod[d,:] = 0; K_mod[:,d] = 0; K_mod[d,d] = 1; f_mod[d] = 0
    K_mod = csr_matrix(K_mod)

    # ── 9. Solve ──────────────────────────────────────────────────────────────
    t0 = time.time()
    u  = spsolve(K_mod, f_mod)
    print(f"  Solved in {time.time()-t0:.2f}s")
    u3 = u.reshape(-1, 3)
    tip_disp = np.mean(u3[tip_nodes], axis=0)

    # ── 10. Von Mises ─────────────────────────────────────────────────────────
    vmises = np.zeros(n_elems)
    for ei, el in enumerate(cells):
        xe = pts[el]; dN = shape_grads_ref(0, 0, 0)
        dNdx = dN @ np.linalg.inv(xe.T @ dN)
        B = np.zeros((6, 24))
        for a in range(8):
            B[0,a*3+0]=dNdx[a,0]; B[1,a*3+1]=dNdx[a,1]; B[2,a*3+2]=dNdx[a,2]
            B[3,a*3+0]=dNdx[a,1]; B[3,a*3+1]=dNdx[a,0]
            B[4,a*3+1]=dNdx[a,2]; B[4,a*3+2]=dNdx[a,1]
            B[5,a*3+0]=dNdx[a,2]; B[5,a*3+2]=dNdx[a,0]
        u_el = np.array([[u3[n,0],u3[n,1],u3[n,2]] for n in el]).flatten()
        eps  = B @ u_el; sig = C @ eps
        sx,sy,sz,txy,tyz,txz = sig
        vmises[ei] = np.sqrt(0.5*((sx-sy)**2+(sy-sz)**2+(sz-sx)**2
                                   + 6*(txy**2+tyz**2+txz**2)))

    sf = 276e6 / vmises.max()
    print(f"  Tip Δz={tip_disp[2]*1000:.4f} mm  σ_VM={vmises.max()/1e6:.4f} MPa  SF={sf:.0f}×")

    # ── 11. Figure ────────────────────────────────────────────────────────────
    deformed = pts + u3 * 200
    fig = plt.figure(figsize=(16, 5), facecolor="#0a0a14")
    fig.suptitle(f"Drone Arm FEM — FR arm  (Cp={Cp_arm:.3f}, Motor={MOTOR_MASS*1000:.0f}g)",
                 color="white", fontsize=13)

    cmap = plt.cm.plasma
    norm = plt.Normalize(vmises.min(), vmises.max())

    ax1 = fig.add_subplot(131, projection="3d", facecolor="#0a0a14")
    for ei, el in enumerate(cells):
        d = deformed[el]; col = cmap(norm(vmises[ei]))
        for face in [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,3,7,4],[1,2,6,5]]:
            xs = d[face,0]; ys = d[face,1]; zs = d[face,2]
            ax1.plot_surface(np.array([[xs[0],xs[1]],[xs[3],xs[2]]]),
                             np.array([[ys[0],ys[1]],[ys[3],ys[2]]]),
                             np.array([[zs[0],zs[1]],[zs[3],zs[2]]]),
                             color=col, alpha=0.8, linewidth=0)
    ax1.set_title("Deformed (200×)", color="white", fontsize=9)
    ax1.tick_params(colors="white", labelsize=6)

    ax2 = fig.add_subplot(132, facecolor="#0a0a14")
    centre_nodes = np.where((np.abs(pts[:,1]-Ly/2) < Ly/(2*Ny)) &
                             (np.abs(pts[:,2]-Lz/2) < Lz/(2*Nz)))[0]
    if not len(centre_nodes): centre_nodes = np.arange(n_nodes)
    xpos = pts[centre_nodes, 0]; order = np.argsort(xpos)
    ax2.plot(xpos[order], u3[centre_nodes,1][order]*1000, "c-", lw=2, label="Δy (aero)")
    ax2.plot(xpos[order], u3[centre_nodes,2][order]*1000, "m-", lw=2, label="Δz (gravity)")
    ax2.axhline(0, color="white", lw=0.5, ls="--", alpha=0.4)
    ax2.set_xlabel("x [m]", color="white"); ax2.set_ylabel("Displacement [mm]", color="white")
    ax2.set_title("Displacement along arm", color="white")
    ax2.legend(framealpha=0.2, labelcolor="white"); ax2.tick_params(colors="white")
    [sp.set_color("#556") for sp in ax2.spines.values()]

    ax3 = fig.add_subplot(133, facecolor="#0a0a14")
    el_x  = np.array([pts[el,0].mean() for el in cells]); order3 = np.argsort(el_x)
    sc    = ax3.scatter(el_x[order3], vmises[order3]/1e6, c=vmises[order3]/1e6, cmap="plasma", s=20)
    ax3.axhline(276, color="#ff4444", lw=1, ls="--", alpha=0.7, label="Al6061 yield")
    ax3.set_xlabel("x [m]", color="white"); ax3.set_ylabel("Von Mises [MPa]", color="white")
    ax3.set_title("Von Mises stress", color="white")
    ax3.legend(framealpha=0.2, labelcolor="white", fontsize=8); ax3.tick_params(colors="white")
    [sp.set_color("#556") for sp in ax3.spines.values()]
    plt.colorbar(sc, ax=ax3, label="MPa").ax.yaxis.set_tick_params(color="white")

    plt.tight_layout()
    Path(out_fig).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_fig, dpi=140, bbox_inches="tight", facecolor="#0a0a14")
    plt.close()
    print(f"  Saved: {out_fig}")

    # ── 12. Export JSON ───────────────────────────────────────────────────────
    result = {
        "arm": "FR", "Cp": Cp_arm, "q_phys_Pa": float(q_phys),
        "F_aero_N": float(F_aero_total), "F_motor_N": float(F_motor),
        "tip_displacement_mm": {"y": float(tip_disp[1]*1000), "z": float(tip_disp[2]*1000)},
        "max_displacement_mm": float(np.max(np.linalg.norm(u3, axis=1)) * 1000),
        "max_vonMises_MPa":   float(vmises.max() / 1e6),
        "yield_safety_factor": float(sf),
        "nodes":                pts.tolist(),
        "displacements_mm":     (u3 * 1000).tolist(),
        "element_vonMises_MPa": (vmises / 1e6).tolist(),
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fp:
        json.dump(result, fp, indent=2)
    print(f"  Saved: {out_json}")
    return result


if __name__ == "__main__":
    r = run()
    print(f"\nMax von Mises: {r['max_vonMises_MPa']:.4f} MPa  SF: {r['yield_safety_factor']:.0f}×")
