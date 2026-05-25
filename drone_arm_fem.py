"""
Drone arm structural analysis — jax-fem-js math ported to Python.

Pipeline:
  drone_arm_loads.json  (Cp from JAXFLUIDS CFD)
      ↓  scale to physical units
  HEX8 FEM cantilever beam (one arm, 20×2×2 elements)
      ↓  assemble K, apply BCs + aero loads + motor weight
  Sparse CG solve → u (nodal displacements)
      ↓
  Von Mises stress at each element centroid
      ↓
  3-panel figure + drone_arm_fem_result.json  (for Three.js)
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ── 1. Load CFD arm pressures ────────────────────────────────────────────────
with open("/home/kaiser/projects/gi/drone_flow/drone_arm_loads.json") as f:
    cfd = json.load(f)

# Physical scaling: nondim CFD → real aero loads
# Drone forward speed 10 m/s, sea-level air
rho_air = 1.225          # kg/m³
U_inf   = 10.0           # m/s
q_phys  = 0.5 * rho_air * U_inf**2   # 61.25 Pa  (physical dynamic pressure)

# Map nondim Cp → physical pressure on arm root
arm_Cp = cfd["arm_Cp"]    # {"FL": -0.077, "FR": -0.175, "RL": ..., "RR": ...}

# ── 2. Arm geometry (from drone.py) ─────────────────────────────────────────
# Each arm: tube, ARM_R=16mm, runs 45° from body edge to motor mount
# Modeled as solid rectangular beam (conservative, same area ≈ same bending stiffness)
ARM_D      = 0.032   # m  (arm outer diameter, 2×ARM_R)
ARM_LENGTH = 0.262   # m  (body-edge to motor, along diagonal)
MOTOR_MASS = 0.100   # kg (motor + ESC typical)
g          = 9.81    # m/s²

# HEX8 mesh dimensions
Nx, Ny, Nz = 20, 2, 2      # elements
Lx, Ly, Lz = ARM_LENGTH, ARM_D, ARM_D  # m

# ── 3. Mesh generator (mirrors jax-fem-js mesh.js) ──────────────────────────
def box_mesh(Nx, Ny, Nz, Lx, Ly, Lz):
    nx, ny, nz = Nx+1, Ny+1, Nz+1
    xs = np.linspace(0, Lx, nx)
    ys = np.linspace(0, Ly, ny)
    zs = np.linspace(0, Lz, nz)
    pts = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                pts.append([xs[i], ys[j], zs[k]])
    pts = np.array(pts)
    def nid(i,j,k): return k*ny*nx + j*nx + i
    cells = []
    for k in range(Nz):
        for j in range(Ny):
            for i in range(Nx):
                cells.append([nid(i,j,k), nid(i+1,j,k), nid(i+1,j+1,k), nid(i,j+1,k),
                               nid(i,j,k+1), nid(i+1,j,k+1), nid(i+1,j+1,k+1), nid(i,j+1,k+1)])
    return pts, np.array(cells)

pts, cells = box_mesh(Nx, Ny, Nz, Lx, Ly, Lz)
n_nodes  = len(pts)
n_elems  = len(cells)
n_dofs   = n_nodes * 3
print(f"Mesh: {n_nodes} nodes, {n_elems} elements, {n_dofs} DOFs")

# ── 4. Material (Aluminium alloy — conservative; CF tube would be stiffer) ──
E, nu_m = 70e9, 0.3
lam = E*nu_m / ((1+nu_m)*(1-2*nu_m))
mu  = E / (2*(1+nu_m))

C = np.zeros((6,6))
C[:3,:3] = lam
for i in range(3): C[i,i] += 2*mu
for i in range(3,6): C[i,i] = mu

# ── 5. HEX8 Gauss quadrature + shape functions (mirrors jax-fem-js basis.js) ─
_gp = 1/np.sqrt(3)
QPTS  = np.array([[s*_gp, t*_gp, u*_gp]
                   for s in [-1,1] for t in [-1,1] for u in [-1,1]])
QWTS  = np.ones(8)

_c = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
               [-1,-1, 1],[1,-1, 1],[1,1, 1],[-1,1, 1]], dtype=float)

def shape_vals(xi, eta, zeta):
    return 0.125*(1+_c[:,0]*xi)*(1+_c[:,1]*eta)*(1+_c[:,2]*zeta)

def shape_grads_ref(xi, eta, zeta):
    dN = np.zeros((8,3))
    dN[:,0] = 0.125*_c[:,0]*(1+_c[:,1]*eta)*(1+_c[:,2]*zeta)
    dN[:,1] = 0.125*(1+_c[:,0]*xi)*_c[:,1]*(1+_c[:,2]*zeta)
    dN[:,2] = 0.125*(1+_c[:,0]*xi)*(1+_c[:,1]*eta)*_c[:,2]
    return dN

# ── 6. Assemble global stiffness K ──────────────────────────────────────────
K_sp = lil_matrix((n_dofs, n_dofs))

for el in cells:
    xe = pts[el]           # (8,3) node coords
    ke = np.zeros((24,24))
    for (xi,eta,zeta), w in zip(QPTS, QWTS):
        dN  = shape_grads_ref(xi, eta, zeta)  # (8,3) ref grads
        J   = xe.T @ dN                        # (3,3) Jacobian
        detJ = np.linalg.det(J)
        JxW  = abs(detJ) * w
        dNdx = dN @ np.linalg.inv(J)          # (8,3) physical grads
        # B matrix (6×24)
        B = np.zeros((6,24))
        for a in range(8):
            B[0,a*3+0] = dNdx[a,0]
            B[1,a*3+1] = dNdx[a,1]
            B[2,a*3+2] = dNdx[a,2]
            B[3,a*3+0] = dNdx[a,1]; B[3,a*3+1] = dNdx[a,0]
            B[4,a*3+1] = dNdx[a,2]; B[4,a*3+2] = dNdx[a,1]
            B[5,a*3+0] = dNdx[a,2]; B[5,a*3+2] = dNdx[a,0]
        ke += JxW * (B.T @ C @ B)
    # Scatter to global
    dofs = np.array([[n*3, n*3+1, n*3+2] for n in el]).flatten()
    for i,di in enumerate(dofs):
        for j,dj in enumerate(dofs):
            K_sp[di, dj] += ke[i, j]

K_sp = csr_matrix(K_sp)
f_vec = np.zeros(n_dofs)
print("K assembled")

# ── 7. Apply loads ───────────────────────────────────────────────────────────
# Use FR arm as the worst-case (highest Cp magnitude)
Cp_arm = arm_Cp["FR"]                 # -0.175
p_aero = Cp_arm * q_phys              # physical pressure [Pa]

# Distributed lateral (y) aero load on x-faces of arm
# Simple node lumping: total_force / n_tip_nodes
n_tip_nx = Ny+1   # nodes at x=Lx
arm_frontal_area = ARM_D * ARM_LENGTH  # m²
F_aero_total = p_aero * arm_frontal_area   # [N] (negative = suction, lateral)

# Tip nodes: x ≈ Lx (within tolerance)
tip_nodes = np.where(pts[:,0] > Lx - 1e-9)[0]
print(f"Tip nodes: {len(tip_nodes)}")

# Motor weight distributed over tip nodes (downward = -z)
F_motor = -MOTOR_MASS * g             # [N]
for n in tip_nodes:
    f_vec[n*3 + 2] += F_motor / len(tip_nodes)   # -z gravity
    f_vec[n*3 + 1] += F_aero_total / len(tip_nodes)  # lateral aero (y)

print(f"Motor load: {F_motor:.3f} N,  Aero load: {F_aero_total:.4f} N (Cp={Cp_arm:.3f})")

# ── 8. Apply Dirichlet BCs (fixed root: x=0, all DOFs = 0) ─────────────────
fixed_nodes = np.where(pts[:,0] < 1e-9)[0]
fixed_dofs  = np.array([[n*3, n*3+1, n*3+2] for n in fixed_nodes]).flatten()

K_mod = K_sp.tolil()
f_mod = f_vec.copy()
for d in fixed_dofs:
    K_mod[d,:] = 0; K_mod[:,d] = 0
    K_mod[d,d] = 1; f_mod[d]   = 0
K_mod = csr_matrix(K_mod)

# ── 9. Solve ─────────────────────────────────────────────────────────────────
import time
t0 = time.time()
u = spsolve(K_mod, f_mod)
print(f"Solved in {time.time()-t0:.2f}s")

u3 = u.reshape(-1, 3)
tip_disp = np.mean(u3[tip_nodes], axis=0)
max_disp  = np.max(np.linalg.norm(u3, axis=1))
print(f"Tip displacement:  Δy={tip_disp[1]*1000:.4f} mm,  Δz={tip_disp[2]*1000:.4f} mm")
print(f"Max displacement:  {max_disp*1000:.4f} mm")

# ── 10. Von Mises stress at element centroids ─────────────────────────────────
vmises = np.zeros(n_elems)
for ei, el in enumerate(cells):
    xe = pts[el]
    dN  = shape_grads_ref(0, 0, 0)     # centroid
    J   = xe.T @ dN
    dNdx = dN @ np.linalg.inv(J)
    B = np.zeros((6, 24))
    for a in range(8):
        B[0,a*3+0] = dNdx[a,0]; B[1,a*3+1] = dNdx[a,1]; B[2,a*3+2] = dNdx[a,2]
        B[3,a*3+0] = dNdx[a,1]; B[3,a*3+1] = dNdx[a,0]
        B[4,a*3+1] = dNdx[a,2]; B[4,a*3+2] = dNdx[a,1]
        B[5,a*3+0] = dNdx[a,2]; B[5,a*3+2] = dNdx[a,0]
    u_el = np.array([[u3[n,0],u3[n,1],u3[n,2]] for n in el]).flatten()
    eps  = B @ u_el
    sig  = C @ eps
    sx,sy,sz,txy,tyz,txz = sig
    vmises[ei] = np.sqrt(0.5*((sx-sy)**2+(sy-sz)**2+(sz-sx)**2+6*(txy**2+tyz**2+txz**2)))

print(f"Max von Mises stress: {vmises.max()/1e6:.4f} MPa")
print(f"Yield safety factor (Al 6061, σy=276 MPa): {276e6/vmises.max():.1f}×")

# ── 11. Visualise ─────────────────────────────────────────────────────────────
deformed = pts + u3 * 200   # scale 200× for visibility

fig = plt.figure(figsize=(16, 5), facecolor="#0a0a14")
fig.suptitle(f"Drone Arm FEM — FR arm  (Aero: Cp={Cp_arm:.3f}, Motor: {MOTOR_MASS*1000:.0f}g)",
             color="white", fontsize=13)

cmap = plt.cm.plasma
norm = plt.Normalize(vmises.min(), vmises.max())

# — Panel 1: 3D deformed arm + von Mises ─────────────────────────────────────
ax1 = fig.add_subplot(131, projection="3d", facecolor="#0a0a14")
for ei, el in enumerate(cells):
    d = deformed[el]
    faces = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,3,7,4],[1,2,6,5]]
    col = cmap(norm(vmises[ei]))
    for face in faces:
        xs = d[face, 0]; ys = d[face, 1]; zs = d[face, 2]
        verts = list(zip(xs, ys, zs))
        poly  = plt.Polygon(list(zip(xs[:2], ys[:2])), closed=True)  # placeholder
        ax1.plot_surface(np.array([[xs[0],xs[1]],[xs[3],xs[2]]]),
                         np.array([[ys[0],ys[1]],[ys[3],ys[2]]]),
                         np.array([[zs[0],zs[1]],[zs[3],zs[2]]]),
                         color=col, alpha=0.8, linewidth=0)
ax1.set_title("Deformed arm\n(200× scale)", color="white", fontsize=9)
ax1.tick_params(colors="white", labelsize=6)
ax1.set_xlabel("x [m]", color="white", fontsize=7)
ax1.set_ylabel("y [m]", color="white", fontsize=7)
ax1.set_zlabel("z [m]", color="white", fontsize=7)

# — Panel 2: Lateral (y) displacement along arm ──────────────────────────────
ax2 = fig.add_subplot(132, facecolor="#0a0a14")
# Sample nodes along centreline (y≈Ly/2, z≈Lz/2)
cx = np.round(pts[:,1], 5) == np.round(Ly/2 * Ny/(Ny+1), 5)  # fallback: all
centre_nodes = np.where((np.abs(pts[:,1]-Ly/2) < Ly/(2*Ny)) &
                         (np.abs(pts[:,2]-Lz/2) < Lz/(2*Nz)))[0]
if len(centre_nodes) == 0:
    centre_nodes = np.arange(n_nodes)
xpos = pts[centre_nodes, 0]
dy   = u3[centre_nodes, 1] * 1000   # mm
dz   = u3[centre_nodes, 2] * 1000

order = np.argsort(xpos)
ax2.plot(xpos[order], dy[order], "c-", lw=2, label="Δy (aero)")
ax2.plot(xpos[order], dz[order], "m-", lw=2, label="Δz (gravity)")
ax2.axhline(0, color="white", lw=0.5, ls="--", alpha=0.4)
ax2.set_xlabel("x along arm [m]", color="white")
ax2.set_ylabel("Displacement [mm]", color="white")
ax2.set_title("Displacement along arm", color="white")
ax2.legend(framealpha=0.2, labelcolor="white")
ax2.tick_params(colors="white")
[sp.set_color("#556") for sp in ax2.spines.values()]

# — Panel 3: Von Mises along arm ──────────────────────────────────────────────
ax3 = fig.add_subplot(133, facecolor="#0a0a14")
el_x = np.array([pts[el,0].mean() for el in cells])
order = np.argsort(el_x)
sc = ax3.scatter(el_x[order], vmises[order]/1e6,
                 c=vmises[order]/1e6, cmap="plasma", s=20)
ax3.set_xlabel("x along arm [m]", color="white")
ax3.set_ylabel("Von Mises stress [MPa]", color="white")
ax3.set_title("Von Mises stress", color="white")
ax3.axhline(276, color="#ff4444", lw=1, ls="--", alpha=0.7, label="Al6061 yield")
ax3.legend(framealpha=0.2, labelcolor="white", fontsize=8)
ax3.tick_params(colors="white")
[sp.set_color("#556") for sp in ax3.spines.values()]
plt.colorbar(sc, ax=ax3, label="MPa").ax.yaxis.set_tick_params(color="white")

plt.tight_layout()
out_fig = "/home/kaiser/projects/gi/drone_flow/drone_arm_fem.png"
plt.savefig(out_fig, dpi=140, bbox_inches="tight", facecolor="#0a0a14")
print(f"Saved: {out_fig}")

# ── 12. Export result JSON for Three.js overlay ──────────────────────────────
result = {
    "arm": "FR",
    "Cp": Cp_arm,
    "q_phys_Pa": float(q_phys),
    "F_aero_N": float(F_aero_total),
    "F_motor_N": float(F_motor),
    "tip_displacement_mm": {"y": float(tip_disp[1]*1000), "z": float(tip_disp[2]*1000)},
    "max_displacement_mm": float(max_disp * 1000),
    "max_vonMises_MPa": float(vmises.max() / 1e6),
    "yield_safety_factor": float(276e6 / vmises.max()),
    "nodes": pts.tolist(),
    "displacements_mm": (u3 * 1000).tolist(),
    "element_vonMises_MPa": (vmises / 1e6).tolist(),
}
out_json = "/home/kaiser/projects/gi/drone_flow/drone_arm_fem_result.json"
with open(out_json, "w") as fp:
    json.dump(result, fp, indent=2)
print(f"Saved: {out_json}")
print(f"\n{'='*50}")
print(f"  Max tip deflection:   Δy={tip_disp[1]*1000:.4f} mm  Δz={tip_disp[2]*1000:.4f} mm")
print(f"  Max von Mises:        {vmises.max()/1e6:.4f} MPa")
print(f"  Al6061 yield (276MPa) safety factor: {276e6/vmises.max():.1f}×")
print(f"  Arm stiffness (F/δ):  {abs(F_motor)/max(abs(tip_disp[2]),1e-12):.1f} N/m")
print(f"{'='*50}")
