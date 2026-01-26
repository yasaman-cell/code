# =========================================================
# Fully Physics-Based PINN for Fluid-Structure Interaction (FSI)
# COMPLETE EDITED VERSION (Physics-based solid via Neo-Hookean PK1)
# - Fluid net: (x,y,t) -> (u,v,p)
# - Solid net: (x,y,t) -> (dx,dy)
# - Solid physics: F = I + grad(d), Neo-Hookean PK1 P(F), f = div(P)
# - Solid momentum: div(P) - rho_s * d_tt = 0
# - Coupling:
#   * Velocity coupling inside solid domain: u_f = d_t
#   * Force coupling: inject f_solid as body force into Navier–Stokes (masked by chi)
# - Includes essential solid IC and J-positivity penalty
# =========================================================

import torch
import torch.nn as nn
import torch.autograd as autograd
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# -------------------------
# MLP Network
# -------------------------
class MLP(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        logger.debug(f"MLP created with layers: {layers}")

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = torch.tanh(self.layers[i](x))
        return self.layers[-1](x)

# -------------------------
# Fluid PINN: (x,y,t) -> (u,v,p)
# -------------------------
class FluidPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = MLP([3, 128, 128, 128, 3])
        logger.info("FluidPINN network initialized")

    def forward(self, x, y, t):
        out = self.net(torch.cat([x, y, t], dim=1))
        u = out[:, 0:1]
        v = out[:, 1:2]
        p = out[:, 2:3]
        return u, v, p

# -------------------------
# Solid PINN: (x,y,t) -> (dx,dy)
# -------------------------
class SolidPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = MLP([3, 128, 128, 128, 2])
        logger.info("SolidPINN network initialized")

    def forward(self, x, y, t):
        out = self.net(torch.cat([x, y, t], dim=1))
        dx = out[:, 0:1]
        dy = out[:, 1:2]
        return dx, dy

# =========================================================
# Geometry indicator: chi(x,y,t) = 1 inside solid, 0 outside
# Example: static circular solid
# =========================================================
def chi_solid(x, y, t=None, center=(0.5, 0.5), radius=0.15):
    cx, cy = center
    r2 = radius * radius
    inside = ((x - cx)**2 + (y - cy)**2 <= r2).float()
    return inside

# =========================================================
# Solid kinematics and Neo-Hookean PK1
# =========================================================
def solid_velocity(dx, dy, t):
    # u_s = d(dx)/dt , v_s = d(dy)/dt
    u_s = autograd.grad(dx, t, torch.ones_like(dx), create_graph=True)[0]
    v_s = autograd.grad(dy, t, torch.ones_like(dy), create_graph=True)[0]
    return u_s, v_s

def deformation_gradient(dx, dy, x, y):
    # F = I + grad(d)
    dx_x = autograd.grad(dx, x, torch.ones_like(dx), create_graph=True)[0]
    dx_y = autograd.grad(dx, y, torch.ones_like(dx), create_graph=True)[0]
    dy_x = autograd.grad(dy, x, torch.ones_like(dy), create_graph=True)[0]
    dy_y = autograd.grad(dy, y, torch.ones_like(dy), create_graph=True)[0]

    F11 = 1.0 + dx_x
    F12 = dx_y
    F21 = dy_x
    F22 = 1.0 + dy_y
    return F11, F12, F21, F22

def neo_hookean_PK1(dx, dy, x, y, mu=1.0, lam=1.0, epsJ=1e-6):
    """
    Compressible Neo-Hookean (PK1):
      P = mu (F - F^{-T}) + lam ln(J) F^{-T}
    In 2D: F is 2x2, J = det(F)
    """
    F11, F12, F21, F22 = deformation_gradient(dx, dy, x, y)

    J = F11*F22 - F12*F21

    # Safety: avoid log of non-positive J
    # We'll use J_safe for log/inversion, and separately penalize invalid J
    J_safe = torch.clamp(J, min=epsJ)

    # F^{-T} for 2x2:
    FinvT11 =  F22 / J_safe
    FinvT12 = -F21 / J_safe
    FinvT21 = -F12 / J_safe
    FinvT22 =  F11 / J_safe

    logJ = torch.log(J_safe)

    P11 = mu*(F11 - FinvT11) + lam*logJ*FinvT11
    P12 = mu*(F12 - FinvT12) + lam*logJ*FinvT12
    P21 = mu*(F21 - FinvT21) + lam*logJ*FinvT21
    P22 = mu*(F22 - FinvT22) + lam*logJ*FinvT22

    return (P11, P12, P21, P22, J)

def solid_internal_force_PK1(P11, P12, P21, P22, x, y):
    """
    f = div(P) in reference coordinates:
      fx = dP11/dx + dP12/dy
      fy = dP21/dx + dP22/dy
    """
    fx = autograd.grad(P11, x, torch.ones_like(P11), create_graph=True)[0] + \
         autograd.grad(P12, y, torch.ones_like(P12), create_graph=True)[0]

    fy = autograd.grad(P21, x, torch.ones_like(P21), create_graph=True)[0] + \
         autograd.grad(P22, y, torch.ones_like(P22), create_graph=True)[0]
    return fx, fy

def compute_solid_force(dx, dy, x, y, mu=1.0, lam=1.0, epsJ=1e-6):
    P11, P12, P21, P22, J = neo_hookean_PK1(dx, dy, x, y, mu=mu, lam=lam, epsJ=epsJ)
    fx, fy = solid_internal_force_PK1(P11, P12, P21, P22, x, y)
    return fx, fy, J

# =========================================================
# Navier-Stokes residuals WITH body force
# res_u = rho (u_t + u u_x + v u_y) + p_x - nu Lap(u) - fx
# res_v = rho (v_t + u v_x + v v_y) + p_y - nu Lap(v) - fy
# res_c = u_x + v_y
# =========================================================
def navier_stokes_residual(u, v, p, x, y, t, fx=None, fy=None, rho_f=1.0, nu=0.01):
    u_t = autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    v_t = autograd.grad(v, t, torch.ones_like(v), create_graph=True)[0]

    u_x = autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_y = autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]
    v_x = autograd.grad(v, x, torch.ones_like(v), create_graph=True)[0]
    v_y = autograd.grad(v, y, torch.ones_like(v), create_graph=True)[0]

    p_x = autograd.grad(p, x, torch.ones_like(p), create_graph=True)[0]
    p_y = autograd.grad(p, y, torch.ones_like(p), create_graph=True)[0]

    u_xx = autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
    v_xx = autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0]
    v_yy = autograd.grad(v_y, y, torch.ones_like(v_y), create_graph=True)[0]

    if fx is None:
        fx = torch.zeros_like(u)
    if fy is None:
        fy = torch.zeros_like(v)

    res_u = rho_f * (u_t + u*u_x + v*u_y) + p_x - nu * (u_xx + u_yy) - fx
    res_v = rho_f * (v_t + u*v_x + v*v_y) + p_y - nu * (v_xx + v_yy) - fy
    res_c = u_x + v_y

    return res_u, res_v, res_c

# =========================================================
# Losses
# =========================================================
def loss_fluid(u, v, p, x, y, t, fx=None, fy=None, rho_f=1.0, nu=0.01):
    ru, rv, rc = navier_stokes_residual(u, v, p, x, y, t, fx=fx, fy=fy, rho_f=rho_f, nu=nu)
    return torch.mean(ru**2) + torch.mean(rv**2) + torch.mean(rc**2)

def loss_solid(dx, dy, x, y, t, rho_s=1.0, mu=1.0, lam=1.0, epsJ=1e-6):
    # Compute d_tt directly (more stable/clean)
    dx_t = autograd.grad(dx, t, torch.ones_like(dx), create_graph=True)[0]
    dy_t = autograd.grad(dy, t, torch.ones_like(dy), create_graph=True)[0]
    dx_tt = autograd.grad(dx_t, t, torch.ones_like(dx_t), create_graph=True)[0]
    dy_tt = autograd.grad(dy_t, t, torch.ones_like(dy_t), create_graph=True)[0]

    fx, fy, J = compute_solid_force(dx, dy, x, y, mu=mu, lam=lam, epsJ=epsJ)

    # Solid momentum residual: div(P) - rho*d_tt = 0
    Lm = torch.mean((fx - rho_s*dx_tt)**2 + (fy - rho_s*dy_tt)**2)

    # Penalty for invalid/near-singular deformation (J <= epsJ)
    # This helps Neo-Hookean stability a LOT.
    LJ = torch.mean(torch.relu(epsJ - J)**2)

    return Lm + 1.0 * LJ

def loss_velocity_coupling(u_f, v_f, u_s, v_s):
    return torch.mean((u_f - u_s)**2 + (v_f - v_s)**2)

# -------------------------
# Solid initial condition: d(x,y,0) = 0 (essential)
# -------------------------
def loss_ic_solid(solid_net, N=500):
    x = torch.rand(N, 1, device=device)
    y = torch.rand(N, 1, device=device)
    t = torch.zeros(N, 1, device=device)
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)

    dx, dy = solid_net(x, y, t)
    return torch.mean(dx**2 + dy**2)

# -------------------------
# Optional: fluid BC/IC placeholders
# Replace with your real constraints
# -------------------------
def loss_bc_ic(fluid_net, solid_net):
    # Example placeholders:
    # - No-slip at outer boundary
    # - Inlet velocity profile
    # - Pressure outlet
    # - Fluid initial condition
    return torch.tensor(0.0, device=device)

# =========================================================
# TOTAL LOSS
# =========================================================
def total_loss(fluid_net, solid_net,
               x_f, y_f, t_f,
               x_s, y_s, t_s,
               rho_f=1.0, nu=0.01,
               rho_s=1.0, mu_s=1.0, lam_s=1.0,
               w_f=1.0, w_s=1.0, w_c=1.0, w_bc=1.0, w_ic=1.0,
               solid_center=(0.5, 0.5), solid_radius=0.15,
               epsJ=1e-6):

    # ---------
    # Fluid PDE on fluid points (body force included)
    # ---------
    u_f, v_f, p_f = fluid_net(x_f, y_f, t_f)

    # Evaluate solid displacement everywhere (on fluid points), compute solid force
    dx_f, dy_f = solid_net(x_f, y_f, t_f)
    fx_f, fy_f, _Jf = compute_solid_force(dx_f, dy_f, x_f, y_f, mu=mu_s, lam=lam_s, epsJ=epsJ)

    # Mask the body force to solid region
    chi_f = chi_solid(x_f, y_f, t_f, center=solid_center, radius=solid_radius)
    fx_f = chi_f * fx_f
    fy_f = chi_f * fy_f

    Lf = loss_fluid(u_f, v_f, p_f, x_f, y_f, t_f, fx=fx_f, fy=fy_f, rho_f=rho_f, nu=nu)

    # ---------
    # Solid PDE on solid points
    # ---------
    dx_s, dy_s = solid_net(x_s, y_s, t_s)
    Ls = loss_solid(dx_s, dy_s, x_s, y_s, t_s, rho_s=rho_s, mu=mu_s, lam=lam_s, epsJ=epsJ)

    # ---------
    # Velocity coupling inside solid domain:
    # evaluate fluid at solid points, compare to solid velocity
    # ---------
    u_fs, v_fs, _ = fluid_net(x_s, y_s, t_s)
    u_s, v_s = solid_velocity(dx_s, dy_s, t_s)
    Lc = loss_velocity_coupling(u_fs, v_fs, u_s, v_s)

    # ---------
    # BC/IC
    # ---------
    Lbc = loss_bc_ic(fluid_net, solid_net)
    Lic = loss_ic_solid(solid_net, N=500)

    return w_f*Lf + w_s*Ls + w_c*Lc + w_bc*Lbc + w_ic*Lic


# =========================================================
# Sampling utilities
# =========================================================
def sample_uniform(N, device):
    x = torch.rand(N, 1, device=device)
    y = torch.rand(N, 1, device=device)
    t = torch.rand(N, 1, device=device)
    return x, y, t

def sample_solid_points(N, device, center=(0.5,0.5), radius=0.15, max_tries=30):
    # rejection sampling inside circle
    logger.debug(f"Sampling {N} solid points with center={center}, radius={radius}")
    xs, ys, ts = [], [], []
    need = N
    for _ in range(max_tries):
        x, y, t = sample_uniform(max(need*3, 200), device)
        chi = chi_solid(x, y, t, center=center, radius=radius)
        mask = (chi[:, 0] > 0.5)
        if mask.any():
            xs.append(x[mask])
            ys.append(y[mask])
            ts.append(t[mask])
        total = sum(z.numel() for z in xs)
        if total >= N:
            break

    if len(xs) == 0:
        # fallback: just return uniform (should not happen unless radius tiny)
        logger.warning("No solid points sampled, using uniform sampling as fallback")
        return sample_uniform(N, device)

    x_s = torch.cat(xs, dim=0)[:N].reshape(N,1)
    y_s = torch.cat(ys, dim=0)[:N].reshape(N,1)
    t_s = torch.cat(ts, dim=0)[:N].reshape(N,1)
    logger.debug(f"Successfully sampled {N} solid points")
    return x_s, y_s, t_s


# =========================================================
# TRAIN LOOP
# =========================================================
def train(
    epochs=5000,
    N_f=2000,
    N_s=1000,
    lr=1e-3,
    rho_f=1.0,
    nu=0.01,
    rho_s=1.0,
    mu_s=1.0,
    lam_s=1.0,
    solid_center=(0.5, 0.5),
    solid_radius=0.15,
    print_every=200,
    w_f=1.0, w_s=1.0, w_c=1.0, w_bc=1.0, w_ic=1.0,
    epsJ=1e-6
):
    logger.info("=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)
    logger.info(f"Training configuration:")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Fluid points (N_f): {N_f}")
    logger.info(f"  Solid points (N_s): {N_s}")
    logger.info(f"  Learning rate: {lr}")
    logger.info(f"  Loss weights - Fluid: {w_f}, Solid: {w_s}, Coupling: {w_c}, BC: {w_bc}, IC: {w_ic}")
    
    fluid_net = FluidPINN().to(device)
    solid_net = SolidPINN().to(device)

    opt = torch.optim.Adam(
        list(fluid_net.parameters()) + list(solid_net.parameters()),
        lr=lr
    )
    logger.info("Networks created and optimizer initialized")

    for epoch in range(1, epochs+1):
        # sample points
        logger.debug(f"Epoch {epoch}/{epochs} - Sampling points")
        x_f, y_f, t_f = sample_uniform(N_f, device)
        x_s, y_s, t_s = sample_solid_points(N_s, device, center=solid_center, radius=solid_radius)

        # enable gradients w.r.t. coordinates for PINN derivatives
        logger.debug(f"Epoch {epoch}/{epochs} - Enabling gradients")
        for z in (x_f, y_f, t_f, x_s, y_s, t_s):
            z.requires_grad_(True)

        opt.zero_grad()
        logger.debug(f"Epoch {epoch}/{epochs} - Computing loss")

        L = total_loss(
            fluid_net, solid_net,
            x_f, y_f, t_f,
            x_s, y_s, t_s,
            rho_f=rho_f, nu=nu,
            rho_s=rho_s, mu_s=mu_s, lam_s=lam_s,
            w_f=w_f, w_s=w_s, w_c=w_c, w_bc=w_bc, w_ic=w_ic,
            solid_center=solid_center, solid_radius=solid_radius,
            epsJ=epsJ
        )

        logger.debug(f"Epoch {epoch}/{epochs} - Backpropagation")
        L.backward()
        logger.debug(f"Epoch {epoch}/{epochs} - Optimizer step")
        opt.step()

        if epoch % print_every == 0:
            logger.info(f"Epoch {epoch:6d}/{epochs} | Loss: {L.item():.6e}")
            logger.debug(f"Epoch {epoch}/{epochs} - Loss components calculated")

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 60)
    return fluid_net, solid_net


if __name__ == "__main__":
    # Example run
    logger.info("Script started")
    fluid_net, solid_net = train(
        epochs=2000,
        N_f=2000,
        N_s=1000,
        lr=1e-3,
        rho_f=1.0,
        nu=0.01,
        rho_s=1.0,
        mu_s=1.0,
        lam_s=1.0,
        solid_center=(0.5, 0.5),
        solid_radius=0.15,
        print_every=200,
        w_f=1.0, w_s=1.0, w_c=1.0, w_bc=1.0, w_ic=1.0,
        epsJ=1e-6
    )
    logger.info("Script completed successfully")
