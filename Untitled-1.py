# =========================================================
# Fully Physics-Based PINN for Fluid-Structure Interaction
# FINAL (based on your code, with step-by-step required changes)
# - Separate fluid/solid points
# - Velocity coupling inside solid domain
# - Force coupling: solid internal force injected as body force into NS
# - Minimal training loop
# =========================================================

import torch
import torch.nn as nn
import torch.autograd as autograd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# MLP Network
# -------------------------
class MLP(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))

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

    def forward(self, x, y, t):
        out = self.net(torch.cat([x, y, t], dim=1))
        dx = out[:, 0:1]
        dy = out[:, 1:2]
        return dx, dy

# -------------------------
# Solid kinematics
# -------------------------
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

# -------------------------
# Linear elastic stress (small strain)
# sigma = lambda tr(eps) I + 2 mu eps
# -------------------------
def solid_stress_linear(dx, dy, x, y, mu=1.0, lam=1.0):
    dx_x = autograd.grad(dx, x, torch.ones_like(dx), create_graph=True)[0]
    dx_y = autograd.grad(dx, y, torch.ones_like(dx), create_graph=True)[0]
    dy_x = autograd.grad(dy, x, torch.ones_like(dy), create_graph=True)[0]
    dy_y = autograd.grad(dy, y, torch.ones_like(dy), create_graph=True)[0]

    eps_xx = dx_x
    eps_yy = dy_y
    eps_xy = 0.5 * (dx_y + dy_x)

    s_xx = lam * (eps_xx + eps_yy) + 2 * mu * eps_xx
    s_yy = lam * (eps_xx + eps_yy) + 2 * mu * eps_yy
    s_xy = 2 * mu * eps_xy

    return s_xx, s_yy, s_xy

# -------------------------
# Solid internal force: div(sigma)
# fx = d(s_xx)/dx + d(s_xy)/dy
# fy = d(s_xy)/dx + d(s_yy)/dy
# -------------------------
def solid_internal_force(s_xx, s_yy, s_xy, x, y):
    fx = autograd.grad(s_xx, x, torch.ones_like(s_xx), create_graph=True)[0] + \
         autograd.grad(s_xy, y, torch.ones_like(s_xy), create_graph=True)[0]

    fy = autograd.grad(s_xy, x, torch.ones_like(s_xy), create_graph=True)[0] + \
         autograd.grad(s_yy, y, torch.ones_like(s_yy), create_graph=True)[0]
    return fx, fy

# -------------------------
# NEW: Compute solid force (re-usable)
# -------------------------
def compute_solid_force(dx, dy, x, y, mu=1.0, lam=1.0):
    s_xx, s_yy, s_xy = solid_stress_linear(dx, dy, x, y, mu=mu, lam=lam)
    fx, fy = solid_internal_force(s_xx, s_yy, s_xy, x, y)
    return fx, fy

# -------------------------
# Geometry: chi(x,y,t) = 1 inside solid, 0 outside
# Example: static circular solid
# -------------------------
def chi_solid(x, y, t=None, center=(0.5, 0.5), radius=0.15):
    cx, cy = center
    r2 = radius * radius
    inside = ((x - cx)**2 + (y - cy)**2 <= r2).float()
    return inside

# -------------------------
# Navier-Stokes residuals WITH body force
# res_u = rho (u_t + u u_x + v u_y) + p_x - nu Lap(u) - fx
# res_v = rho (v_t + u v_x + v v_y) + p_y - nu Lap(v) - fy
# res_c = u_x + v_y
# -------------------------
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

# -------------------------
# Losses
# -------------------------
def loss_fluid(u, v, p, x, y, t, fx=None, fy=None, rho_f=1.0, nu=0.01):
    ru, rv, rc = navier_stokes_residual(u, v, p, x, y, t, fx=fx, fy=fy, rho_f=rho_f, nu=nu)
    return torch.mean(ru**2) + torch.mean(rv**2) + torch.mean(rc**2)

def loss_solid(dx, dy, x, y, t, rho_s=1.0, mu=1.0, lam=1.0):
    u_s, v_s = solid_velocity(dx, dy, t)
    u_tt = autograd.grad(u_s, t, torch.ones_like(u_s), create_graph=True)[0]
    v_tt = autograd.grad(v_s, t, torch.ones_like(v_s), create_graph=True)[0]

    fx, fy = compute_solid_force(dx, dy, x, y, mu=mu, lam=lam)

    # solid momentum residual: div(sigma) - rho * d_tt = 0
    return torch.mean((fx - rho_s*u_tt)**2 + (fy - rho_s*v_tt)**2)

def loss_velocity_coupling(u_f, v_f, u_s, v_s):
    # u_f = u_s inside solid domain
    return torch.mean((u_f - u_s)**2 + (v_f - v_s)**2)

# -------------------------
# Optional: BC/IC hooks (currently zero)
# Add your real boundary/initial constraints here
# -------------------------
def loss_bc_ic(fluid_net, solid_net):
    # Example placeholders:
    # - no-slip at outer boundary
    # - initial condition at t=0
    # - solid anchored boundary condition, etc.
    return torch.tensor(0.0, device=device)

# -------------------------
# TOTAL LOSS (final)
# - Fluid PDE on fluid points (x_f,y_f,t_f) WITH body force
# - Solid PDE on solid points (x_s,y_s,t_s)
# - Velocity coupling enforced inside solid domain (use fluid evaluated at solid points)
# -------------------------
def total_loss(fluid_net, solid_net,
               x_f, y_f, t_f,
               x_s, y_s, t_s,
               rho_f=1.0, nu=0.01,
               rho_s=1.0, mu_s=1.0, lam_s=1.0,
               w_f=1.0, w_s=1.0, w_c=1.0, w_bc=1.0,
               solid_center=(0.5, 0.5), solid_radius=0.15):

    # ---------
    # Fluid forward on fluid points
    # ---------
    u_f, v_f, p_f = fluid_net(x_f, y_f, t_f)

    # ---------
    # Build body force on fluid points by evaluating solid net there, then masking to solid region
    # (simple alternative to IBM spreading)
    # ---------
    dx_f, dy_f = solid_net(x_f, y_f, t_f)  # evaluate everywhere
    fx_f, fy_f = compute_solid_force(dx_f, dy_f, x_f, y_f, mu=mu_s, lam=lam_s)

    chi_f = chi_solid(x_f, y_f, t_f, center=solid_center, radius=solid_radius)
    fx_f = chi_f * fx_f
    fy_f = chi_f * fy_f

    Lf = loss_fluid(u_f, v_f, p_f, x_f, y_f, t_f, fx=fx_f, fy=fy_f, rho_f=rho_f, nu=nu)

    # ---------
    # Solid forward on solid points
    # ---------
    dx_s, dy_s = solid_net(x_s, y_s, t_s)
    Ls = loss_solid(dx_s, dy_s, x_s, y_s, t_s, rho_s=rho_s, mu=mu_s, lam=lam_s)

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

    return w_f*Lf + w_s*Ls + w_c*Lc + w_bc*Lbc


# =========================================================
# Sampling utilities (minimal, uniform sampling in [0,1]x[0,1]x[0,1])
# Replace with your real domain/time ranges and sampling strategy
# =========================================================
def sample_uniform(N, device):
    x = torch.rand(N, 1, device=device)
    y = torch.rand(N, 1, device=device)
    t = torch.rand(N, 1, device=device)
    return x, y, t

def sample_solid_points(N, device, center=(0.5,0.5), radius=0.15, max_tries=20):
    # rejection sampling inside circle
    xs = []
    ys = []
    ts = []
    need = N
    for _ in range(max_tries):
        x, y, t = sample_uniform(max(need*2, 100), device)
        chi = chi_solid(x, y, t, center=center, radius=radius)
        mask = (chi[:,0] > 0.5)
        x_in = x[mask]
        y_in = y[mask]
        t_in = t[mask]
        if x_in.numel() > 0:
            xs.append(x_in)
            ys.append(y_in)
            ts.append(t_in)
        total = sum(z.numel() for z in xs)
        if total >= N:
            break

    x_s = torch.cat(xs, dim=0)[:N].reshape(N,1)
    y_s = torch.cat(ys, dim=0)[:N].reshape(N,1)
    t_s = torch.cat(ts, dim=0)[:N].reshape(N,1)
    return x_s, y_s, t_s


# =========================================================
# TRAIN LOOP (minimal)
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
    print_every=200
):
    fluid_net = FluidPINN().to(device)
    solid_net = SolidPINN().to(device)

    opt = torch.optim.Adam(
        list(fluid_net.parameters()) + list(solid_net.parameters()),
        lr=lr
    )

    for epoch in range(1, epochs+1):
        # sample points
        x_f, y_f, t_f = sample_uniform(N_f, device)
        x_s, y_s, t_s = sample_solid_points(N_s, device, center=solid_center, radius=solid_radius)

        # enable gradients w.r.t. coordinates for PINN derivatives
        for z in (x_f, y_f, t_f, x_s, y_s, t_s):
            z.requires_grad_(True)

        opt.zero_grad()
        L = total_loss(
            fluid_net, solid_net,
            x_f, y_f, t_f,
            x_s, y_s, t_s,
            rho_f=rho_f, nu=nu,
            rho_s=rho_s, mu_s=mu_s, lam_s=lam_s,
            w_f=1.0, w_s=1.0, w_c=1.0, w_bc=1.0,
            solid_center=solid_center, solid_radius=solid_radius
        )
        L.backward()
        opt.step()

        if epoch % print_every == 0:
            print(f"epoch {epoch:6d} | loss {L.item():.6e}")

    return fluid_net, solid_net


if __name__ == "__main__":
    # Example run
    train(
        epochs=2000,
        N_f=2000,
        N_s=1000,
        lr=1e-3,
        rho_f=1.0,
        nu=0.01,
        rho_s=1.0,
        mu_s=1.0,
        lam_s=1.0,
        solid_center=(0.5,0.5),
        solid_radius=0.15,
        print_every=200
    )