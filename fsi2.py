
# =========================================================
# Fully Physics-Based PINN for Fluid-Structure Interaction
# Revised Version (Extended)
# Base: afrah/pinn_fsi_ibm
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
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))

    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = torch.tanh(self.layers[i](x))
        return self.layers[-1](x)

# -------------------------
# Fluid PINN
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
# Solid PINN
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
    u_s = autograd.grad(dx, t, torch.ones_like(dx), create_graph=True)[0]
    v_s = autograd.grad(dy, t, torch.ones_like(dy), create_graph=True)[0]
    return u_s, v_s

def deformation_gradient(dx, dy, x, y):
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
# Linear elastic stress
# -------------------------
def solid_stress_linear(dx, dy, x, y, mu=1.0, lam=1.0):
    dx_x = autograd.grad(dx, x, torch.ones_like(dx), create_graph=True)[0]
    dx_y = autograd.grad(dx, y, torch.ones_like(dx), create_graph=True)[0]
    dy_x = autograd.grad(dy, x, torch.ones_like(dy), create_graph=True)[0]
    dy_y = autograd.grad(dy, y, torch.ones_like(dy), create_graph=True)[0]

    eps_xx = dx_x
    eps_yy = dy_y
    eps_xy = 0.5*(dx_y + dy_x)

    s_xx = lam*(eps_xx+eps_yy) + 2*mu*eps_xx
    s_yy = lam*(eps_xx+eps_yy) + 2*mu*eps_yy
    s_xy = 2*mu*eps_xy

    return s_xx, s_yy, s_xy

# -------------------------
# Solid internal force
# -------------------------
def solid_internal_force(s_xx, s_yy, s_xy, x, y):
    fx = autograd.grad(s_xx, x, torch.ones_like(s_xx), create_graph=True)[0] +          autograd.grad(s_xy, y, torch.ones_like(s_xy), create_graph=True)[0]

    fy = autograd.grad(s_xy, x, torch.ones_like(s_xy), create_graph=True)[0] +          autograd.grad(s_yy, y, torch.ones_like(s_yy), create_graph=True)[0]
    return fx, fy

# -------------------------
# Navier-Stokes residuals
# -------------------------
def navier_stokes_residual(u, v, p, x, y, t, nu=0.01):
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

    res_u = u_t + u*u_x + v*u_y + p_x - nu*(u_xx + u_yy)
    res_v = v_t + u*v_x + v*v_y + p_y - nu*(v_xx + v_yy)
    res_c = u_x + v_y

    return res_u, res_v, res_c

# -------------------------
# Losses
# -------------------------
def loss_fluid(u, v, p, x, y, t):
    ru, rv, rc = navier_stokes_residual(u, v, p, x, y, t)
    return torch.mean(ru**2) + torch.mean(rv**2) + torch.mean(rc**2)

def loss_solid(dx, dy, x, y, t, rho_s=1.0):
    u_s, v_s = solid_velocity(dx, dy, t)
    u_tt = autograd.grad(u_s, t, torch.ones_like(u_s), create_graph=True)[0]
    v_tt = autograd.grad(v_s, t, torch.ones_like(v_s), create_graph=True)[0]

    s_xx, s_yy, s_xy = solid_stress_linear(dx, dy, x, y)
    fx, fy = solid_internal_force(s_xx, s_yy, s_xy, x, y)

    return torch.mean((fx - rho_s*u_tt)**2 + (fy - rho_s*v_tt)**2)

def loss_velocity_coupling(u_f, v_f, u_s, v_s):
    return torch.mean((u_f - u_s)**2 + (v_f - v_s)**2)

def total_loss(fluid_net, solid_net, x, y, t):
    u_f, v_f, p_f = fluid_net(x, y, t)
    dx, dy = solid_net(x, y, t)

    u_s, v_s = solid_velocity(dx, dy, t)

    Lf = loss_fluid(u_f, v_f, p_f, x, y, t)
    Ls = loss_solid(dx, dy, x, y, t)
    Lc = loss_velocity_coupling(u_f, v_f, u_s, v_s)

    return Lf + Ls + Lc
