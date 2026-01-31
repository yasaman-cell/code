import torch
import torch.autograd as autograd
from train import load_run,navier_stokes_residual,chi_solid, neo_hookean_PK1, solid_internal_force_PK1, neo_hookean_PK1, solid_internal_force_PK1


def compute_solid_force(dx, dy, x, y, mu=1.0, lam=1.0, epsJ=1e-6):
    P11, P12, P21, P22, J = neo_hookean_PK1(dx, dy, x, y, mu=mu, lam=lam, epsJ=epsJ)
    fx, fy = solid_internal_force_PK1(P11, P12, P21, P22, x, y)
    return fx, fy, J

def validate_pde(fluid_net, solid_net, device, N=5000,
                 rho_f=1.0, nu=0.01,
                 mu_s=1.0, lam_s=1.0,
                 solid_center=(0.5,0.5), solid_radius=0.15,
                 epsJ=1e-6):
    fluid_net.eval()
    solid_net.eval()

    # نقاط کاملاً جدید
    x = torch.rand(N,1,device=device); x.requires_grad_(True)
    y = torch.rand(N,1,device=device); y.requires_grad_(True)
    t = torch.rand(N,1,device=device); t.requires_grad_(True)

    # سیال
    u, v, p = fluid_net(x,y,t)

    # نیروی جامد (مثل train)
    dx, dy = solid_net(x,y,t)
    fx, fy, _J = compute_solid_force(dx, dy, x, y, mu=mu_s, lam=lam_s, epsJ=epsJ)
    chi = chi_solid(x, y, t, center=solid_center, radius=solid_radius)
    fx = chi * fx
    fy = chi * fy

    # residual
    ru, rv, rc = navier_stokes_residual(u, v, p, x, y, t, fx=fx, fy=fy, rho_f=rho_f, nu=nu)

    # یک عدد ساده برای گزارش
    L_val = torch.mean(ru**2 + rv**2 + rc**2).item()
    return L_val

# Use this when Plotting is needed

run_dir = "runs/20260131-171330" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fluid_net, solid_net = load_run(run_dir, device=device)

val_loss = validate_pde(
    fluid_net,
    solid_net,
    device,
    N=5000
)

print(f"[Validation | unseen PDE] L = {val_loss:.3e}")