import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import logging


from train import load_run

logger = logging.getLogger(__name__)

def make_plots(
    run_dir: str,
    fluid_net,
    solid_net,
    device,
    t_list=(0.25, 0.5, 0.75),
    N=150,
):
    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    logger.info(f"Saving plots to: {plot_dir}")

    # ---- loss curve (optional) ----
    loss_path = os.path.join(run_dir, "loss.npy")
    if os.path.exists(loss_path):
        loss_history = np.load(loss_path)
        plt.figure()
        plt.semilogy(loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Total Loss")
        plt.title("Training Loss")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "loss.png"), dpi=300)
        plt.close()
    else:
        logger.warning(f"No loss.npy found in {run_dir}; skipping loss plot.")

    # ---- field snapshots ----
    fluid_net.eval()
    solid_net.eval()

    x = torch.linspace(0, 1, N, device=device)
    y = torch.linspace(0, 1, N, device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    xv = X.reshape(-1, 1)
    yv = Y.reshape(-1, 1)

    Xnp = X.detach().cpu().numpy()
    Ynp = Y.detach().cpu().numpy()

    for t0 in t_list:
        tv = (float(t0) * torch.ones_like(xv)).to(device)

        with torch.no_grad():
            u, v, p = fluid_net(xv, yv, tv)
            dx, dy = solid_net(xv, yv, tv)

        U = u.reshape(N, N).detach().cpu().numpy()
        V = v.reshape(N, N).detach().cpu().numpy()
        P = p.reshape(N, N).detach().cpu().numpy()
        DX = dx.reshape(N, N).detach().cpu().numpy()
        DY = dy.reshape(N, N).detach().cpu().numpy()
        SPEED = np.sqrt(U**2 + V**2)

        def save_contour(Z, name, title):
            plt.figure()
            plt.contourf(Xnp, Ynp, Z, 50)
            plt.colorbar()
            plt.title(title)
            plt.xlabel("x"); plt.ylabel("y")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, name), dpi=300)
            plt.close()

        save_contour(U,  f"u_t{t0}.png",     f"Fluid velocity u(x,y,t={t0})")
        save_contour(V,  f"v_t{t0}.png",     f"Fluid velocity v(x,y,t={t0})")
        save_contour(SPEED, f"speed_t{t0}.png", f"Speed magnitude |u|(x,y,t={t0})")
        save_contour(P,  f"p_t{t0}.png",     f"Pressure p(x,y,t={t0})")
        save_contour(DX, f"dx_t{t0}.png",    f"Solid displacement dx(x,y,t={t0})")
        save_contour(DY, f"dy_t{t0}.png",    f"Solid displacement dy(x,y,t={t0})")

        # optional quiver
        step = max(N // 25, 1)
        plt.figure()
        plt.quiver(Xnp[::step, ::step], Ynp[::step, ::step],
                   U[::step, ::step],   V[::step, ::step])
        plt.title(f"Velocity vectors (u,v) at t={t0}")
        plt.xlabel("x"); plt.ylabel("y")        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"quiver_t{t0}.png"), dpi=300)
        plt.close()

    logger.info(f"Plots saved in: {plot_dir}")
    return plot_dir


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Use this when Plotting is needed
if __name__ == "__main__":
    run_dir = "runs/20260131-171330" 
    fluid_net, solid_net = load_run(run_dir, device=device)
    make_plots(run_dir, fluid_net, solid_net, device=device)
    logger.info(f"Plotted from existing run: {run_dir}")
