import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import logging
from matplotlib.animation import FuncAnimation


from train import load_run

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


@torch.no_grad()
def animate_pinn_fields(
    run_dir: str,
    fluid_net,
    solid_net,
    device,
    field: str = "speed",          # "u","v","p","speed","dx","dy"
    N: int = 150,
    t_min: float = 0.0,
    t_max: float = 1.0,
    n_frames: int = 120,
    fps: int = 30,
    save: bool = True,
    save_format: str = "mp4",      # "mp4" or "gif"
):
    """
    Creates an animation of the chosen field over time using FuncAnimation.

    field options:
      - "u", "v", "p", "speed", "dx", "dy"
    """

    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    out_path = os.path.join(run_dir, "plots", f"anim_{field}.{save_format}")

    fluid_net.eval()
    solid_net.eval()

    # ---- grid (fixed) ----
    x = torch.linspace(0, 1, N, device=device)
    y = torch.linspace(0, 1, N, device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    xv = X.reshape(-1, 1)
    yv = Y.reshape(-1, 1)

    # times
    t_list = np.linspace(t_min, t_max, n_frames)

    # ---- figure ----
    fig, ax = plt.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    title = ax.set_title(f"{field}(x,y,t={t_list[0]:.3f})")

    # We will create an image object once and update its data every frame
    im = None
    cb = None

    def compute_field(t0: float):
        tv = torch.full_like(xv, float(t0))

        u, v, p = fluid_net(xv, yv, tv)
        dx, dy = solid_net(xv, yv, tv)

        U  = u.reshape(N, N).detach().cpu().numpy()
        V  = v.reshape(N, N).detach().cpu().numpy()
        P  = p.reshape(N, N).detach().cpu().numpy()
        DX = dx.reshape(N, N).detach().cpu().numpy()
        DY = dy.reshape(N, N).detach().cpu().numpy()

        if field == "u":
            Z = U
        elif field == "v":
            Z = V
        elif field == "p":
            Z = P
        elif field == "dx":
            Z = DX
        elif field == "dy":
            Z = DY
        elif field == "speed":
            Z = np.sqrt(U**2 + V**2)
        else:
            raise ValueError("field must be one of: u, v, p, speed, dx, dy")

        return Z

    # init: draw first frame
    Z0 = compute_field(t_list[0])

    # extent makes axes correspond to x,y in [0,1]
    im = ax.imshow(
        Z0,
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="auto",
        animated=True
    )
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(field)

    # Optional: fix color scale to avoid flicker:
    # vmin, vmax = np.percentile(Z0, 1), np.percentile(Z0, 99)
    # im.set_clim(vmin, vmax)

    def update(frame_idx: int):
        t0 = t_list[frame_idx]
        Z = compute_field(t0)
        im.set_data(Z)
        title.set_text(f"{field}(x,y,t={t0:.3f})")
        return (im, title)

    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=1000 / fps,
        blit=False  # blit=False is more robust with colorbars
    )

    if save:
        if save_format.lower() == "mp4":
            try:
                anim.save(out_path, writer="ffmpeg", fps=fps, dpi=150)
                print(f"Saved MP4 to: {out_path}")
            except Exception as e:
                print("MP4 save failed (ffmpeg not available?). Error:", e)
                print("Try save_format='gif' instead, or install ffmpeg.")
        elif save_format.lower() == "gif":
            anim.save(out_path, writer="pillow", fps=fps, dpi=150)
            print(f"Saved GIF to: {out_path}")
        else:
            raise ValueError("save_format must be 'mp4' or 'gif'")

    plt.show()
    return anim


if __name__ == "__main__":
    run_dir = "runs/20260131-171330"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fluid_net, solid_net = load_run(run_dir, device=device)

    animate_pinn_fields(
        run_dir=run_dir,
        fluid_net=fluid_net,
        solid_net=solid_net,
        device=device,
        field="speed",   # try: "u","v","p","dx","dy"
        N=150,
        t_min=0.0,
        t_max=1.0,
        n_frames=120,
        fps=30,
        save=True,
        save_format="mp4"  # or "gif"
    )


