# SPDX-License-Identifier: Apache-2.0

import torch
import timeit
import matplotlib.pyplot as plt

import slangpy as spy


def main():
    W, H = 1024, 1024

    # Easiest way to create a PyTorch-compatible SlangPy device. Defaults to CUDA when available (best performance).
    # Docs: https://slangpy.shader-slang.org/en/latest/src/autodiff/pytorch.html
    device = spy.create_torch_device(spy.DeviceType.cuda)
    module = spy.Module.load_from_file(device, "soft_rasterizer2d_itensor.slang")

    camera = module.Camera(o=[0.0, 0.0], scale=[1.0, 1.0], frameDim=[float(W), float(H)])
    sigma = 0.02

    # Target
    target_vertices = torch.tensor(
        [[0.7, -0.3], [-0.3, 0.2], [-0.6, -0.6]], dtype=torch.float32, device="cuda"
    )
    target_color = torch.tensor([0.3, 0.8, 0.3], dtype=torch.float32, device="cuda")
    target_img = module.rasterize_pixel(
        camera=camera,
        vertices=target_vertices,
        color=target_color,
        sigma=sigma,
        grid_cell=spy.grid((W, H)),
    )

    # Learnable params
    vertices = torch.tensor(
        [[0.5, -0.5], [-0.5, 0.5], [-0.5, -0.5]],
        dtype=torch.float32,
        device="cuda",
        requires_grad=True,
    )
    color = torch.tensor([0.8, 0.3, 0.3], dtype=torch.float32, device="cuda", requires_grad=True)
    optim = torch.optim.Adam([vertices, color], lr=5e-3)

    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    # Preallocate output once and reuse each iteration
    img = torch.empty((W, H, 3), dtype=torch.float32, device="cuda", requires_grad=True)

    def step(i):
        print(f"Iteration {i}")
        optim.zero_grad()

        nonlocal img
        img = img.detach()

        start = timeit.default_timer()
        module.rasterize_pixel(
            camera=camera,
            vertices=vertices,
            color=color,
            sigma=sigma,
            grid_cell=spy.call_id(),
            _result=img,
        )
        end = timeit.default_timer()
        print(f"Forward pass: {end - start:.6f}s")

        loss = torch.mean((img - target_img) ** 2)
        img.retain_grad()
        loss.backward()
        optim.step()

        if i % 10 == 0:
            ax1.clear()
            ax1.imshow(
                img.permute(1, 0, 2).detach().cpu().numpy(), origin="lower", extent=[-1, 1, -1, 1]
            )
            ax2.clear()
            ax2.imshow(
                img.grad[:, :, 1].T.detach().cpu().numpy(), origin="lower", extent=[-1, 1, -1, 1]
            )
            ax3.clear()
            ax3.imshow(
                target_img.permute(1, 0, 2).detach().cpu().numpy(),
                origin="lower",
                extent=[-1, 1, -1, 1],
            )

    import matplotlib.animation as animation

    ani = animation.FuncAnimation(fig, step, frames=400, interval=10)
    writer = animation.FFMpegWriter(fps=30)
    ani.save("rasterizer2d_itensor.mp4", writer=writer)


if __name__ == "__main__":
    main()
