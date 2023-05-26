import re

import torch
import torchvision.utils as vutils


def seed_to_vec(seed: int, noise_dim: int, device: torch.device) -> torch.Tensor:
    torch.manual_seed(seed)
    z = torch.randn(1, noise_dim, 1, 1, device=device)
    return z


def line_interpolate(zs: list[torch.Tensor], steps: int) -> list[torch.Tensor]:
    out = []
    for i in range(len(zs) - 1):
        for index in range(steps):
            t = index / float(steps)
            out.append(zs[i + 1] * t + zs[i] * (1 - t))
    return out


def images(gen: torch.nn.Module, zs: list[torch.Tensor], outdir: str):
    for z_idx, z in enumerate(zs):
        print(f"Generating image for frame {z_idx}/{len(zs)}")
        img = gen(z)
        vutils.save_image(img.detach(), f"{outdir}/frame_{z_idx:04d}.png", normalize=True)


def interpolate(gen: torch.nn.Module, zs: list[torch.Tensor], frames: int, outdir: str):
    points = line_interpolate(zs, frames)
    images(gen=gen, zs=points, outdir=outdir)


def num_range(s: str) -> list[int]:
    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    vals = s.split(',')
    return [int(x) for x in vals]
