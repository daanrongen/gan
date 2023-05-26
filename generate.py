import argparse
import json
import os
import subprocess

import torch
import torchvision.utils as vutils

from models.esrgan import RRDBNet
from models.generator import Generator
from utils import num_range, interpolate, seed_to_vec

parser = argparse.ArgumentParser()
parser.add_argument("--parameters", type=str, required=True, help="path to parameters json")
parser.add_argument("--gen", type=str, required=True, help="path to Generator")
parser.add_argument("--seeds", type=num_range, help="list of seeds")
parser.add_argument("--upscale", action="store_true", default=False, help="uses esrgan to upscale output image")
parser.add_argument("--esrgan", type=str, help="path to ESRGAN")
parser.add_argument("--interpolate", action="store_true", default=False, help="whether to interpolate inbetween seeds")
parser.add_argument("--frames", default=120, type=int, help="how many frames to produce")
parser.add_argument("--fps", default=24, type=int, help="framerate for video")
parser.add_argument("--outdir", required=True, help="path to out dir")
parser.add_argument("--cuda", action="store_true", default=False, help="enables cuda")
parser.add_argument("--mps", action="store_true", default=False, help="enables macOS GPU training")
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if torch.backends.mps.is_available() and not opt.mps:
    print("WARNING: You have mps device, to enable macOS GPU run with --mps")

if opt.cuda:
    device = torch.device("cuda:0")
elif opt.mps and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def main():
    global esrgan
    with open(opt.parameters) as file:
        p = json.load(file)
        size = p["size"]
        noise_dim = p["noise_dim"]
        img_channels = p["img_channels"]
        features = p["features"]

    gen = Generator(noise_dim=noise_dim, img_channels=img_channels, features=features).to(device)
    gen.load_state_dict(torch.load(opt.gen))
    gen.eval()

    if opt.esrgan:
        esrgan = RRDBNet(3, 3, 64, 23, gc=32)
        esrgan.load_state_dict(torch.load(opt.esrgan), strict=True)
        esrgan.eval()
        esrgan.to(device)

    if (opt.interpolate):
        print(f"Calculating frames inbetween seeds for seeds {opt.seeds}")
        os.makedirs(f"{opt.outdir}/frames", exist_ok=True)

        zs = []
        for seed_idx, seed in enumerate(opt.seeds):
            z = seed_to_vec(seed, noise_dim, device)
            zs.append(z)

        interpolate(
            gen=gen,
            zs=zs,
            frames=opt.frames,
            outdir=f"{opt.outdir}/frames",
            upscale=opt.upscale,
            esrgan=esrgan if opt.esrgan else False
        )

        seedstr = "_".join([str(seed) for seed in opt.seeds])
        vidname = f"interpolation_seeds_{seedstr}_{opt.fps}fps"
        cmd = f"ffmpeg -y -r {opt.fps} -i {opt.outdir}/frames/frame_%04d.png" \
              f" -vcodec libx264 -pix_fmt yuv420p " \
              f"{opt.outdir}/{vidname}.mp4"
        subprocess.call(cmd, shell=True)

    for seed_idx, seed in enumerate(opt.seeds):
        print(f"Generating image for seed {seed} ({seed_idx} / {len(opt.seeds)})")
        z = seed_to_vec(seed, noise_dim, device)
        fake = gen(z)

        if opt.upscale:
            fake = esrgan(fake)

        vutils.save_image(fake.detach(), f"{opt.outdir}/{seed:06d}.png", normalize=True)


if __name__ == "__main__":
    main()
