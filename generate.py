import argparse
import json
import re

import torch
import torchvision.utils as vutils

from models.generator import Generator


def num_range(s: str) -> list[int]:
    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    vals = s.split(',')
    return [int(x) for x in vals]


parser = argparse.ArgumentParser()
parser.add_argument("--parameters", required=True, help="path to parameters json")
parser.add_argument("--gen", default="", help="path to Generator")
parser.add_argument("--seeds", type=num_range, help="list of seeds")
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
    with open(opt.parameters) as file:
        p = json.load(file)
        noise_dim = p["noise_dim"]
        img_channels = p["img_channels"]
        features = p["features"]

    gen = Generator(noise_dim=noise_dim, img_channels=img_channels, features=features).to(device)
    gen.load_state_dict(torch.load(opt.gen))
    gen.eval()

    for seed_idx, seed in enumerate(opt.seeds):
        print(f"Generating image for seed {seed} ({seed_idx} / {len(opt.seeds)})")
        torch.manual_seed(seed)
        z = torch.randn(1, noise_dim, 1, 1, device=device)
        fake = gen(z)
        vutils.save_image(fake.detach(), f"{opt.outdir}/{seed:06d}.png", normalize=True)

    if isinstance(opt.seeds, list):
        pass


if __name__ == "__main__":
    main()
