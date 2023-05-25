import argparse
import os
import random

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from models.discriminator import Discriminator
from models.generator import Generator
from models.utils import weights_init

import datetime
import json
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, help="path to training set")
parser.add_argument("--outdir", required=True, help="path to out dir")
parser.add_argument("--batch", type=int, default=64, help="input batch size")
parser.add_argument("--size", type=int, default=64, help="the height / width of the input image to network")
parser.add_argument("--epochs", type=int, default=50, help="number of epochs to train for")
parser.add_argument("--noise_dim", type=int, default=100, help="size of the latent z vector")
parser.add_argument("--features", type=int, default=64, help="features of the disc and gen")
parser.add_argument("--lr", type=float, default=0.0002, help="learning rate, default=0.0002")
parser.add_argument("--beta", type=float, default=0.5, help="beta for adam. default=0.5")
parser.add_argument("--seed", type=int, help="manual seed")
parser.add_argument("--gen", default="", help="path to Generator (to continue training)")
parser.add_argument("--disc", default="", help="path to Discriminator (to continue training)")
parser.add_argument("--cuda", action="store_true", default=False, help="enables cuda")
parser.add_argument("--mps", action="store_true", default=False, help="enables macOS GPU training")
opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outdir, exist_ok=True)
    rundir = datetime.datetime.strftime(datetime.datetime.now(), "%Y%M%d_%H%M")
    os.makedirs(f"{opt.outdir}/{rundir}", exist_ok=True)
except OSError:
    pass

if opt.seed is None:
    opt.seed = random.randint(1, 10000)
print(f"Random seed: {opt.seed}")
random.seed(opt.seed)
torch.manual_seed(opt.seed)
cudnn.benchmark = True

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
    dataset = dset.ImageFolder(
        root=opt.data,
        transform=transforms.Compose([
            transforms.Resize(opt.size),
            transforms.CenterCrop(opt.size),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)
            ),
        ])
    )
    img_channels = 3

    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch, shuffle=True)

    noise_dim = int(opt.noise_dim)
    features = int(opt.features)

    js = json.dumps({
        "noise_dim": noise_dim,
        "img_channels": img_channels,
        "features": features,
    }, sort_keys=True, indent=4, separators=(',', ': '))
    with open(f"{opt.outdir}/{rundir}/parameters.json", "w") as file:
        file.write(js)

    gen = Generator(noise_dim=noise_dim, img_channels=img_channels, features=features).to(device)
    gen.apply(weights_init)
    if opt.gen != "":
        gen.load_state_dict(torch.load(opt.gen))
    print(gen)

    disc = Discriminator(img_channels=img_channels, features=features).to(device)
    disc.apply(weights_init)
    if opt.disc != "":
        disc.load_state_dict(torch.load(opt.disc))
    print(disc)

    criterion = nn.BCELoss()
    fixed_noise = torch.randn(opt.batch, noise_dim, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    # setup optimizer
    gen_optimiser = optim.Adam(gen.parameters(), lr=opt.lr, betas=(opt.beta, 0.999))
    disc_optimiser = optim.Adam(disc.parameters(), lr=opt.lr, betas=(opt.beta, 0.999))

    for epoch in range(opt.epochs):
        for i, data in enumerate(dataloader, 0):
            # update Discriminator
            # train with real
            disc.zero_grad()
            real_cpu = data[0].to(device)

            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, dtype=real_cpu.dtype, device=device)
            output = disc(real_cpu)

            disc_error_real = criterion(output, label)
            disc_error_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, noise_dim, 1, 1, device=device)
            fake = gen(noise)
            label.fill_(fake_label)
            output = disc(fake.detach())
            disc_error_fake = criterion(output, label)
            disc_error_fake.backward()
            D_G_z1 = output.mean().item()
            disc_error = disc_error_real + disc_error_fake
            disc_optimiser.step()

            # update Generator
            gen.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = disc(fake)
            gen_error = criterion(output, label)
            gen_error.backward()
            D_G_z2 = output.mean().item()
            gen_optimiser.step()

            print(f"[{epoch}/{opt.epochs}][{i}/{len(dataloader)}] D loss: {disc_error.item():.4f} | G loss:"
                  f" {gen_error.item():.4f} | D(x): {D_x:.4f} | D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}")

            if i % 100 == 0:
                vutils.save_image(real_cpu, f"{opt.outdir}/{rundir}/real_samples_init.png", normalize=True)
                fake = gen(fixed_noise)
                vutils.save_image(fake.detach(), f"{opt.outdir}/{rundir}/fake_samples_epoch_{epoch:04d}.png",
                                  normalize=True)

        torch.save(gen.state_dict(), f"{opt.outdir}/{rundir}/gen_epoch_{epoch}.pth")
        torch.save(disc.state_dict(), f"{opt.outdir}/{rundir}/disc_epoch_{epoch}.pth")


if __name__ == "__main__":
    main()
