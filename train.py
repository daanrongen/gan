import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.loader import PaintDropDataset
from models.discriminator import Discriminator
from models.generator import Generator
from models.utils import initialize_weights
import os
import click
import datetime
import sys
import matplotlib.pyplot as plt


@click.command()
@click.option("--dir", default=None, help="Path to training set dir")
@click.option("--out", default=None, help="Path to out dir")
@click.option("--epochs", default=20, help="Number of epochs")
@click.option("--size", default=128, help="Image dimensions")
@click.option("--batch", default=128, help="Batch size")
@click.option("--dim", default=100, help="Noise dimensions")
def main(dir, out, epochs, size, batch, dim):
    print(f"Training dir: {dir} \n"
          f"Out dir: {out} \n"
          f"Epochs: {epochs} \n"
          f"Image size: {size} \n"
          f"Batch size: {batch} \n"
          f"Noise dimensions: {dim}")

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps")

    ROOT_DIR = dir
    OUT_DIR = out
    LEARNING_RATE = 2e-4
    BATCH_SIZE = batch
    IMAGE_SIZE = size
    CHANNELS_IMG = 3
    NOISE_DIM = dim
    NUM_EPOCHS = epochs
    FEATURES_DISC = 64
    FEATURES_GEN = 64

    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
            ),
        ]
    )

    dataset = PaintDropDataset(root_dir=ROOT_DIR, transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

    gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
    disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)

    initialize_weights(gen)
    initialize_weights(disc)

    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)

    dirname = datetime.datetime.strftime(datetime.datetime.now(), "%Y%d%m_%H%M")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(f"{OUT_DIR}/{dirname}", exist_ok=True)

    writer_real = SummaryWriter(f"{OUT_DIR}/{dirname}/logs/real")
    writer_fake = SummaryWriter(f"{OUT_DIR}/{dirname}/logs/fake")

    step = 0

    gen.train()
    disc.train()

    for epoch in range(NUM_EPOCHS):
        for batch_idx, real in enumerate(dataloader):
            real = real.to(device)
            noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
            fake = gen(noise)

            disc_real = disc(real).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake.detach()).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            output = disc(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )

            if batch_idx % BATCH_SIZE == 0:
                with torch.no_grad():
                    fake = gen(fixed_noise)
                    # take out (up to) 32 examples
                    img_grid_real = torchvision.utils.make_grid(real[:64], normalize=True)
                    img_grid_fake = torchvision.utils.make_grid(fake[:64], normalize=True)

                    torchvision.utils.save_image(img_grid_real, f"{OUT_DIR}/{dirname}/real_init.png")
                    torchvision.utils.save_image(img_grid_fake, f"{OUT_DIR}/{dirname}/fake_{epoch:04d}.png")

                    writer_real.add_image("Real", img_grid_real, global_step=step)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=step)

                step += 1


if __name__ == "__main__":
    main()
