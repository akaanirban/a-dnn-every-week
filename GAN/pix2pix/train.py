import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
from dataset_utils import MapDataSet
from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(disc, gen, train_loader, opt_disc, opt_gen, l1_loss, bce_loss,  disc_scaler, gen_scaler):
    loop =tqdm(train_loader, leave=True)
    for idx, (x, y) in enumerate(loop):
        x, y, = x.to(config.DEVICE), y.to(config.DEVICE)

        # train the discriminator
        # with torch.cuda.amp.autocast():
        y_fake = gen(x)
        D_real = disc(x, y)
        D_fake = disc(x, y_fake.detach())
        D_real_loss = bce_loss(D_real, torch.ones_like(D_real))
        D_fake_loss = bce_loss(D_fake, torch.zeros_like(D_fake))
        D_loss = (D_real_loss + D_fake_loss)/2

        # disc.zero_grad()
        # disc_scaler.scale(D_loss).backward()
        # disc_scaler.step(opt_disc)
        # disc_scaler.update()
        disc.zero_grad()
        D_loss.backward()
        opt_disc.step()
        # train the generator

        # train the discriminator
        # with torch.cuda.amp.autocast():
        D_fake = disc(x, y_fake)
        G_fake_loss = bce_loss(D_fake, torch.ones_like(D_fake))
        L1 = l1_loss(y_fake, y)*config.L1_LAMBDA
        G_loss = G_fake_loss + L1

        # gen.zero_grad()
        # gen_scaler.scale(G_loss).backward()
        # gen_scaler.step(opt_gen)
        # gen_scaler.update()
        gen.zero_grad()
        G_loss.backward()
        opt_gen.step()






def main():
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)

    train_dataset = MapDataSet("datasets/maps/train")
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    # for float 16 training optional
    gen_scaler = None#torch.cuda.amp.GradScaler()
    disc_scaler = None#torch.cuda.amp.GradScaler()
    val_dataset = MapDataSet("datasets/maps/val")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in range(config.NUM_EPOCHS):
        train(disc, gen, train_loader, opt_disc, opt_gen, l1_loss, bce_loss,  disc_scaler, gen_scaler)
        if config.SAVE_MODEL and epoch%10==0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

        save_some_examples(gen, val_loader, epoch, folder="savedevaluations")


if __name__ == "__main__":
    main()













