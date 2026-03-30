import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.generative.autoencoder import Autoencoder
from model.dataset import get_datasets
from model.generative.latent_utils import get_transforms


def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, test_dataset = get_datasets(
        "data/train",
        "data/test",
        transform=None
    )

    train_dataset.transform = get_transforms(train=True)
    test_dataset.transform = get_transforms(train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    model = Autoencoder().to(device)

    model = model.to(memory_format=torch.channels_last)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
        weight_decay=1e-4
    )

    scaler = torch.amp.GradScaler("cuda" if torch.cuda.is_available() else "cpu")

    l1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()

    epochs = 1

    print("Start Training...")

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for images, _ in train_loader:

            images = images.to(device, non_blocking=True)
            images = images.to(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):

                recon, _ = model(images)

                l1 = l1_loss(recon, images)
                #mse = mse_loss(recon, images)

                loss = l1

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")
        torch.save(model.state_dict(), f"model/weights/autoencoder_epoch_{epoch+1}.pt")

    torch.save(model.state_dict(), "model/weights/autoencoder.pt")


if __name__ == "__main__":
    train()
