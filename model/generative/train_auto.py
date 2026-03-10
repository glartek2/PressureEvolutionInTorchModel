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

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    model = Autoencoder().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    criterion = nn.L1Loss()

    epochs = 30

    print("Start Trenning ...")
    
    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for images, _ in train_loader:

            images = images.to(device)

            optimizer.zero_grad()

            recon, _ = model(images)

            loss = criterion(recon, images)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader)}")
        print(images.min(), images.max())

    torch.save(model.state_dict(), "model/weights/autoencoder.pt")


if __name__ == "__main__":
    train()