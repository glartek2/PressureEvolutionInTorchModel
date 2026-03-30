import torch
import matplotlib.pyplot as plt

from model.generative.autoencoder import Autoencoder
from model.dataset import get_datasets
from model.generative.latent_utils import get_transforms


device = "cuda"


model = Autoencoder().to(device)
model.load_state_dict(torch.load("model/weights/autoencoder_150.pt"))
model.eval()

train_dataset, _ = get_datasets("data/train", "data/test", transform=None)
train_dataset.transform = get_transforms(train=False)

image, _ = train_dataset[5]

with torch.no_grad():

    x = image.unsqueeze(0).to(device)

    recon, z = model(x)

    z_mut = z + torch.randn_like(z) * 0.05
    
    for i in range(1000):
        z_mut = z_mut + torch.randn_like(z_mut) * 0.05
    
    
    mutated = model.decoder(z_mut)

    recon = recon.squeeze().cpu()
    image = image.cpu()


def denorm(x):
    return (x + 1) / 2


plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(denorm(image).permute(1,2,0))

plt.subplot(1,3,2)
plt.title("Reconstruction")
plt.imshow(denorm(recon).permute(1,2,0))

plt.subplot(1,3,3)
plt.title("Mutation")
plt.imshow(denorm(mutated.squeeze().cpu()).permute(1,2,0))

plt.show()