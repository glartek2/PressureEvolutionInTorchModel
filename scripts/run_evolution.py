import torch
import os
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from model.generative.autoencoder import Autoencoder
from model.architecture import get_model

from model.dataset import get_datasets
from model.generative.latent_utils import get_transforms

from model.evolution.population import create_initial_population
from model.evolution.evolution_engine import evolve


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


POPULATION_SIZE = 200
GENERATIONS = 20


OUTPUT_DIR = "outputs"


def save_generation(images, fitness, generation):

    gen_dir = os.path.join(OUTPUT_DIR, f"generation_{generation}")
    os.makedirs(gen_dir, exist_ok=True)

    idx = torch.argsort(fitness, descending=True)

    for i in range(8):

        img = images[idx[i]]

        save_image(
            img,
            os.path.join(gen_dir, f"best_{i}.png")
        )


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ----------------------
    # Load models
    # ----------------------

    autoencoder = Autoencoder().to(DEVICE)
    autoencoder.load_state_dict(torch.load("model/weights/autoencoder.pt"))

    classifier = get_model().to(DEVICE)
    classifier.load_state_dict(torch.load("model/weights/classifier.pt"))
    classifier.eval()

    encoder = autoencoder.encoder
    decoder = autoencoder.decoder

    # ----------------------
    # Load dataset
    # ----------------------

    transform = get_transforms(train=False)

    train_dataset, _ = get_datasets(
        "data/train",
        "data/test",
        transform=None
    )

    train_dataset.transform = transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    # ----------------------
    # Initial population
    # ----------------------

    population = create_initial_population(
        encoder,
        train_loader,
        DEVICE,
        POPULATION_SIZE
    )

    print("Initial population:", population.shape)

    # ----------------------
    # Evolution loop
    # ----------------------

    for generation in range(GENERATIONS):

        print(f"\nGeneration {generation}")

        population, fitness, images = evolve(
            population,
            decoder,
            encoder,
            classifier,
            DEVICE
        )

        print("Fitness mean:", fitness.mean().item())
        print("Best fitness:", fitness.max().item())

        save_generation(images, fitness, generation)


if __name__ == "__main__":
    main()