import torch


def create_initial_population(encoder, dataloader, device, population_size=64):

    encoder.eval()

    latents = []

    with torch.no_grad():
        for images, _ in dataloader:

            images = images.to(device)
            z = encoder(images)

            latents.append(z.cpu())

            if len(torch.cat(latents)) >= population_size:
                break

    population = torch.cat(latents)[:population_size]

    return population