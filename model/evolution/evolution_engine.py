import torch

from model.evolution.mutate import mutate_towards, crossover
from model.evolution.fitness import evaluate_population


def tournament_select(population, fitness, k=3):
    idx = torch.randint(0, len(population), (k,))
    best = idx[fitness[idx].argmax()]
    return population[best]


def project_latent(population, encoder, decoder, device):

    with torch.no_grad():

        population = population.to(device)
        images = decoder(population)
        z_projected = encoder(images)

    return z_projected


def evolve(population, decoder, encoder, classifier, device, generation):

    population = population.to(device)

    fitness, images = evaluate_population(
        decoder,
        classifier,
        population,
        device
    )

    new_population = []


    elite_count = max(1, len(population) // 10)
    best_idx = torch.argsort(fitness, descending=True)[:elite_count]
    elites = population[best_idx]

    new_population.extend(elites)


    top_k = population[torch.argsort(fitness, descending=True)[:5]]

    sigma = max(0.05, 0.4 * (0.9 ** generation))


    while len(new_population) < len(population):

        parent1 = tournament_select(population, fitness)
        parent2 = tournament_select(population, fitness)

        child = crossover(parent1, parent2)

        guide = top_k[torch.randint(0, len(top_k), (1,))][0]

        child = mutate_towards(
            child.unsqueeze(0),
            guide,
            alpha=0.25,
            sigma=sigma
        )[0]

        child = project_latent(
            child.unsqueeze(0),
            encoder,
            decoder,
            device
        )[0]

        new_population.append(child)

    new_population = torch.stack(new_population)

    return new_population, fitness, images
