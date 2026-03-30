import torch

from model.evolution.mutate import mutate
from model.evolution.fitness import evaluate_population


def select_top(population, fitness, k):

    idx = torch.argsort(fitness, descending=True)

    return population[idx[:k]]

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


def evolve(population, decoder, encoder, classifier, device):

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
    new_population.extend(population[best_idx])

    while len(new_population) < len(population):

        parent = tournament_select(population, fitness)

        child = parent.clone()

        child = mutate(child.unsqueeze(0))[0]

        child = project_latent(
            child.unsqueeze(0),
            encoder,
            decoder,
            device
        )[0]

        new_population.append(child)


    new_population = torch.stack(new_population)

    return new_population, fitness, images