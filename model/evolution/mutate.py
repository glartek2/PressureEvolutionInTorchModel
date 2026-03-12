import torch


def mutate(population, mutation_prob=0.5, sigma=0.5):

    mask = torch.rand_like(population) < mutation_prob
    noise = torch.randn_like(population) * sigma

    mutated = population + mask * noise

    mutated = torch.clamp(mutated, -5, 5)

    return mutated