import torch


def mutate_towards(population, guide, alpha=0.5, sigma=0.2):

    guide = guide.unsqueeze(0)

    direction = guide - population
    noise = torch.randn_like(population) * sigma

    mutated = population + alpha * direction + noise

    return torch.clamp(mutated, -5, 5)


def crossover(parent1, parent2):
    alpha = torch.rand(1).item()
    return alpha * parent1 + (1 - alpha) * parent2
