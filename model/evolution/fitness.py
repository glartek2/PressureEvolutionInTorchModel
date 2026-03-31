import torch
import torch.nn.functional as F
import torchvision.transforms as T


def evaluate_population(decoder, classifier, population, device):

    decoder.eval()
    classifier.eval()

    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    with torch.no_grad():

        population = population.to(device)

        images = decoder(population)
        images_vis = images


        color_penalty = images_vis.std(dim=(1, 2, 3))


        flat = images_vis.view(images_vis.size(0), -1)
        flat = F.normalize(flat, dim=1)

        similarity = torch.mm(flat, flat.t())

        mask = ~torch.eye(similarity.size(0), dtype=bool, device=similarity.device)
        similarity_penalty = similarity[mask].view(similarity.size(0), -1).mean(dim=1)


        images_cls = torch.stack([normalize(img) for img in images_vis])

        logits = classifier(images_cls)
        probs = F.softmax(logits, dim=1)

        venomous_prob = probs[:, 1]

        fitness = (
            100 * venomous_prob
            - 40 * similarity_penalty
            - 20 * color_penalty
        )

    return fitness.cpu(), images_vis.cpu()
