from torchvision.datasets import ImageFolder


def get_datasets(train_dir, test_dir, transform):
    train_dataset = ImageFolder(root=train_dir, transform=transform)
    test_dataset = ImageFolder(root=test_dir, transform=transform)

    return train_dataset, test_dataset