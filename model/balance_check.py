from collections import Counter
from dataset import get_datasets
from utils import get_transforms

train_dataset, test_dataset = get_datasets(
        train_dir="data/train",
        test_dir="data/test",
        transform=get_transforms(224)  # ResNet-18
    )

print(Counter(train_dataset.targets))
print(Counter(test_dataset.targets))