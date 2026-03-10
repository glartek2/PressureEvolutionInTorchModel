import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model.dataset import get_datasets
from model.utils import get_transforms
from model.architecture import get_model

from sklearn.metrics import confusion_matrix
import numpy as np


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_layer4(model):
    for param in model.layer4.parameters():
        param.requires_grad = True


def evaluate(model, loader, device):

    model.eval()
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    from sklearn.metrics import classification_report


    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
    
            outputs = model(images)
            _, preds = torch.max(outputs, 1)


            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    
    print(classification_report(all_labels, all_preds))
    print(confusion_matrix(all_labels, all_preds))

    acc = correct / total
    print(f"Validation Accuracy: {acc:.4f}") # TODO: in later/final versions save it to file instead



    return acc



def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, test_dataset = get_datasets(
    train_dir="data/train",
    test_dir="data/test",
    transform=None
    )

    train_dataset.transform = get_transforms(train=True)
    test_dataset.transform = get_transforms(train=False)   

    # TODO: Make num_workers work on positive numbers (fork-issue)
    #
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    #

    model = get_model().to(device)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer = optim.SGD(model.named_parameters(), lr=0.01, momentum=0.9),
    T_max=10
    )

    # PHASE 1 - training FC  ---------------------------------------------------------------
    print("Phase 1: Training only FC layer")

    freeze(model)

    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 3
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        print(f"[Phase1] Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")
        evaluate(model, test_loader, device)

        


    # PHASE 2 - Unfreezing some other layers --------------------------------------------
    print("Phase 2: Unfreezing layer4")

    unfreeze_layer4(model)

    optimizer = optim.Adam([
        {"params": model.layer4.parameters(), "lr": 1e-5},
        {"params": model.fc.parameters(), "lr": 1e-4}
    ])

    for epoch in range(15):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Phase2] Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")
        evaluate(model, test_loader, device)

    torch.save(model.state_dict(), "model/classifier/classifier.pt")
    print("Model saved!")


if __name__ == "__main__":
    train()