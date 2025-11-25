import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / len(dataloader), 100 * correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / len(dataloader), 100 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/plate_classification")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device(args.device)

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Dataset
    train_ds = datasets.ImageFolder(os.path.join(args.data_dir, "train"), transform=transform)
    valid_ds = datasets.ImageFolder(os.path.join(args.data_dir, "valid"), transform=transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch, shuffle=False)

    # Model
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)  # binary output
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, valid_loader, criterion, device)

        print(f"[{epoch}/{args.epochs}] "
              f"Train Loss={train_loss:.4f} Acc={train_acc:.2f}% | "
              f"Valid Loss={val_loss:.4f} Acc={val_acc:.2f}%")

    # Save model
    torch.save(model.state_dict(), "runs/tunis_classifier.pt")
    print("✔️ Modèle sauvegardé : runs/tunis_classifier.pt")


if __name__ == "__main__":
    main()
