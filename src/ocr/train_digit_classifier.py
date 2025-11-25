import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Petit CNN simple pour 10 classes (0-9)
class DigitCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(DigitCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (B,32,32,32)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # (B,32,16,16)

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # (B,64,16,16)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # (B,64,8,8)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # (B,128,4,4)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, 100.0 * correct / total


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    return running_loss / total, 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/digit_dataset",
                        help="Chemin vers digit_dataset (contenant train/ et valid/)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default="runs/digit_classifier.pt")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print("Device :", device)

    # Transforms : on met tout en niveaux de gris et taille fixe 32x32
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # centré [-1,1]
    ])

    train_dir = os.path.join(args.data_dir, "train")
    valid_dir = os.path.join(args.data_dir, "valid")

    train_ds = datasets.ImageFolder(train_dir, transform=transform)
    valid_ds = datasets.ImageFolder(valid_dir, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print("Classes :", train_ds.classes)  # doit donner ['0','1',...,'9']

    model = DigitCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_acc = eval_one_epoch(model, valid_loader, criterion, device)

        print(f"[{epoch}/{args.epochs}] "
              f"Train Loss={train_loss:.4f} Acc={train_acc:.2f}% | "
              f"Valid Loss={valid_loss:.4f} Acc={valid_acc:.2f}%")

        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": train_ds.classes
            }, args.out)
            print(f"✔️ Nouveau meilleur modèle sauvegardé : {args.out} (Acc={best_acc:.2f}%)")

    print("Entraînement terminé.")


if __name__ == "__main__":
    main()
