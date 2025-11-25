# src/ocr/eval_digit_classifier.py
import argparse
import os
import random
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
import numpy as np

# ⚠️ adapte ce nom à TON modèle dans train_digit_classifier.py
from train_digit_classifier import DigitCNN
  # ou le nom de ta classe


def build_valid_loader(data_dir, batch_size=64):
    valid_dir = os.path.join(data_dir, "valid")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    valid_ds = datasets.ImageFolder(valid_dir, transform=transform)
    valid_loader = torch.utils.data.DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False
    )

    return valid_loader, valid_ds.classes


def load_model(weights_path, device, num_classes=10):
    model = DigitCNN(num_classes=num_classes)
    state = torch.load(weights_path, map_location=device)

    # si tu as sauvegardé directement state_dict
    if isinstance(state, dict) and not isinstance(state, nn.Module):
        # parfois on a un champ "state_dict" ou "model_state_dict"
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
    else:
        # sauvegarde directe du modèle
        model = state

    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/digit_dataset")
    parser.add_argument("--weights", type=str, default="runs/digit_classifier.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device : {device}")

    # 1) Data
    valid_loader, class_names = build_valid_loader(
        args.data_dir, batch_size=args.batch_size
    )
    print("Classes :", class_names)

    # 2) Modèle
    model = load_model(args.weights, device, num_classes=len(class_names))

    # 3) Évaluation
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in valid_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    acc = (all_preds == all_labels).mean() * 100.0
    print(f"\n✅ Accuracy globale sur valid : {acc:.2f}%")

    # 4) Matrice de confusion
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))
    print("\nMatrice de confusion (lignes = vrai, colonnes = prédiction) :")
    print("    " + "  ".join(class_names))
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:3d}" for v in row)
        print(f"{class_names[i]} | {row_str}")

    # 5) Quelques exemples
    print("\nQuelques exemples (vrai -> prédiction) :")
    idxs = list(range(len(all_labels)))
    random.shuffle(idxs)
    for i in idxs[:20]:
        true = class_names[all_labels[i]]
        pred = class_names[all_preds[i]]
        ok = "✅" if true == pred else "❌"
        print(f"{ok} {true} -> {pred}")


if __name__ == "__main__":
    main()
