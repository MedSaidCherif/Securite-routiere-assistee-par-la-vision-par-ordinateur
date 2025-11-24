import argparse
import json
from pathlib import Path
import csv

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
except ImportError:  # fallback si tqdm n'est pas installé
    tqdm = lambda x, *args, **kwargs: x


# ========================
#  Charset / encodage texte
# ========================

class CharsetMapper:
    """
    Gère la correspondance caractères <-> indices pour CTC.
    index 0 = blank CTC
    """
    def __init__(self, chars):
        # chars = liste ordonnée de tous les caractères (hors blank)
        self.chars = list(chars)
        self.blank_idx = 0

        self.id2char = {self.blank_idx: "<BLANK>"}
        self.char2id = {}

        for i, ch in enumerate(self.chars, start=1):
            self.id2char[i] = ch
            self.char2id[ch] = i

    @property
    def num_classes(self):
        # +1 pour le blank
        return len(self.chars) + 1

    def encode(self, text: str):
        """Encode une string en liste d'indices (sans blank)."""
        ids = []
        for ch in text:
            if ch not in self.char2id:
                # on ignore les caractères inconnus
                continue
            ids.append(self.char2id[ch])
        return ids

    def decode(self, ids):
        """Decode une séquence d'indices (avec blank/repetitions) en texte."""
        res = []
        prev = None
        for i in ids:
            if i == self.blank_idx:
                prev = i
                continue
            if i == prev:
                continue
            res.append(self.id2char.get(i, ""))
            prev = i
        return "".join(res)

    def encode_batch(self, texts, device):
        """
        Encode une liste de textes en:
        - targets: tensor 1D concaténé
        - lengths: longueurs de chaque séquence
        """
        encoded = [self.encode(t) for t in texts]
        lengths = torch.tensor([len(seq) for seq in encoded], dtype=torch.long, device=device)
        flat = [i for seq in encoded for i in seq]
        if len(flat) == 0:
            flat = [self.blank_idx]
            lengths = torch.tensor([1] * len(texts), dtype=torch.long, device=device)
        targets = torch.tensor(flat, dtype=torch.long, device=device)
        return targets, lengths

    def save(self, path: Path):
        data = {
            "chars": self.chars,
            "blank_idx": self.blank_idx,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: Path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return CharsetMapper(data["chars"])


# ========================
#  Dataset OCR
# ========================

class PlateOCRDataset(Dataset):
    def __init__(self, csv_path: Path, img_height=32, img_width=160):
        self.samples = []
        self.img_height = img_height
        self.img_width = img_width

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = (row.get("plate_text") or "").strip()
                if text == "":
                    continue  # on garde uniquement les lignes annotées

                img_path = Path(row["crop_path"])
                # Si chemin relatif, on le fait par rapport au projet
                if not img_path.is_absolute():
                    img_path = Path.cwd() / img_path

                if img_path.exists():
                    self.samples.append((img_path, text))

        if len(self.samples) == 0:
            raise RuntimeError(f"Aucun sample valide trouvé dans {csv_path} (plate_text vide ?)")

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: Path):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Impossible de lire l'image: {path}")
        # resize direct (on déforme un peu l'aspect, c'est acceptable)
        img = cv2.resize(img, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # (1, H, W)
        return img

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        img = self._load_image(img_path)
        img = torch.from_numpy(img)
        return img, text


def ocr_collate_fn(batch):
    imgs, texts = zip(*batch)  # listes
    imgs = torch.stack(imgs, dim=0)
    return imgs, list(texts)


# ========================
#  Modèle CRNN + CTC
# ========================

class CRNNCTC(nn.Module):
    def __init__(self, img_height, num_channels, num_classes):
        super().__init__()

        # CNN très simple
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # H/2, W/2

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # H/4, W/4

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        # On force la hauteur à 1 par pooling adaptatif
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))

        # RNN bidirectionnel
        self.rnn = nn.LSTM(
            input_size=256,  # canaux après CNN
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

        self.fc = nn.Linear(256 * 2, num_classes)

    def forward(self, images):
        """
        images: (B, C, H, W)
        retourne:
        - logits: (T, B, C) où T = longueur temporelle
        - input_lengths: tensor (B,) des longueurs T pour chaque sample
        """
        x = self.cnn(images)              # (B, 256, H', W')
        x = self.adaptive_pool(x)         # (B, 256, 1, W'')
        x = x.squeeze(2)                  # (B, 256, W'')
        x = x.permute(0, 2, 1)            # (B, W'', 256)

        x, _ = self.rnn(x)                # (B, W'', 512)
        x = self.fc(x)                    # (B, W'', num_classes)

        x = x.permute(1, 0, 2)            # (W'', B, num_classes) pour CTC

        T = x.size(0)
        B = x.size(1)
        input_lengths = torch.full(
            size=(B,),
            fill_value=T,
            dtype=torch.long,
            device=x.device,
        )

        return x, input_lengths


# ========================
#  Utils de training
# ========================

def build_charset_from_csv(csv_paths):
    chars = set()
    for csv_path in csv_paths:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = (row.get("plate_text") or "").strip()
                if text == "":
                    continue
                for ch in text:
                    chars.add(ch)
    # tri pour reproductibilité
    chars = sorted(list(chars))
    return CharsetMapper(chars)


def train_one_epoch(
    model, criterion, optimizer, dataloader, charset, device,
):
    model.train()
    epoch_loss = 0.0
    n_batches = 0

    for images, texts in tqdm(dataloader, desc="Train", leave=False):
        images = images.to(device)

        targets, target_lengths = charset.encode_batch(texts, device)
        logits, input_lengths = model(images)

        log_probs = logits.log_softmax(2)

        loss = criterion(log_probs, targets, input_lengths, target_lengths)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    return epoch_loss / max(n_batches, 1)


def evaluate(
    model, criterion, dataloader, charset, device,
):
    model.eval()
    epoch_loss = 0.0
    n_batches = 0
    n_total = 0
    n_correct = 0

    with torch.no_grad():
        for images, texts in tqdm(dataloader, desc="Valid", leave=False):
            images = images.to(device)
            targets, target_lengths = charset.encode_batch(texts, device)
            logits, input_lengths = model(images)
            log_probs = logits.log_softmax(2)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            epoch_loss += loss.item()
            n_batches += 1

            # Décodage greedy pour une métrique simple
            # logits: (T, B, C)
            preds = logits.argmax(2).permute(1, 0)  # (B, T)
            for pred_seq, gt in zip(preds, texts):
                pred_ids = pred_seq.cpu().numpy().tolist()
                pred_text = charset.decode(pred_ids)
                if pred_text == gt:
                    n_correct += 1
                n_total += 1

    avg_loss = epoch_loss / max(n_batches, 1)
    acc = n_correct / max(n_total, 1)
    return avg_loss, acc


# ========================
#  Main script
# ========================

def parse_args():
    parser = argparse.ArgumentParser(description="Train CRNN-CTC OCR on Tunisian plates crops")

    parser.add_argument(
        "--train_csv",
        type=str,
        required=True,
        help="Chemin vers data/ocr_dataset/train/labels.csv",
    )
    parser.add_argument(
        "--valid_csv",
        type=str,
        required=True,
        help="Chemin vers data/ocr_dataset/valid/labels.csv",
    )
    parser.add_argument(
        "--img_height",
        type=int,
        default=32,
        help="Hauteur des images d'entrée",
    )
    parser.add_argument(
        "--img_width",
        type=int,
        default=160,
        help="Largeur des images d'entrée",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="'cpu' ou 'cuda' / '0' ...",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="runs/ocr",
        help="Dossier où sauvegarder le modèle et le charset",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    train_csv = Path(args.train_csv)
    valid_csv = Path(args.valid_csv)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=== OCR Training config ===")
    print(f"Train CSV : {train_csv}")
    print(f"Valid CSV : {valid_csv}")
    print(f"Image size: {args.img_height}x{args.img_width}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs    : {args.epochs}")
    print(f"Device    : {args.device}")
    print("============================")

    device = torch.device(args.device)

    # Charset basé sur tous les textes annotés (train + valid)
    charset = build_charset_from_csv([train_csv, valid_csv])
    print(f"Nombre de caractères (hors blank): {len(charset.chars)}")
    print("Charset :", "".join(charset.chars))

    charset_path = save_dir / "ocr_charset.json"
    charset.save(charset_path)
    print(f"Charset sauvegardé dans {charset_path}")

    # Datasets / loaders
    train_dataset = PlateOCRDataset(train_csv, img_height=args.img_height, img_width=args.img_width)
    valid_dataset = PlateOCRDataset(valid_csv, img_height=args.img_height, img_width=args.img_width)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=ocr_collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=ocr_collate_fn,
    )

    # Modèle
    model = CRNNCTC(
        img_height=args.img_height,
        num_channels=1,
        num_classes=charset.num_classes,
    ).to(device)

    criterion = nn.CTCLoss(blank=charset.blank_idx, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_model_path = save_dir / "ocr_best.pth"

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_one_epoch(
            model, criterion, optimizer, train_loader, charset, device
        )
        val_loss, val_acc = evaluate(
            model, criterion, valid_loader, charset, device
        )

        print(f"Train loss: {train_loss:.4f}")
        print(f"Valid loss: {val_loss:.4f} | Exact match acc: {val_acc*100:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "charset_path": str(charset_path),
                    "img_height": args.img_height,
                    "img_width": args.img_width,
                },
                best_model_path,
            )
            print(f"--> Nouveau meilleur modèle sauvegardé dans {best_model_path}")


if __name__ == "__main__":
    main()
