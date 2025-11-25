import argparse
import os
import csv
import sys

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# ---------------------------------------------------------------------
# Pour pouvoir importer DigitCNN depuis train_digit_classifier.py
# ---------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.ocr.train_digit_classifier import DigitCNN  # même classe que pour l'entraînement


# ---------- Segmentation des digits dans une plaque ----------

def binarize_image(gray_img: np.ndarray, thresh: int = 128):
    """
    gray_img : tableau numpy HxW (0..255)
    Retourne une image binaire : True = encre (noir), False = fond.
    """
    return gray_img < thresh


def segment_digits_from_plate(pil_img: Image.Image,
                              min_col_sum: int = 5,
                              min_width: int = 5):
    """
    Segmente automatiquement les digits dans une image de plaque.

    - Binarisation
    - Projection verticale
    - Regroupement de colonnes actives
    """
    gray = pil_img.convert("L")
    arr = np.array(gray)

    # Binarisation
    bin_img = binarize_image(arr, thresh=128)

    # Projection verticale : sommation par colonne
    col_sums = bin_img.sum(axis=0)
    active = col_sums > min_col_sum

    segments = []
    in_char = False
    start = 0

    for x, is_active in enumerate(active):
        if is_active and not in_char:
            in_char = True
            start = x
        elif not is_active and in_char:
            in_char = False
            end = x
            if end - start >= min_width:
                segments.append((start, end))

    if in_char:
        end = len(active)
        if end - start >= min_width:
            segments.append((start, end))

    digit_images = []

    for (x1, x2) in segments:
        sub = bin_img[:, x1:x2]

        row_sums = sub.sum(axis=1)
        rows_active = row_sums > 0
        if not rows_active.any():
            continue
        y_indices = np.where(rows_active)[0]
        y1, y2 = y_indices[0], y_indices[-1] + 1

        sub_crop = arr[y1:y2, x1:x2]
        digit_img = Image.fromarray(sub_crop)
        digit_images.append(digit_img)

    return digit_images


# ---------- Chargement du modèle de digits ----------

def load_digit_model(weights_path: str, device: torch.device):
    """
    Charge DigitCNN à partir de ton checkpoint d'entraînement.
    On gère le cas où le checkpoint contient un dict avec 'model_state_dict'.
    """
    model = DigitCNN(num_classes=10)
    checkpoint = torch.load(weights_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        classes = checkpoint.get("classes", None)
        if classes is not None:
            print("Classes chargées depuis le checkpoint :", classes)
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ---------- Traitement d'une image de plaque (croppée) ----------

def read_digits_from_plate(plate_img: Image.Image, digit_model, device: torch.device,
                           debug_digits_dir: str = None, image_id: str = ""):
    """
    Retourne la chaîne de chiffres prédite pour une image de plaque.
    """
    # Normalisation taille de la plaque (pour stabiliser la segmentation)
    target_height = 64
    w, h = plate_img.size
    new_w = int(w * (target_height / h))
    plate_img = plate_img.resize((new_w, target_height), Image.BILINEAR)

    # Segmentation digits
    digit_imgs = segment_digits_from_plate(plate_img)

    if len(digit_imgs) == 0:
        return "", [], 0

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    predictions = []
    confidences = []

    if debug_digits_dir is not None:
        os.makedirs(debug_digits_dir, exist_ok=True)

    for idx, dimg in enumerate(digit_imgs):
        if debug_digits_dir is not None:
            dimg.save(os.path.join(debug_digits_dir, f"{image_id}_digit_{idx:02d}.png"))

        x = transform(dimg).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = digit_model(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)

        digit = int(pred.item())
        predictions.append(str(digit))
        confidences.append(float(conf.item()))

    plate_digits = "".join(predictions)
    return plate_digits, confidences, len(digit_imgs)


# ---------- Pipeline sur un dossier d'images de plaques CROPPÉES ----------

def main():
    parser = argparse.ArgumentParser(
        description="OCR chiffres sur plaques DÉJÀ CROPPÉES (sans YOLO, seulement DigitCNN)."
    )
    parser.add_argument(
        "--weights_digits",
        type=str,
        default="runs/digit_classifier.pt",
        help="Poids du classifieur de chiffres."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Dossier contenant les plaques recadrées (par ex. plate_infer/plate_infer_test)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu ou cuda"
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="runs/ocr_crops_predictions.csv",
        help="Fichier CSV de sortie pour les prédictions."
    )
    parser.add_argument(
        "--debug_digits_dir",
        type=str,
        default=None,
        help="Dossier où sauvegarder les crops de digits (optionnel)."
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    # Vérifications
    if not os.path.isfile(args.weights_digits):
        print(f"[ERREUR] Poids digits introuvables : {args.weights_digits}")
        return
    if not os.path.isdir(args.source):
        print(f"[ERREUR] Dossier source introuvable : {args.source}")
        return

    print("Device :", device)
    print("Chargement classifieur de chiffres...")
    digit_model = load_digit_model(args.weights_digits, device)

    # Lister les images
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    image_paths = [
        os.path.join(args.source, f)
        for f in os.listdir(args.source)
        if os.path.splitext(f)[1].lower() in exts
    ]
    image_paths.sort()

    print(f"Nombre d'images de plaques à traiter : {len(image_paths)}")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_path",
            "digits_pred",
            "digits_conf_mean",
            "digits_conf_min",
            "num_digits"
        ])

        for img_idx, img_path in enumerate(image_paths, start=1):
            print(f"\n=== Plaque {img_idx}/{len(image_paths)} : {img_path} ===")

            # Charger l'image de plaque complète
            try:
                plate_img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"  [ERREUR] Impossible de lire {img_path} : {e}")
                continue

            image_id = os.path.basename(img_path).split(".")[0]

            digits_pred, conf_list, n_digits = read_digits_from_plate(
                plate_img,
                digit_model,
                device,
                debug_digits_dir=args.debug_digits_dir,
                image_id=image_id
            )

            if digits_pred == "":
                print(f"  → ÉCHEC lecture digits (aucun segment trouvé)")
                mean_conf = 0.0
                min_conf = 0.0
            else:
                mean_conf = float(np.mean(conf_list))
                min_conf = float(np.min(conf_list))
                print(f"  → digits='{digits_pred}' "
                      f"(mean_conf={mean_conf:.2f}, min_conf={min_conf:.2f}, n_digits={n_digits})")

            writer.writerow([
                img_path,
                digits_pred,
                mean_conf,
                min_conf,
                n_digits
            ])

    print(f"\n✅ OCR sur plaques croppées terminé. Résultats sauvegardés dans : {args.out_csv}")


if __name__ == "__main__":
    main()
