import argparse
import os
import csv

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from ultralytics import YOLO

# ⚠️ IMPORTANT : on importe depuis le fichier dans le même dossier
# (puisque tu exécutes : python src/ocr/run_full_ocr.py)
from train_digit_classifier import DigitCNN  # même classe que pour l'entraînement


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
    Charge le modèle DigitCNN en tenant compte du format du checkpoint :
    - soit un dict "brut" de state_dict
    - soit un dict avec "model_state_dict" + éventuellement "classes"
    """
    model = DigitCNN(num_classes=10)
    state = torch.load(weights_path, map_location=device)

    # Cas 1 : checkpoint complet avec 'model_state_dict'
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
        classes = state.get("classes", None)
        if classes is not None:
            print(f"Classes chargées depuis le checkpoint : {classes}")
    # Cas 2 : on a sauvegardé directement le state_dict
    else:
        model.load_state_dict(state)

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
    if h == 0:
        return "", []
    new_w = int(w * (target_height / h))
    if new_w <= 0:
        new_w = 1

    plate_img = plate_img.resize((new_w, target_height), Image.BILINEAR)

    # Segmentation digits
    digit_imgs = segment_digits_from_plate(plate_img)

    if len(digit_imgs) == 0:
        return "", []

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
        # Sauvegarde debug si demandé
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
    return plate_digits, confidences


# ---------- Pipeline complet sur un dossier d'images ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights_plate",
        type=str,
        default="runs/plate_train/plate_yolov8s/weights/best.pt",
        help="Poids YOLO de détection de plaques."
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
        help="Dossier d'images d'entrée (par ex. data/.../valid/images)"
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
        default="runs/ocr_predictions.csv",
        help="Fichier CSV de sortie pour les prédictions."
    )
    parser.add_argument(
        "--debug_digits_dir",
        type=str,
        default=None,
        help="Dossier où sauvegarder les crops de digits (optionnel)."
    )
    parser.add_argument(
        "--conf_thres",
        type=float,
        default=0.3,
        help="Seuil de confiance YOLO pour les plaques."
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    # Vérifications
    if not os.path.isfile(args.weights_plate):
        print(f"[ERREUR] Poids plaques introuvables : {args.weights_plate}")
        return
    if not os.path.isfile(args.weights_digits):
        print(f"[ERREUR] Poids digits introuvables : {args.weights_digits}")
        return
    if not os.path.isdir(args.source):
        print(f"[ERREUR] Dossier source introuvable : {args.source}")
        return

    # Charger les modèles
    print("Chargement YOLO (plaques)...")
    plate_model = YOLO(args.weights_plate)

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

    print(f"Nombre d'images à traiter : {len(image_paths)}")

    out_dir = os.path.dirname(args.out_csv)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_path",
            "plate_index",
            "x1", "y1", "x2", "y2",
            "digits_pred",
            "digits_conf_mean",
            "digits_conf_min"
        ])

        for img_idx, img_path in enumerate(image_paths, start=1):
            print(f"\n=== Image {img_idx}/{len(image_paths)} : {img_path} ===")

            # YOLO inference
            results = plate_model.predict(
                img_path,
                device=str(device),
                imgsz=640,
                conf=args.conf_thres,
                verbose=False
            )

            if len(results) == 0:
                print("  → Aucun résultat YOLO.")
                continue

            res = results[0]
            boxes = res.boxes

            if boxes is None or len(boxes) == 0:
                print("  → Aucune plaque détectée.")
                continue

            # Charger l'image en PIL
            full_img = Image.open(img_path).convert("RGB")
            w, h = full_img.size

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                cls_id = int(box.cls[0].item())
                conf_box = float(box.conf[0].item())

                # Crop plaque
                x1c = max(int(x1), 0)
                y1c = max(int(y1), 0)
                x2c = min(int(x2), w)
                y2c = min(int(y2), h)

                plate_crop = full_img.crop((x1c, y1c, x2c, y2c))

                # Lecture digits
                image_id = f"{os.path.basename(img_path).split('.')[0]}_p{i}"
                digits_pred, conf_list = read_digits_from_plate(
                    plate_crop,
                    digit_model,
                    device,
                    debug_digits_dir=args.debug_digits_dir,
                    image_id=image_id
                )

                if digits_pred == "":
                    print(f"  Plaque #{i} (conf={conf_box:.2f}) → ÉCHEC lecture digits")
                    mean_conf = 0.0
                    min_conf = 0.0
                else:
                    mean_conf = float(np.mean(conf_list))
                    min_conf = float(np.min(conf_list))
                    print(
                        f"  Plaque #{i} (conf={conf_box:.2f}) → digits='{digits_pred}' "
                        f"(mean_conf={mean_conf:.2f}, min_conf={min_conf:.2f})"
                    )

                writer.writerow([
                    img_path,
                    i,
                    x1c, y1c, x2c, y2c,
                    digits_pred,
                    mean_conf,
                    min_conf
                ])

    print(f"\n✅ OCR terminé. Résultats sauvegardés dans : {args.out_csv}")


if __name__ == "__main__":
    main()
