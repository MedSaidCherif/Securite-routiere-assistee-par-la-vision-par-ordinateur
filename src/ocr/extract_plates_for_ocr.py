# src/ocr/extract_plates_for_ocr.py

import argparse
import csv
import os
from pathlib import Path
from typing import List

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extraire automatiquement les crops de plaques pour l'OCR à partir d'un modèle YOLOv8"
    )

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Chemin vers les poids YOLOv8 entraînés (best.pt)",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Dossier d'images source (par ex: data/.../train/images)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Dossier de sortie pour les crops & le CSV (sera créé si n'existe pas)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Taille d'image pour l'inférence YOLO",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device: 'cpu', 'cuda', '0', etc.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Seuil de confiance minimal pour garder une détection",
    )
    parser.add_argument(
        "--max_per_image",
        type=int,
        default=2,
        help="Nombre max de plaques à garder par image (ordonnées par confiance)",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.05,
        help="Padding relatif autour de la bbox (0.05 = 5%% de marge)",
    )

    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_crop(
    img,
    bbox: List[float],
    padding: float,
    out_path: Path,
):
    """
    img: image numpy BGR
    bbox: [x1, y1, x2, y2] en pixels
    padding: ratio par rapport à largeur/hauteur
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox

    bw = x2 - x1
    bh = y2 - y1
    pad_x = padding * bw
    pad_y = padding * bh

    x1 = int(max(0, x1 - pad_x))
    y1 = int(max(0, y1 - pad_y))
    x2 = int(min(w - 1, x2 + pad_x))
    y2 = int(min(h - 1, y2 + pad_y))

    crop = img[y1:y2, x1:x2]
    cv2.imwrite(str(out_path), crop)


def main():
    args = parse_args()

    weights_path = Path(args.weights)
    source_path = Path(args.source)
    out_root = Path(args.out_dir)

    if not weights_path.exists():
        raise FileNotFoundError(f"Fichier de poids introuvable: {weights_path.resolve()}")
    if not source_path.exists():
        raise FileNotFoundError(f"Dossier source introuvable: {source_path.resolve()}")

    crops_dir = out_root / "images"
    ensure_dir(crops_dir)

    csv_path = out_root / "labels.csv"

    print("=== Extraction des plaques pour OCR ===")
    print(f"Weights : {weights_path}")
    print(f"Source  : {source_path}")
    print(f"Sortie  : {out_root}")
    print(f"Device  : {args.device}")
    print("=======================================")

    model = YOLO(str(weights_path))

    # On va streamer les résultats pour garder le lien image -> détection
    results = model.predict(
        source=str(source_path),
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        stream=True,
        verbose=True,
    )

    rows = []
    crop_idx = 0

    for r in results:
        img_path = Path(r.path)
        img_name = img_path.name
        img = r.orig_img  # BGR numpy

        if r.boxes is None or len(r.boxes) == 0:
            continue

        # On récupère les bboxes avec leur confiance
        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        # On suppose 1 seule classe : car-license-plate (id=0)
        # mais on garde la logique au cas où.
        detections = []
        for bbox, conf, cls_id in zip(boxes_xyxy, confs, classes):
            detections.append((float(conf), bbox, int(cls_id)))

        # Tri par confiance décroissante
        detections.sort(key=lambda x: x[0], reverse=True)
        detections = detections[: args.max_per_image]

        for det_rank, (conf, bbox, cls_id) in enumerate(detections):
            crop_idx += 1
            crop_filename = f"{img_path.stem}_plate{det_rank+1:02d}.jpg"
            crop_path = crops_dir / crop_filename

            save_crop(img, bbox, args.padding, crop_path)

            rows.append(
                {
                    "crop_id": crop_idx,
                    "crop_path": str(crop_path).replace("\\", "/"),
                    "source_image": str(img_path).replace("\\", "/"),
                    "x1": bbox[0],
                    "y1": bbox[1],
                    "x2": bbox[2],
                    "y2": bbox[3],
                    "conf": conf,
                    # A REMPLIR MANUELLEMENT PLUS TARD
                    "plate_text": "",
                }
            )

    # Écriture du CSV
    fieldnames = [
        "crop_id",
        "crop_path",
        "source_image",
        "x1",
        "y1",
        "x2",
        "y2",
        "conf",
        "plate_text",
    ]

    ensure_dir(out_root)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nExtraction terminée.")
    print(f"- Nombre total de crops : {len(rows)}")
    print(f"- Dossier images OCR :   {crops_dir}")
    print(f"- Fichier labels CSV :   {csv_path}")
    print("Tu peux maintenant remplir la colonne 'plate_text' avec la vraie plaque.")


if __name__ == "__main__":
    main()
