# src/detectors/plate_infer_yolov8.py

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with trained YOLOv8 license plate model"
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="runs/plate_train/plate_yolov8s/weights/best.pt",
        help="Path to trained weights (.pt)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/Tunisian license plate.v2i.yolov8/valid/images",
        help="Source for inference: image, video, directory, or glob pattern",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size used for inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: 'cpu', 'cuda', '0', '0,1', etc. (None = auto)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/plate_infer",
        help="Root folder where inference results will be saved",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="plate_infer_test",
        help="Run name (subfolder in project)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Fichier de poids introuvable : {weights_path.resolve()}\n"
            "Assure-toi que l'entraînement est terminé et que le chemin est correct."
        )

    print("=== Configuration d'inférence YOLOv8 ===")
    print(f"  Weights : {weights_path}")
    print(f"  Source  : {args.source}")
    print(f"  Img size: {args.imgsz}")
    print(f"  Device : {args.device}")
    print(f"  Project: {args.project}")
    print(f"  Name   : {args.name}")
    print("========================================")

    # Charger le modèle
    model = YOLO(str(weights_path))

    # Lancer la prédiction
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        save=True,       # sauvegarde les images / vidéos annotées
        save_txt=True,   # sauvegarde les bboxes au format txt
        save_conf=True,  # sauvegarde les scores de confiance
        verbose=True,
    )

    print("\nInférence terminée.")
    print(f"Résultats sauvegardés dans : {args.project}/{args.name}")


if __name__ == "__main__":
    main()
