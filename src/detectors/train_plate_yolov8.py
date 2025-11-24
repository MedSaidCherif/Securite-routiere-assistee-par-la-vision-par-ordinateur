# src/detectors/train_plate_yolov8.py

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 on Tunisian license plate dataset"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s.pt",
        help="Base model or checkpoint (ex: yolov8s.pt, runs/detect/exp/weights/best.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/Tunisian license plate.v2i.yolov8/data.yaml",
        help="Path to YOLO data.yaml",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Training device: 'cuda', 'cuda:0', 'cpu' (None = auto)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/plate_train",
        help="Root folder where runs will be saved",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="plate_yolov8s",
        help="Run name (subfolder in project)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Vérification du chemin data.yaml
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"data.yaml non trouvé à l’emplacement : {data_path.resolve()}"
        )

    # Vérification du modèle de base
    model_path = Path(args.model)
    if not model_path.exists() and not str(args.model).startswith("yolov8"):
        # Si ce n’est pas un nom de modèle officiel (yolov8s.pt, yolov8n.pt, etc.)
        # et que le fichier n’existe pas, on lève une erreur.
        raise FileNotFoundError(
            f"Fichier de poids du modèle non trouvé : {model_path.resolve()}"
        )

    print("=== Configuration d'entraînement YOLOv8 ===")
    print(f"  Model :   {args.model}")
    print(f"  Data :    {data_path}")
    print(f"  Img size: {args.imgsz}")
    print(f"  Epochs :  {args.epochs}")
    print(f"  Batch :   {args.batch}")
    print(f"  Device :  {args.device}")
    print(f"  Project:  {args.project}")
    print(f"  Name   :  {args.name}")
    print("=========================================")

    # Chargement du modèle YOLOv8
    model = YOLO(args.model)

    # Lancement de l’entraînement
    model.train(
        data=str(data_path),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        pretrained=True,   # on part des poids pré-entraînés
    )

    # À la fin, le meilleur modèle sera dans:
    # runs/plate_train/<name>/weights/best.pt
    print("\nEntraînement terminé.")
    print(f"Meilleurs poids dans: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
