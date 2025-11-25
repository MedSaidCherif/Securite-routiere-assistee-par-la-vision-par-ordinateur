import argparse
import os

import numpy as np
from PIL import Image


# ---------- Segmentation des digits (même logique que run_ocr_on_crops) ----------

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


# ---------- Boucle d'annotation ----------

def annotate_plates(source_dir: str, out_dir: str):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    plate_paths = [
        os.path.join(source_dir, f)
        for f in os.listdir(source_dir)
        if os.path.splitext(f)[1].lower() in exts
    ]
    plate_paths.sort()

    if not plate_paths:
        print(f"[ERREUR] Aucune image trouvée dans {source_dir}")
        return

    print(f"Nombre de plaques à traiter : {len(plate_paths)}")
    print("Les digits annotés seront sauvegardés dans :", out_dir)
    print("Commandes :")
    print("  - 0..9 : label du digit")
    print("  - s    : skip ce digit")
    print("  - q    : quitter le script\n")

    os.makedirs(out_dir, exist_ok=True)

    for i, plate_path in enumerate(plate_paths, start=1):
        try:
            plate_img = Image.open(plate_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Impossible d'ouvrir {plate_path} : {e}")
            continue

        # Normaliser la hauteur comme dans run_ocr_on_crops
        target_height = 64
        w, h = plate_img.size
        new_w = int(w * (target_height / h))
        plate_img_resized = plate_img.resize((new_w, target_height), Image.BILINEAR)

        digit_imgs = segment_digits_from_plate(plate_img_resized)

        print(f"\n=== Plaque {i}/{len(plate_paths)} : {plate_path} ===")
        print(f"  -> {len(digit_imgs)} digits détectés")

        if len(digit_imgs) == 0:
            continue

        base_name = os.path.splitext(os.path.basename(plate_path))[0]

        for j, dimg in enumerate(digit_imgs):
            # on agrandit un peu pour l'affichage
            preview = dimg.resize((96, 96), Image.NEAREST)
            preview.show(title=f"{base_name} - digit {j}")

            while True:
                lab = input("Label pour ce digit [0-9], 's' pour skip, 'q' pour quitter : ").strip()

                if lab.lower() == "q":
                    print("Arrêt demandé par l'utilisateur.")
                    return

                if lab.lower() == "s" or lab == "":
                    print("  -> digit ignoré.")
                    break

                if lab in list("0123456789"):
                    digit_dir = os.path.join(out_dir, lab)
                    os.makedirs(digit_dir, exist_ok=True)

                    save_name = f"{base_name}_d{j:02d}.png"
                    save_path = os.path.join(digit_dir, save_name)
                    dimg.save(save_path)
                    print(f"  -> sauvegardé dans {save_path}")
                    break

                print("Entrée invalide. Tape un chiffre 0..9, 's' ou 'q'.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Dossier contenant les plaques croppées (ex: runs/plate_infer/plate_infer_test)"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Racine du dataset de digits (ex: data/digit_from_plates/train)"
    )

    args = parser.parse_args()

    if not os.path.isdir(args.source):
        print(f"[ERREUR] Dossier source introuvable : {args.source}")
        return

    annotate_plates(args.source, args.out_dir)


if __name__ == "__main__":
    main()
