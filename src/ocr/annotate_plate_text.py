import argparse
import os
import pandas as pd
import cv2


def annotate_csv(csv_path):
    # Charger le CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV introuvable : {csv_path}")

    print(f"Chargement du CSV : {csv_path}")
    df = pd.read_csv(csv_path)

    if "crop_path" not in df.columns:
        raise RuntimeError("La colonne 'crop_path' est introuvable dans le CSV.")
    if "plate_text" not in df.columns:
        print("Colonne 'plate_text' absente, création...")
        df["plate_text"] = ""

    # Convertir en string pour éviter les NaN flottants
    df["plate_text"] = df["plate_text"].astype("string")

    total = len(df)
    print(f"Nombre total de lignes : {total}")

    for idx, row in df.iterrows():
        current_value = row["plate_text"]

        # Si déjà annoté, on saute
        if pd.notna(current_value) and str(current_value).strip() != "":
            continue

        img_path = row["crop_path"]

        # Si le chemin est relatif, on le résout par rapport au CSV
        if not os.path.isabs(img_path):
            csv_dir = os.path.dirname(os.path.abspath(csv_path))
            # On remonte jusqu'à la racine du projet si nécessaire
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(csv_dir)))
            img_path = os.path.join(project_root, img_path)

        if not os.path.exists(img_path):
            print(f"[{idx+1}/{total}] Image introuvable : {img_path}")
            txt = input("Chemin invalide. Tape ENTER pour passer, ou 'q' pour quitter : ").strip()
            if txt.lower() == "q":
                break
            else:
                continue

        # Afficher l'image avec OpenCV
        img = cv2.imread(img_path)
        if img is None:
            print(f"[{idx+1}/{total}] Impossible de lire l'image : {img_path}")
            continue

        window_name = f"Plaque #{idx+1}/{total}"
        cv2.imshow(window_name, img)
        cv2.waitKey(1)  # rafraîchit la fenêtre

        print("\n======================================")
        print(f"Ligne      : {idx+1}/{total}")
        print(f"crop_path  : {row['crop_path']}")
        print("Regarde la fenêtre d'image pour voir la plaque.")
        print("Tape le texte exact de la plaque.")
        print(" - ENTER pour laisser vide et passer")
        print(" - 'q' puis ENTER pour quitter l'annotation")
        print("======================================")

        txt = input("Texte de la plaque : ").strip()

        # Fermer la fenêtre de cette plaque
        cv2.destroyWindow(window_name)

        if txt.lower() == "q":
            print("Arrêt demandé par l'utilisateur.")
            break

        # Enregistrer (même si vide, on note quand même)
        df.at[idx, "plate_text"] = txt

        # Sauvegarde après chaque annotation pour ne rien perdre
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"→ Sauvegardé dans {csv_path}")

    # Fermer toutes les fenêtres au cas où
    cv2.destroyAllWindows()
    print("Annotation terminée.")


def main():
    parser = argparse.ArgumentParser(description="Annoter la colonne plate_text dans un CSV d'OCR.")
    parser.add_argument("--csv", type=str, required=True, help="Chemin vers le fichier labels.csv")
    args = parser.parse_args()

    annotate_csv(args.csv)


if __name__ == "__main__":
    main()
