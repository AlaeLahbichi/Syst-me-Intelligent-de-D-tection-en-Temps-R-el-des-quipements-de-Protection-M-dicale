"""
Image Data Augmentation Script
================================
Prend un dossier d'images et génère des images augmentées
(rotation, flip, filtre, zoom, bruit, etc.)
Toutes les images (originales + augmentées) sont sauvegardées
dans le même dossier avec le format : alae_img_1, alae_img_2, ...

Usage:
    python image_augmentation.py --input_dir ./images --output_dir ./augmented --prefix alae_img
"""

import os
import sys
import argparse
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from pathlib import Path


# ─────────────────────────────────────────────
#  TRANSFORMATIONS D'AUGMENTATION
# ─────────────────────────────────────────────

def rotate(img, angle=None):
    """Rotation aléatoire entre -45° et 45° (ou angle fixe)."""
    if angle is None:
        angle = random.uniform(-45, 45)
    return img.rotate(angle, expand=True, fillcolor=(0, 0, 0))


def flip_horizontal(img):
    """Miroir horizontal."""
    return ImageOps.mirror(img)


def flip_vertical(img):
    """Miroir vertical."""
    return ImageOps.flip(img)


def zoom_crop(img, zoom_factor=None):
    """Zoom + recadrage au centre."""
    if zoom_factor is None:
        zoom_factor = random.uniform(1.1, 1.5)
    w, h = img.size
    new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - w) // 2
    top = (new_h - h) // 2
    return img_resized.crop((left, top, left + w, top + h))


def adjust_brightness(img, factor=None):
    """Ajuste la luminosité (0.5 = sombre, 1.5 = clair)."""
    if factor is None:
        factor = random.uniform(0.5, 1.5)
    return ImageEnhance.Brightness(img).enhance(factor)


def adjust_contrast(img, factor=None):
    """Ajuste le contraste."""
    if factor is None:
        factor = random.uniform(0.5, 2.0)
    return ImageEnhance.Contrast(img).enhance(factor)


def adjust_saturation(img, factor=None):
    """Ajuste la saturation."""
    if factor is None:
        factor = random.uniform(0.0, 2.0)
    return ImageEnhance.Color(img).enhance(factor)


def adjust_sharpness(img, factor=None):
    """Ajuste la netteté."""
    if factor is None:
        factor = random.uniform(0.0, 3.0)
    return ImageEnhance.Sharpness(img).enhance(factor)


def gaussian_blur(img, radius=None):
    """Flou gaussien."""
    if radius is None:
        radius = random.uniform(0.5, 2.5)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def add_noise(img, intensity=None):
    """Ajout de bruit gaussien."""
    if intensity is None:
        intensity = random.uniform(5, 30)
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, intensity, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def grayscale(img):
    """Conversion en niveaux de gris (maintient 3 canaux RGB)."""
    return ImageOps.grayscale(img).convert("RGB")


def sepia(img):
    """Filtre sépia."""
    arr = np.array(img.convert("RGB"), dtype=np.float64)
    r = arr[:, :, 0] * 0.393 + arr[:, :, 1] * 0.769 + arr[:, :, 2] * 0.189
    g = arr[:, :, 0] * 0.349 + arr[:, :, 1] * 0.686 + arr[:, :, 2] * 0.168
    b = arr[:, :, 0] * 0.272 + arr[:, :, 1] * 0.534 + arr[:, :, 2] * 0.131
    sepia_arr = np.stack([r, g, b], axis=2)
    sepia_arr = np.clip(sepia_arr, 0, 255).astype(np.uint8)
    return Image.fromarray(sepia_arr)


def shear(img, shear_range=None):
    """Déformation (shear) horizontale."""
    if shear_range is None:
        shear_range = random.uniform(-0.3, 0.3)
    w, h = img.size
    transform = (1, shear_range, -shear_range * h / 2,
                 0, 1, 0)
    return img.transform((w, h), Image.AFFINE, transform,
                         resample=Image.BILINEAR)


def edge_enhance(img):
    """Renforcement des contours."""
    return img.filter(ImageFilter.EDGE_ENHANCE_MORE)


def equalize(img):
    """Égalisation de l'histogramme."""
    return ImageOps.equalize(img.convert("RGB"))


def random_crop_pad(img, padding=20):
    """Crop aléatoire + rembourrage."""
    w, h = img.size
    left   = random.randint(0, padding)
    top    = random.randint(0, padding)
    right  = random.randint(0, padding)
    bottom = random.randint(0, padding)
    cropped = img.crop((left, top, w - right, h - bottom))
    return cropped.resize((w, h), Image.LANCZOS)


# ─────────────────────────────────────────────
#  CATALOGUE DES AUGMENTATIONS
# ─────────────────────────────────────────────

AUGMENTATIONS = {
    "rotate_15":         lambda img: rotate(img, 15),
    "rotate_-15":        lambda img: rotate(img, -15),
    "flip_h":            flip_horizontal,
    "flip_v":            flip_vertical,
    "zoom_1.4":          lambda img: zoom_crop(img, 1.4),
    "brightness_dark":   lambda img: adjust_brightness(img, 0.5),
    "brightness_light":  lambda img: adjust_brightness(img, 1.5),
    "contrast_low":      lambda img: adjust_contrast(img, 0.5),
    "contrast_high":     lambda img: adjust_contrast(img, 2.0),
    "saturation_low":    lambda img: adjust_saturation(img, 0.3),
    "gaussian_blur":     gaussian_blur,
    "noise":             add_noise,
    "random_crop":       random_crop_pad,
    "flip_h+rotate_15":  lambda img: rotate(flip_horizontal(img), 15),
    "bright+contrast":   lambda img: adjust_contrast(adjust_brightness(img, 1.3), 1.5),
    "noise+blur":        lambda img: gaussian_blur(add_noise(img), 1.5),
}


# ─────────────────────────────────────────────
#  CHARGEMENT DES IMAGES
# ─────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def load_images_from_folder(folder: str):
    """Retourne une liste de (chemin, objet PIL.Image)."""
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Dossier introuvable : {folder}")

    images = []
    for file in sorted(folder_path.iterdir()):
        if file.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                img = Image.open(file).convert("RGB")
                images.append((file, img))
                print(f"  ✔  Chargé : {file.name}")
            except Exception as e:
                print(f"  ✘  Erreur sur {file.name} : {e}")

    return images


# ─────────────────────────────────────────────
#  PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def augment_dataset(
    input_dir: str,
    output_dir: str,
    prefix: str = "alae_img",
    augmentations: list = None,
    include_originals: bool = True,
):
    """
    Génère les images augmentées et les sauvegarde dans output_dir.

    Paramètres
    ----------
    input_dir        : dossier contenant les images sources
    output_dir       : dossier de sortie (créé si absent)
    prefix           : préfixe des noms de fichiers (ex: alae_img)
    augmentations    : liste de clés dans AUGMENTATIONS (None = toutes)
    include_originals: si True, copie aussi les images originales
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  📂  Chargement des images depuis : {input_dir}")
    print(f"{'='*55}")

    images = load_images_from_folder(input_dir)
    if not images:
        print("Aucune image trouvée. Vérifiez le dossier.")
        sys.exit(1)

    selected_augs = augmentations or list(AUGMENTATIONS.keys())
    counter = 1
    saved_files = []

    print(f"\n{'='*55}")
    print(f"  🔄  Application des augmentations…")
    print(f"{'='*55}")

    for src_path, img in images:
        # ── 1. Copie de l'original ──────────────────────────
        if include_originals:
            out_name = f"{prefix}_{counter}.jpg"
            out_file = output_path / out_name
            img.save(out_file, "JPEG", quality=95)
            saved_files.append(out_name)
            print(f"  [{counter:>4}] Original  ← {src_path.name}")
            counter += 1

        # ── 2. Augmentations ───────────────────────────────
        for aug_key in selected_augs:
            transform = AUGMENTATIONS.get(aug_key)
            if transform is None:
                continue
            try:
                aug_img = transform(img)
                out_name = f"{prefix}_{counter}.jpg"
                out_file = output_path / out_name
                aug_img.save(out_file, "JPEG", quality=95)
                saved_files.append(out_name)
                print(f"  [{counter:>4}] {aug_key:<22} ← {src_path.name}")
                counter += 1
            except Exception as e:
                print(f"  ✘  Erreur [{aug_key}] sur {src_path.name} : {e}")

    # ── Résumé ────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  ✅  {len(saved_files)} images sauvegardées dans : {output_dir}")
    print(f"  📁  Images sources   : {len(images)}")
    print(f"  🔢  Par image        : {len(selected_augs)} augmentations"
          + (" + original" if include_originals else ""))
    print(f"{'='*55}\n")

    return saved_files


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Augmentation de données d'images avec renommage séquentiel."
    )
    parser.add_argument(
        "--input_dir", "-i",
        required=True,
        help="Dossier contenant les images sources."
    )
    parser.add_argument(
        "--output_dir", "-o",
        default="./augmented_images",
        help="Dossier de sortie (défaut : ./augmented_images)."
    )
    parser.add_argument(
        "--prefix", "-p",
        default="alae_img",
        help="Préfixe des noms de fichiers (défaut : alae_img)."
    )
    parser.add_argument(
        "--augmentations", "-a",
        nargs="+",
        default=None,
        help=(
            "Liste des augmentations à appliquer (défaut : toutes). "
            "Valeurs possibles : " + ", ".join(AUGMENTATIONS.keys())
        )
    )
    parser.add_argument(
        "--no_originals",
        action="store_true",
        help="Ne pas inclure les images originales dans la sortie."
    )
    parser.add_argument(
        "--list_augmentations",
        action="store_true",
        help="Affiche la liste des augmentations disponibles et quitte."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_augmentations:
        print("\nAugmentations disponibles :")
        for key in AUGMENTATIONS:
            print(f"  • {key}")
        print()
        sys.exit(0)

    augment_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        prefix=args.prefix,
        augmentations=args.augmentations,
        include_originals=not args.no_originals,
    )


if __name__ == "__main__":
    main()