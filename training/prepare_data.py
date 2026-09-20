"""
STAGE 1 OF TRAINING: Get labeled photos onto disk, organized by category.

Food-101 is a public dataset: ~101,000 real food photos, pre-sorted into
101 folders (one per food type). This script downloads it, then keeps
only the categories we care about (veg/non-veg/seafood relevant to a
fitness-tracking app), copying them into a clean train/val split that
the training script can read directly.

Run once:
    python prepare_data.py
"""

import shutil
from pathlib import Path
from torchvision.datasets import Food101

# Feel free to edit this list. Every name here must exactly match one of
# Food-101's 101 official category folder names.
SELECTED_CLASSES = [
    # Non-veg: meat & poultry
    "chicken_wings", "chicken_curry", "steak", "pork_chop", "prime_rib",
    "beef_tartare", "bacon",
    # Non-veg: seafood & shellfish
    "grilled_salmon", "mussels", "shrimp_and_grits", "lobster_bisque",
    "crab_cakes", "sushi", "ceviche", "oysters",
    # Non-veg: eggs
    "omelette", "scrambled_eggs", "eggs_benedict", "deviled_eggs",
    # Veg: salads & vegetables
    "caesar_salad", "greek_salad", "caprese_salad", "beet_salad",
    "edamame", "guacamole",
    # Veg: grains & carbs
    "fried_rice", "risotto", "pancakes", "waffles", "french_toast",
    "french_fries",
    # Mixed / general
    "hamburger", "pizza", "ramen", "spaghetti_bolognese",
    "spaghetti_carbonara", "club_sandwich", "hot_dog", "tacos", "hummus",
]

DATA_ROOT = Path("./data")
SUBSET_ROOT = Path("./data_subset")


def download_full_dataset():
    """
    Downloads the entire Food-101 archive (~5GB) via torchvision's
    built-in loader. This is a one-time, unattended step - kick it off
    and do something else while it runs.
    """
    print("Downloading Food-101 (~5GB, one-time)...")
    DATA_ROOT.mkdir(exist_ok=True)
    Food101(root=str(DATA_ROOT), split="train", download=True)
    Food101(root=str(DATA_ROOT), split="test", download=True)
    print("Download complete.")


def build_subset():
    """
    Food-101 ships two text files (train.txt / test.txt) listing exactly
    which images belong to which split. We read those, keep only our
    selected classes, and COPY (not move) matching images into a clean
    data_subset/train/<class>/ and data_subset/val/<class>/ structure -
    the exact shape PyTorch's ImageFolder loader expects.
    """
    images_dir = DATA_ROOT / "food-101" / "images"
    meta_dir = DATA_ROOT / "food-101" / "meta"

    if not images_dir.exists():
        raise RuntimeError(f"Expected {images_dir} to exist - did the download run first?")

    train_list = (meta_dir / "train.txt").read_text().splitlines()
    test_list = (meta_dir / "test.txt").read_text().splitlines()

    for split_name, file_list in [("train", train_list), ("val", test_list)]:
        for entry in file_list:
            class_name, image_id = entry.split("/")
            if class_name not in SELECTED_CLASSES:
                continue

            src = images_dir / class_name / f"{image_id}.jpg"
            dst_dir = SUBSET_ROOT / split_name / class_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"{image_id}.jpg"

            if not dst.exists():
                shutil.copy2(src, dst)

    print(f"Subset built at {SUBSET_ROOT} with {len(SELECTED_CLASSES)} classes:")
    print(SELECTED_CLASSES)


if __name__ == "__main__":
    download_full_dataset()
    build_subset()
