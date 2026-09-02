"""
STAGE 2 OF TRAINING: Teach a pretrained model to recognize YOUR food
categories.

Core idea, in one sentence: show the model a photo, let it guess, compare
the guess to the truth (which we know from the folder it came from),
nudge the model's internal numbers slightly toward being more correct,
repeat thousands of times.

Run:
    python train.py
"""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from tqdm import tqdm

DATA_DIR = Path("./data_subset")
MODEL_OUT = Path("../models/food_classifier.pt")
CLASSES_OUT = Path("../models/class_names.json")

BATCH_SIZE = 32          # how many images the model looks at per learning step
EPOCHS = 25              # max full passes through the training data
LEARNING_RATE = 1e-3     # how big a step to take when adjusting the new layer
FINE_TUNE_LR = 1e-5      # much smaller step size for the pretrained layers we unlock later
UNFREEZE_AFTER_EPOCH = 5
EARLY_STOP_PATIENCE = 5  # stop if val accuracy hasn't improved in this many epochs


def get_device():
    """Use the Mac GPU if available, otherwise fall back to CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_dataloaders():
    # These exact numbers are required because the pretrained model was
    # originally trained with images preprocessed this specific way -
    # matching it lets us reuse its learned knowledge correctly.
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # Data augmentation: randomly flip/rotate/recolor training images so
    # the model learns to recognize food from many angles/lighting
    # conditions, not just the specific photos in the dataset. This is
    # what makes it generalize to real, messy phone photos later.
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.1),
    ])

    # No augmentation for validation - we want to measure real accuracy,
    # not accuracy-on-artificially-modified-images.
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, val_loader, train_dataset.classes


def build_model(num_classes: int):
    # Downloads the pretrained weights automatically the first time.
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Freeze everything - "don't touch what you already know about
    # recognizing shapes/edges/textures".
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace the final layer: originally outputs "which of 1000 ImageNet
    # things is this", we replace it with "which of OUR food categories
    # is this". This new layer starts randomly and is the only part that
    # learns anything at first.
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


def train_one_epoch(model, loader, optimizer, criterion, device):
    """One full pass through the training data, WITH learning happening."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()               # clear old gradients
        outputs = model(images)             # the model's current guesses
        loss = criterion(outputs, labels)   # how wrong were they? (lower = better)
        loss.backward()                     # figure out which direction to adjust weights
        optimizer.step()                    # actually apply the adjustment

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    """
    Same idea, but NO learning happens here (no loss.backward(), no
    optimizer.step()). This measures honest accuracy on images the model
    has never trained on - this is your real, defensible accuracy number.
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, class_names = build_dataloaders()
    print(f"Classes ({len(class_names)}): {class_names}")

    model = build_model(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()

    # Phase 1: only train the new final layer
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val_acc = 0.0
    epochs_without_improvement = 0
    unfrozen = False

    for epoch in range(1, EPOCHS + 1):
        # Phase 2 (after epoch 5): unlock the last 2 pretrained feature
        # blocks too, so the model can adapt deeper, food-specific visual
        # patterns instead of relying purely on generic ImageNet features.
        # A much lower learning rate here protects the valuable pretrained
        # weights from being destroyed by large updates.
        if epoch == UNFREEZE_AFTER_EPOCH and not unfrozen:
            print(f"\n--- Epoch {epoch}: unfreezing last 2 feature blocks ---\n")
            for param in model.features[-2:].parameters():
                param.requires_grad = True

            optimizer = torch.optim.Adam([
                {"params": model.classifier.parameters(), "lr": LEARNING_RATE},
                {"params": model.features[-2:].parameters(), "lr": FINE_TUNE_LR},
            ])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
            unfrozen = True

        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        elapsed = time.time() - start

        print(
            f"Epoch {epoch}/{EPOCHS} ({elapsed:.0f}s) - "
            f"train_acc={train_acc:.4f} - val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"  -> New best model saved (val_acc={val_acc:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"\nNo improvement for {EARLY_STOP_PATIENCE} epochs - stopping early.")
                break

    CLASSES_OUT.write_text(json.dumps(class_names, indent=2))
    print(f"\nDone. Best val accuracy: {best_val_acc:.4f}")
    print("Write this number down (with the class list) for your resume/README - it's your real, honest accuracy claim.")


if __name__ == "__main__":
    main()
