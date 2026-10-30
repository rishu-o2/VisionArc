import argparse
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from src.dataset import DRDataset
from src.evaluate import evaluate

REPO_ROOT = Path(__file__).resolve().parent.parent


def compute_class_weights(train_csv_path: Path, num_classes: int = 5, device: torch.device = torch.device("cpu")):
    df = pd.read_csv(train_csv_path)
    counts = df["severity"].value_counts().to_dict()
    total_samples = len(df)
    weights = []
    for c in range(num_classes):
        cnt = counts.get(c, 0)
        w = total_samples / (num_classes * cnt) if cnt > 0 else 0.0
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float, device=device)


def build_model(num_classes: int = 5):
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def set_backbone_trainable(model, trainable: bool = True):
    for param in model.features.parameters():
        param.requires_grad = trainable


def train(epochs_smoke: bool = False, num_workers: int = 0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_csv = REPO_ROOT / "data" / "labels" / "train_dedup.csv"
    val_csv = REPO_ROOT / "data" / "labels" / "val_dedup.csv"

    class_weights = compute_class_weights(train_csv, num_classes=5, device=device)
    print(f"Class weights: {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_dataset = DRDataset(train_csv, split="train", root_dir=REPO_ROOT)
    val_dataset = DRDataset(val_csv, split="eval", root_dir=REPO_ROOT)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = build_model(num_classes=5).to(device)

    models_dir = REPO_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = models_dir / "checkpoint_best.pth"

    reports_dir = REPO_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    log_csv_path = reports_dir / "02_training_log.csv"

    log_records = []
    best_val_macro_f1 = -1.0
    global_epoch = 1

    # PHASE A: 3 epochs, backbone frozen
    phase_a_epochs = 1 if epochs_smoke else 3
    print(f"\n--- Starting Phase A: {phase_a_epochs} epochs (backbone frozen) ---")
    set_backbone_trainable(model, trainable=False)
    optimizer_a = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    for epoch in range(1, phase_a_epochs + 1):
        model.train()
        running_train_loss = 0.0
        total_train_samples = 0

        for b_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            optimizer_a.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_a.step()

            running_train_loss += loss.item() * images.size(0)
            total_train_samples += images.size(0)

            if epochs_smoke:
                break

        train_loss = running_train_loss / total_train_samples if total_train_samples > 0 else 0.0
        val_metrics = evaluate(model, val_loader, device, criterion=criterion)

        print(
            f"Phase A Epoch {epoch}/{phase_a_epochs} (Total {global_epoch}) - "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | Val Macro-F1: {val_metrics['macro_f1']:.4f}"
        )

        log_records.append({
            "epoch": global_epoch,
            "phase": "A",
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "recall_class3": val_metrics["recall_class3"],
            "recall_class4": val_metrics["recall_class4"],
        })

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            torch.save({
                "epoch": global_epoch,
                "model_state_dict": model.state_dict(),
                "val_macro_f1": best_val_macro_f1,
                "val_metrics": val_metrics,
            }, best_checkpoint_path)

        global_epoch += 1
        if epochs_smoke:
            break

    # PHASE B: 10 epochs, all unfrozen
    phase_b_epochs = 1 if epochs_smoke else 10
    print(f"\n--- Starting Phase B: {phase_b_epochs} epochs (all layers unfrozen) ---")
    set_backbone_trainable(model, trainable=True)
    optimizer_b = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler_b = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_b, mode="max", factor=0.5, patience=2
    )

    for epoch in range(1, phase_b_epochs + 1):
        model.train()
        running_train_loss = 0.0
        total_train_samples = 0

        for b_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            optimizer_b.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_b.step()

            running_train_loss += loss.item() * images.size(0)
            total_train_samples += images.size(0)

            if epochs_smoke:
                break

        train_loss = running_train_loss / total_train_samples if total_train_samples > 0 else 0.0
        val_metrics = evaluate(model, val_loader, device, criterion=criterion)
        scheduler_b.step(val_metrics["macro_f1"])

        print(
            f"Phase B Epoch {epoch}/{phase_b_epochs} (Total {global_epoch}) - "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | Val Macro-F1: {val_metrics['macro_f1']:.4f}"
        )

        log_records.append({
            "epoch": global_epoch,
            "phase": "B",
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "recall_class3": val_metrics["recall_class3"],
            "recall_class4": val_metrics["recall_class4"],
        })

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            torch.save({
                "epoch": global_epoch,
                "model_state_dict": model.state_dict(),
                "val_macro_f1": best_val_macro_f1,
                "val_metrics": val_metrics,
            }, best_checkpoint_path)

        global_epoch += 1
        if epochs_smoke:
            break

    pd.DataFrame(log_records).to_csv(log_csv_path, index=False)
    print(f"\nTraining complete. Best Val Macro-F1: {best_val_macro_f1:.4f}")
    print(f"Best checkpoint saved to: {best_checkpoint_path}")
    print(f"Log written to: {log_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline EfficientNet-B0 on VisionArc DR")
    parser.add_argument("--epochs-smoke", action="store_true", help="Run 1 batch per phase for smoke testing")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers")
    args = parser.parse_args()

    train(epochs_smoke=args.epochs_smoke, num_workers=args.num_workers)

