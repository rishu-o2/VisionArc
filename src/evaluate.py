import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


def evaluate(model, loader, device, criterion=None):
    model.eval()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    total_samples = len(loader.dataset)
    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0

    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, labels=[0, 1, 2, 3, 4], zero_division=0
    )
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2, 3, 4])

    recall_class3 = rec[3]
    recall_class4 = rec[4]

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "recall_class3": recall_class3,
        "recall_class4": recall_class4,
    }

