import torch
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve

def train_model(model, train_loader, val_loader, epochs, loss_function, optimizer, device, best_model_path, patience=10, verbose=True):

    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    best_val_loss = float("inf")
    patience_counter = 0

    best_val_metrics = {
        "accuracy": 0,
        "precision": 0,
        "recall": 0,
        "f1": 0
    }

    model.to(device)

    for epoch in range(epochs):
        if verbose:
            print(f"\nEpoch [{epoch+1}/{epochs}]")

        # Training
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_batch = loss.item()
            train_loss += loss_batch * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if verbose:
                loop.set_postfix({
                    "loss": f"{loss_batch:.4f}",
                    "acc": f"{correct / total:.4f}"
                })

        train_loss /= total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = loss_function(outputs, labels)

                val_loss += loss.item() * inputs.size(0)

                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= total
        val_acc = correct / total

        precision = precision_score(all_labels, all_preds, average="binary")
        recall = recall_score(all_labels, all_preds, average="binary")
        f1 = f1_score(all_labels, all_preds, average="binary")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if verbose:
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            best_val_metrics = {
                "accuracy": val_acc,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "best_val_metrics": best_val_metrics
            }, best_model_path)

            patience_counter = 0
            if verbose:
                print(f"Best model saved to: {best_model_path}")
        else:
            patience_counter += 1
            if verbose:
                print(f"Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                if verbose:
                    print("Early stopping triggered.")
                break

    return history, best_val_metrics


def plot_history(history):
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Loss")
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.title("Accuracy")
    plt.legend()

    plt.show()


def log_to_tensorboard(log_dir, hparams, history, best_val_metrics):

    writer = SummaryWriter(log_dir=log_dir)

    num_epochs = len(history["val_loss"])

    # Log per-epoch validation & training metrics
    for epoch in range(num_epochs):
        writer.add_scalar("Loss/val", history["val_loss"][epoch], epoch)
        writer.add_scalar("Accuracy/val", history["val_acc"][epoch], epoch)

        writer.add_scalar("Loss/train", history["train_loss"][epoch], epoch)
        writer.add_scalar("Accuracy/train", history["train_acc"][epoch], epoch)

    # Log hyperparameters + final metrics
    final_metrics = {
        "hparam/accuracy": best_val_metrics["accuracy"],
        "hparam/precision": best_val_metrics["precision"],
        "hparam/recall": best_val_metrics["recall"],
        "hparam/f1": best_val_metrics["f1"],
    }

    writer.add_hparams(hparam_dict=hparams, metric_dict=final_metrics, run_name=log_dir)

    writer.close()


def plot_precision_recall_curve(model, val_loader, device):
    model.eval()
    
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            # Convert logits to probabilities for class 1
            probs = torch.softmax(logits, dim=1)[:, 1]

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Precision-recall curve
    precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

    # Best F1
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]

    # thresholds has len = len(precision) - 1
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    # Plot
    plt.figure(figsize=(3, 3))

    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid()
    plt.show()

    default_preds = (all_probs >= 0.5).astype(int)
    default_f1 = f1_score(all_labels, default_preds)
    print(f"Default F1 (0.5 threshold): {default_f1:.4f}\n")

    print(f"Best F1: {best_f1:.4f}")
    print(f"Best Threshold: {best_threshold:.4f}")

    return best_threshold


def evaluate_model(model, test_loader, device, threshold=0.5):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs >= threshold).long()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print("\nCLASSIFICATION REPORT")
    print(classification_report(all_labels, all_preds, digits=4))

    print("\nOVERALL")
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nCONFUSION MATRIX")
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = confusion_matrix(all_labels, all_preds, normalize="true")

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3))

    # Raw counts
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["Actual 0", "Actual 1"],
        ax=axes[0]
    )
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # Normalized
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["Actual 0", "Actual 1"],
        ax=axes[1]
    )
    axes[1].set_title("Confusion Matrix (Normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    plt.show()