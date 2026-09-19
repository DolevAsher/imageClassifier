import time
from pathlib import Path
from collections import defaultdict
import random
import joblib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from sklearn.metrics import precision_recall_fscore_support
import copy

# Assuming these are imported from your local setup
from base_model import ImageNetSubset
from model import ModelArchitecture

# Point to the root directory containing the split subfolders
DATA_ROOT = Path("/content/IML_Hackathon/local_train_set")
OUTPUT = Path("weights.joblib")
PLOT_OUTPUT = Path("loss_plot.png")
METRICS_PLOT_OUTPUT = Path("metrics_plot.png")

# Set hyperparameters
IMAGE_SIZE = 224
BATCH_SIZE = 64
NUM_EPOCHS = 30  # Increased for proper learning
LEARNING_RATE = 0.0005
SEED = 421

# --- SPLITTING & SCALING HYPERPARAMETERS ---
DATASET_PERCENTAGE = 100.0  # Restored to 100% - Do not starve the model!
TRAIN_RATIO = 0.80
VAL_RATIO = 0.20


def main():
    """Full training pipeline with robust architecture, learning rate scheduling, and smart saving."""

    assert abs((TRAIN_RATIO + VAL_RATIO) - 1.0) < 1e-5, "Ratios must sum to 1.0!"

    # 1. Define transformations
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        ], p=0.5),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 2. Load dataset
    print("Loading original dataset...")
    full_dataset = ImageNetSubset(root=DATA_ROOT, split="train", transform=train_transforms)

    # 3. Stratified Train/Val Splitting
    print("Splitting dataset into Train and Val sets...")
    class_samples = defaultdict(list)
    for sample in full_dataset.samples:
        path, label = sample
        class_samples[label].append(sample)

    random.seed(SEED)
    train_samples = []
    val_samples = []

    for label, samples in class_samples.items():
        random.shuffle(samples)

        if DATASET_PERCENTAGE < 100.0:
            num_to_scale = int(len(samples) * (DATASET_PERCENTAGE / 100.0))
            num_to_scale = max(1, num_to_scale)
            samples = samples[:num_to_scale]

        total_count = len(samples)
        train_end = int(total_count * TRAIN_RATIO)

        train_samples.extend(samples[:train_end])
        val_samples.extend(samples[train_end:])

    # 4. Create separate Python Dataset instances
    train_dataset = copy.deepcopy(full_dataset)
    train_dataset.samples = train_samples
    train_dataset.transform = train_transforms

    val_dataset = copy.deepcopy(full_dataset)
    val_dataset.samples = val_samples
    val_dataset.transform = eval_transforms

    print(f"Dataset Split Completed Summary:")
    print(f"  - Train samples: {len(train_dataset)}")
    print(f"  - Val samples:   {len(val_dataset)}")

    # 5. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # 6. Initialize model & Environment
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = ModelArchitecture(num_classes=20).to(device)

    # 7. Define Loss Function, Optimizer, and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # --- TRACKERS ---
    epoch_losses = []
    val_accuracies = []
    val_error_rates = []
    val_precisions = []
    val_recalls = []
    val_f1s = []
    best_val_f1 = 0.0
    best_model_state = None

    # 8. Training Loop
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = (correct / total) * 100
        epoch_losses.append(epoch_loss)

        # --- EVALUATE ON VALIDATION SET ---
        model.eval()
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_targets.extend(labels.cpu().numpy())

        val_acc = (val_correct / val_total) * 100
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_val_targets, all_val_preds, average='macro', zero_division=0
        )

        val_accuracies.append(val_acc)
        val_error_rates.append(100.0 - val_acc)
        val_precisions.append(precision * 100)
        val_recalls.append(recall * 100)
        val_f1s.append(f1 * 100)

        time_elapsed = time.time() - start_time
        print(f"--- Epoch {epoch + 1} Summary ({time_elapsed:.1f}s) ---")
        print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | Val Acc: {val_acc:.2f}% | Val F1: {(f1 * 100):.2f}%")

        # SMART SAVING LOGIC
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            print(f"⭐ New best model found! F1: {(f1*100):.2f}% (Saved to memory)")

        # STEP SCHEDULER
        scheduler.step(f1)

    # 9. Save Best Model Weights
    if best_model_state is not None:
        print(f"\nTraining complete. Saving BEST weights (F1: {(best_val_f1*100):.2f}%) to {OUTPUT}...")
        joblib.dump(best_model_state, OUTPUT)
        print("Saved trained weights.joblib successfully!")
    else:
        print("\nTraining complete, but no valid model state was found to save.")

    # 10. Generate and save performance graphs
    print("Generating performance graphs...")
    epochs_range = range(1, NUM_EPOCHS + 1)

    # Graph 1
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, epoch_losses, marker="o", color="b", linestyle="-", linewidth=2)
    plt.title(f"Training Loss Over Epochs ({DATASET_PERCENTAGE}% Capacity)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(epochs_range)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(PLOT_OUTPUT)

    # Graph 2
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, val_accuracies, label="Accuracy", marker="o", color="green", linewidth=2)
    plt.plot(epochs_range, val_error_rates, label="Error Rate", marker="x", color="red", linewidth=2)
    plt.plot(epochs_range, val_precisions, label="Precision (Macro)", marker="s", color="purple", linestyle="--")
    plt.plot(epochs_range, val_recalls, label="Recall (Macro)", marker="^", color="orange", linestyle="--")
    plt.plot(epochs_range, val_f1s, label="F1 Score (Macro)", marker="d", color="blue", linestyle="-.")

    plt.title(f"Validation Metrics Over Epochs ({DATASET_PERCENTAGE}% Capacity)")
    plt.xlabel("Epoch")
    plt.ylabel("Percentage (%)")
    plt.yticks(range(0, 105, 10))
    plt.xticks(epochs_range)
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(METRICS_PLOT_OUTPUT)

    plt.show()

if __name__ == "__main__":
    main()