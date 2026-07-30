"""
Combined implementation for:
- Task 4: Option B (Noise Injection) + Option C (Data Filtering) for Group 5
- Task 5: Performance Evaluation and Analysis, including a separate training-loss summary
- Bonus 1: PGD attack comparison
- Bonus 2: Attack-strength (epsilon) sensitivity analysis

Run this file after model.py and data_poisoning_attack.py have generated:
- mnist_baseline_model.pth
- mnist_poisoned_model.pth
"""

from pathlib import Path
import csv
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


# ============================================================
# 1. Reproducibility, folders, and device
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

print("Device used:", DEVICE)


# ============================================================
# 2. Task 3 poisoning configuration (must match Task 3)
# ============================================================

LABEL_FLIP_PERCENTAGE = 60
NOISE_PERCENTAGE = 60
NOISE_STANDARD_DEVIATION = 1.50


# ============================================================
# 3. Task 4 defense configuration: Group 5 = Option B + C
# ============================================================

# Option C: reference-model consistency + image roughness filtering
CONFIDENCE_THRESHOLD = 0.80
ROUGHNESS_THRESHOLD = 0.15

# Option B: controlled Gaussian-noise injection during retraining
NOISE_INJECTION_STD = 0.20
NOISE_INJECTION_PROBABILITY = 0.50

# Attack/evaluation settings. FGSM remains an attacked condition in Task 5;
# it is not used as a Task 4 training defense for Group 5.
ATTACK_EVALUATION_EPSILON = 0.20
BATCH_SIZE = 64
FILTER_ONLY_EPOCHS = 7
DEFENSE_EPOCHS = 10
LEARNING_RATE = 0.001

print("\n========== TASK 4 DEFENSE CONFIGURATION ==========")
print("Group 5: Option B + Option C")
print("Option C: Data Filtering")
print(f"  Reference confidence threshold : {CONFIDENCE_THRESHOLD:.2f}")
print(f"  Image roughness threshold      : {ROUGHNESS_THRESHOLD:.2f}")
print("Option B: Noise Injection")
print(f"  Gaussian-noise std             : {NOISE_INJECTION_STD:.2f}")
print(f"  Probability per training image : {NOISE_INJECTION_PROBABILITY:.0%}")
print(f"  Filtering-only epochs          : {FILTER_ONLY_EPOCHS}")
print(f"  Combined-defense epochs        : {DEFENSE_EPOCHS}")
print(f"  FGSM evaluation epsilon        : {ATTACK_EVALUATION_EPSILON:.2f}")
print("==================================================")


# ============================================================
# 4. Model and dataset definitions
# ============================================================

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class PoisonedMNIST(Dataset):
    """Recreate the exact severe poisoned dataset used in Task 3."""

    def __init__(self, root):
        clean_dataset = datasets.MNIST(
            root=root,
            train=True,
            download=True,
        )

        self.clean_images = clean_dataset.data.float().unsqueeze(1) / 255.0
        self.images = self.clean_images.clone()
        self.clean_labels = clean_dataset.targets.clone()
        self.labels = self.clean_labels.clone()

        number_of_samples = len(self.labels)
        rng = np.random.default_rng(SEED)

        number_of_flipped_labels = int(
            number_of_samples * LABEL_FLIP_PERCENTAGE / 100
        )
        number_of_noisy_images = int(
            number_of_samples * NOISE_PERCENTAGE / 100
        )

        self.label_flip_indices = np.sort(
            rng.choice(
                number_of_samples,
                size=number_of_flipped_labels,
                replace=False,
            )
        )
        self.noise_indices = np.sort(
            rng.choice(
                number_of_samples,
                size=number_of_noisy_images,
                replace=False,
            )
        )

        # Systematic label flipping: 0 -> 1, ..., 9 -> 0.
        self.labels[self.label_flip_indices] = (
            self.labels[self.label_flip_indices] + 1
        ) % 10

        noise_generator = torch.Generator().manual_seed(SEED + 1)
        noise = torch.randn(
            (number_of_noisy_images, 1, 28, 28),
            generator=noise_generator,
        ) * NOISE_STANDARD_DEVIATION

        self.images[self.noise_indices] = torch.clamp(
            self.images[self.noise_indices] + noise,
            min=0.0,
            max=1.0,
        )

        self.poisoned_indices = np.union1d(
            self.label_flip_indices,
            self.noise_indices,
        )

        self.poison_mask = np.zeros(number_of_samples, dtype=bool)
        self.poison_mask[self.poisoned_indices] = True

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


criterion = nn.CrossEntropyLoss()


def load_model_weights(model, path):
    try:
        state_dict = torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state_dict)


# ============================================================
# 5. Evaluation and FGSM functions
# ============================================================


def evaluate_clean(model, data_loader, return_predictions=False):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            if return_predictions:
                all_labels.append(labels.cpu())
                all_predictions.append(predictions.cpu())

    average_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples

    if return_predictions:
        return (
            average_loss,
            accuracy,
            torch.cat(all_labels),
            torch.cat(all_predictions),
        )
    return average_loss, accuracy


def generate_fgsm(model, images, labels, epsilon):
    model.eval()
    attack_images = images.detach().clone().requires_grad_(True)
    outputs = model(attack_images)
    loss = criterion(outputs, labels)
    gradients = torch.autograd.grad(loss, attack_images)[0]
    adversarial_images = torch.clamp(
        attack_images + epsilon * gradients.sign(),
        min=0.0,
        max=1.0,
    )
    return adversarial_images.detach()


def evaluate_fgsm(model, data_loader, epsilon):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        adversarial_images = generate_fgsm(
            model,
            images,
            labels,
            epsilon,
        )

        with torch.no_grad():
            outputs = model(adversarial_images)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, 100.0 * total_correct / total_samples


def add_controlled_noise(images, probability, noise_std, generator):
    """Return a copy where a controlled portion receives Gaussian noise."""
    augmented_images = images.detach().clone()
    noise_mask = torch.rand(
        augmented_images.size(0),
        generator=generator,
    ) < probability

    if noise_mask.any():
        selected_shape = augmented_images[noise_mask].shape
        noise = torch.randn(
            selected_shape,
            generator=generator,
            dtype=augmented_images.dtype,
        ) * noise_std
        augmented_images[noise_mask] = torch.clamp(
            augmented_images[noise_mask] + noise,
            min=0.0,
            max=1.0,
        )

    return augmented_images, noise_mask


def evaluate_gaussian_noise(model, data_loader, noise_std, seed=SEED + 500):
    """Evaluate using the same deterministic noisy images for fair comparisons."""
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            noise = torch.randn(
                images.shape,
                generator=generator,
                dtype=images.dtype,
            ) * noise_std
            noisy_images = torch.clamp(images + noise, 0.0, 1.0).to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(noisy_images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)

    return total_loss / total_samples, 100.0 * total_correct / total_samples


def evaluate_noise_injected_training_distribution(
    model,
    data_loader,
    probability,
    noise_std,
    seed=SEED + 700,
):
    """Evaluate on the controlled clean/noisy distribution used in training."""
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            augmented_images, _ = add_controlled_noise(
                images,
                probability,
                noise_std,
                generator,
            )
            augmented_images = augmented_images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(augmented_images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)

    return total_loss / total_samples, 100.0 * total_correct / total_samples


# ============================================================
# 6. Load data and existing Task 1/Task 3 models
# ============================================================

poisoned_dataset = PoisonedMNIST(DATA_DIR)

clean_train_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=True,
    transform=transforms.ToTensor(),
    download=True,
)

clean_test_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=False,
    transform=transforms.ToTensor(),
    download=True,
)

clean_train_eval_loader = DataLoader(
    clean_train_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=0,
)

poisoned_train_eval_loader = DataLoader(
    poisoned_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    clean_test_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=0,
)

baseline_path = PROJECT_DIR / "mnist_baseline_model.pth"
poisoned_path = PROJECT_DIR / "mnist_poisoned_model.pth"

if not baseline_path.exists():
    raise FileNotFoundError("Run model.py before Task 4.")
if not poisoned_path.exists():
    raise FileNotFoundError("Run data_poisoning_attack.py before Task 4.")

baseline_model = SimpleNN().to(DEVICE)
load_model_weights(baseline_model, baseline_path)
baseline_model.eval()

poisoned_model = SimpleNN().to(DEVICE)
load_model_weights(poisoned_model, poisoned_path)
poisoned_model.eval()

baseline_clean_loss, baseline_clean_accuracy = evaluate_clean(
    baseline_model,
    test_loader,
)
baseline_fgsm_loss, baseline_fgsm_accuracy = evaluate_fgsm(
    baseline_model,
    test_loader,
    ATTACK_EVALUATION_EPSILON,
)
baseline_noisy_loss, baseline_noisy_accuracy = evaluate_gaussian_noise(
    baseline_model,
    test_loader,
    NOISE_INJECTION_STD,
)
poisoned_clean_loss, poisoned_clean_accuracy = evaluate_clean(
    poisoned_model,
    test_loader,
)


# ============================================================
# 7. Option C: detect and filter suspicious training samples
# ============================================================

inspection_loader = DataLoader(
    poisoned_dataset,
    batch_size=512,
    shuffle=False,
    num_workers=0,
)

kept_indices = []
reference_confidences = []
image_roughness_scores = []
reference_predictions = []
current_index = 0

baseline_model.eval()
with torch.no_grad():
    for images, labels in inspection_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        probabilities = torch.softmax(baseline_model(images), dim=1)
        confidences, predictions = probabilities.max(dim=1)

        horizontal_variation = (
            images[:, :, :, 1:] - images[:, :, :, :-1]
        ).abs().mean(dim=(1, 2, 3))
        vertical_variation = (
            images[:, :, 1:, :] - images[:, :, :-1, :]
        ).abs().mean(dim=(1, 2, 3))
        roughness = (horizontal_variation + vertical_variation) / 2.0

        keep_mask = (
            (predictions == labels)
            & (confidences >= CONFIDENCE_THRESHOLD)
            & (roughness <= ROUGHNESS_THRESHOLD)
        )

        batch_kept = torch.nonzero(
            keep_mask,
            as_tuple=False,
        ).squeeze(1).cpu().numpy()

        kept_indices.extend((batch_kept + current_index).tolist())
        reference_confidences.append(confidences.cpu())
        image_roughness_scores.append(roughness.cpu())
        reference_predictions.append(predictions.cpu())
        current_index += labels.size(0)

kept_indices = np.asarray(kept_indices, dtype=np.int64)
removed_mask = np.ones(len(poisoned_dataset), dtype=bool)
removed_mask[kept_indices] = False
removed_indices = np.where(removed_mask)[0]

actual_poison_mask = poisoned_dataset.poison_mask
poisoned_removed = int((actual_poison_mask & removed_mask).sum())
clean_removed = int((~actual_poison_mask & removed_mask).sum())
poisoned_kept = int((actual_poison_mask & ~removed_mask).sum())
clean_kept = int((~actual_poison_mask & ~removed_mask).sum())

filter_precision = 100.0 * poisoned_removed / max(1, len(removed_indices))
filter_recall = 100.0 * poisoned_removed / max(1, actual_poison_mask.sum())
remaining_label_purity = 100.0 * (
    poisoned_dataset.labels[kept_indices]
    == poisoned_dataset.clean_labels[kept_indices]
).float().mean().item()

np.savez(
    PROJECT_DIR / "Task4_Filtered_Indices.npz",
    kept_indices=kept_indices,
    removed_indices=removed_indices,
)

print("\n========== OPTION C: DATA FILTERING RESULTS ==========")
print(f"Original training samples       : {len(poisoned_dataset)}")
print(f"Samples retained after filtering: {len(kept_indices)}")
print(f"Suspicious samples removed      : {len(removed_indices)}")
print(f"Actually poisoned samples       : {actual_poison_mask.sum()}")
print(f"Poisoned samples removed        : {poisoned_removed}")
print(f"Poisoned samples still retained : {poisoned_kept}")
print(f"Clean samples retained          : {clean_kept}")
print(f"Clean samples removed           : {clean_removed}")
print(f"Filtering precision             : {filter_precision:.2f}%")
print(f"Filtering recall                : {filter_recall:.2f}%")
print(f"Remaining label purity          : {remaining_label_purity:.2f}%")
print("======================================================")

filtered_subset = Subset(poisoned_dataset, kept_indices.tolist())

filtered_eval_loader = DataLoader(
    filtered_subset,
    batch_size=256,
    shuffle=False,
    num_workers=0,
)


def make_filtered_loader(seed):
    return DataLoader(
        filtered_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )


# ============================================================
# 8. Train a filtering-only model (Option C ablation)
# ============================================================


def train_filtering_only():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model = SimpleNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loader = make_filtered_loader(SEED)

    history = {
        "training_loss": [],
        "training_accuracy": [],
        "testing_loss": [],
        "testing_accuracy": [],
    }

    print("\nTraining filtering-only model...\n")

    for epoch in range(FILTER_ONLY_EPOCHS):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)

        train_loss = total_loss / total_samples
        train_accuracy = 100.0 * total_correct / total_samples
        test_loss, test_accuracy = evaluate_clean(model, test_loader)

        history["training_loss"].append(train_loss)
        history["training_accuracy"].append(train_accuracy)
        history["testing_loss"].append(test_loss)
        history["testing_accuracy"].append(test_accuracy)

        print(
            f"Epoch [{epoch + 1}/{FILTER_ONLY_EPOCHS}] | "
            f"Training Loss: {train_loss:.4f} | "
            f"Training Accuracy: {train_accuracy:.2f}% | "
            f"Clean Testing Loss: {test_loss:.4f} | "
            f"Clean Testing Accuracy: {test_accuracy:.2f}%"
        )

    return model, history


filtered_model, filtered_history = train_filtering_only()
torch.save(
    filtered_model.state_dict(),
    PROJECT_DIR / "mnist_filtered_model.pth",
)

filtered_clean_loss, filtered_clean_accuracy = evaluate_clean(
    filtered_model,
    test_loader,
)
filtered_fgsm_loss, filtered_fgsm_accuracy = evaluate_fgsm(
    filtered_model,
    test_loader,
    ATTACK_EVALUATION_EPSILON,
)
filtered_noisy_loss, filtered_noisy_accuracy = evaluate_gaussian_noise(
    filtered_model,
    test_loader,
    NOISE_INJECTION_STD,
)


# ============================================================
# 9. Option B + C: train on filtered data with noise injection
# ============================================================


def train_combined_defense():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model = SimpleNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loader = make_filtered_loader(SEED)
    noise_generator = torch.Generator().manual_seed(SEED + 200)

    history = {
        "training_loss": [],
        "training_accuracy": [],
        "clean_testing_loss": [],
        "clean_testing_accuracy": [],
        "noisy_testing_loss": [],
        "noisy_testing_accuracy": [],
        "fgsm_testing_loss": [],
        "fgsm_testing_accuracy": [],
    }

    print("\nTraining combined Option B + C defended model...\n")

    for epoch in range(DEFENSE_EPOCHS):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_noisy_samples = 0

        for images, labels in loader:
            augmented_images, noise_mask = add_controlled_noise(
                images,
                NOISE_INJECTION_PROBABILITY,
                NOISE_INJECTION_STD,
                noise_generator,
            )
            augmented_images = augmented_images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(augmented_images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * augmented_images.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)
            total_noisy_samples += int(noise_mask.sum().item())

        train_loss = total_loss / total_samples
        train_accuracy = 100.0 * total_correct / total_samples
        clean_loss, clean_accuracy = evaluate_clean(model, test_loader)
        noisy_loss, noisy_accuracy = evaluate_gaussian_noise(
            model,
            test_loader,
            NOISE_INJECTION_STD,
        )
        fgsm_loss, fgsm_accuracy = evaluate_fgsm(
            model,
            test_loader,
            ATTACK_EVALUATION_EPSILON,
        )

        history["training_loss"].append(train_loss)
        history["training_accuracy"].append(train_accuracy)
        history["clean_testing_loss"].append(clean_loss)
        history["clean_testing_accuracy"].append(clean_accuracy)
        history["noisy_testing_loss"].append(noisy_loss)
        history["noisy_testing_accuracy"].append(noisy_accuracy)
        history["fgsm_testing_loss"].append(fgsm_loss)
        history["fgsm_testing_accuracy"].append(fgsm_accuracy)

        applied_percentage = 100.0 * total_noisy_samples / total_samples
        print(
            f"Epoch [{epoch + 1}/{DEFENSE_EPOCHS}] | "
            f"Training Loss: {train_loss:.4f} | "
            f"Training Accuracy: {train_accuracy:.2f}% | "
            f"Noise Applied: {applied_percentage:.1f}% | "
            f"Clean Test Accuracy: {clean_accuracy:.2f}% | "
            f"Noisy Test Accuracy: {noisy_accuracy:.2f}% | "
            f"FGSM Test Accuracy: {fgsm_accuracy:.2f}%"
        )

    return model, history


defended_model, defended_history = train_combined_defense()
torch.save(
    defended_model.state_dict(),
    PROJECT_DIR / "mnist_defended_model.pth",
)

defended_clean_loss, defended_clean_accuracy = evaluate_clean(
    defended_model,
    test_loader,
)
defended_noisy_loss, defended_noisy_accuracy = evaluate_gaussian_noise(
    defended_model,
    test_loader,
    NOISE_INJECTION_STD,
)
defended_fgsm_loss, defended_fgsm_accuracy = evaluate_fgsm(
    defended_model,
    test_loader,
    ATTACK_EVALUATION_EPSILON,
)


# ============================================================
# 10. Final Task 4 comparison
# ============================================================

poison_recovery = defended_clean_accuracy - poisoned_clean_accuracy
noise_robustness_change = defended_noisy_accuracy - baseline_noisy_accuracy
fgsm_robustness_change = defended_fgsm_accuracy - baseline_fgsm_accuracy
clean_accuracy_cost = baseline_clean_accuracy - defended_clean_accuracy

print("\n========== TASK 4 FINAL RESULTS ==========")
print(f"Baseline clean accuracy          : {baseline_clean_accuracy:.2f}%")
print(f"Baseline noisy accuracy (std=.20): {baseline_noisy_accuracy:.2f}%")
print(f"Baseline FGSM accuracy (eps=.20) : {baseline_fgsm_accuracy:.2f}%")
print(f"Poisoned model clean accuracy    : {poisoned_clean_accuracy:.2f}%")
print(f"Filtering-only clean accuracy    : {filtered_clean_accuracy:.2f}%")
print(f"Filtering-only noisy accuracy    : {filtered_noisy_accuracy:.2f}%")
print(f"Filtering-only FGSM accuracy     : {filtered_fgsm_accuracy:.2f}%")
print(f"Defended clean accuracy          : {defended_clean_accuracy:.2f}%")
print(f"Defended noisy accuracy (std=.20): {defended_noisy_accuracy:.2f}%")
print(f"Defended FGSM accuracy (eps=.20) : {defended_fgsm_accuracy:.2f}%")
print(f"Recovery from poisoning          : {poison_recovery:+.2f} points")
print(f"Noisy-input change vs baseline   : {noise_robustness_change:+.2f} points")
print(f"FGSM change vs baseline          : {fgsm_robustness_change:+.2f} points")
print(f"Clean-accuracy cost vs baseline  : {clean_accuracy_cost:+.2f} points")
print("==========================================")


# ============================================================
# 11. Task 4 figures
# ============================================================

# Filtering overview.
plt.figure(figsize=(8, 5))
filter_labels = [
    "Original\ntraining set",
    "Retained\nafter filtering",
    "Removed as\nsuspicious",
]
filter_values = [
    len(poisoned_dataset),
    len(kept_indices),
    len(removed_indices),
]
bars = plt.bar(filter_labels, filter_values)
plt.ylabel("Number of Training Samples")
plt.title("Task 4 Option C: Data Filtering Outcome")
for bar, value in zip(bars, filter_values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 600,
        f"{value:,}",
        ha="center",
    )
plt.ylim(0, 66000)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(
    PROJECT_DIR / "Task4_Data_Filtering_Outcome.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# Combined-defense loss history.
epochs = range(1, DEFENSE_EPOCHS + 1)
plt.figure(figsize=(8, 5))
plt.plot(
    epochs,
    defended_history["training_loss"],
    marker="o",
    label="Noise-injected Training Loss",
)
plt.plot(
    epochs,
    defended_history["clean_testing_loss"],
    marker="s",
    label="Clean Testing Loss",
)
plt.plot(
    epochs,
    defended_history["noisy_testing_loss"],
    marker="^",
    label="Noisy Testing Loss",
)
plt.plot(
    epochs,
    defended_history["fgsm_testing_loss"],
    marker="D",
    label="FGSM Testing Loss",
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Option B + C Training and Testing Loss")
plt.xticks(list(epochs))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(
    PROJECT_DIR / "Task4_Defense_Training_Testing_Loss.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# Combined-defense accuracy history.
plt.figure(figsize=(8, 5))
plt.plot(
    epochs,
    defended_history["training_accuracy"],
    marker="o",
    label="Noise-injected Training Accuracy",
)
plt.plot(
    epochs,
    defended_history["clean_testing_accuracy"],
    marker="s",
    label="Clean Testing Accuracy",
)
plt.plot(
    epochs,
    defended_history["noisy_testing_accuracy"],
    marker="^",
    label="Noisy Testing Accuracy",
)
plt.plot(
    epochs,
    defended_history["fgsm_testing_accuracy"],
    marker="D",
    label="FGSM Testing Accuracy",
)
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Option B + C Training and Testing Accuracy")
plt.xticks(list(epochs))
plt.ylim(0, 100)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(
    PROJECT_DIR / "Task4_Defense_Training_Testing_Accuracy.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# Main defense comparison.
comparison_names = [
    "Baseline\nClean",
    "Baseline\nNoisy",
    "Baseline\nFGSM",
    "Poisoned\nClean",
    "Filtered\nClean",
    "Defended\nClean",
    "Defended\nNoisy",
    "Defended\nFGSM",
]
comparison_values = [
    baseline_clean_accuracy,
    baseline_noisy_accuracy,
    baseline_fgsm_accuracy,
    poisoned_clean_accuracy,
    filtered_clean_accuracy,
    defended_clean_accuracy,
    defended_noisy_accuracy,
    defended_fgsm_accuracy,
]
plt.figure(figsize=(12, 6))
bars = plt.bar(comparison_names, comparison_values)
plt.ylabel("Classification Accuracy (%)")
plt.title("Task 4 Group 5 Defense Performance: Noise Injection + Data Filtering")
plt.ylim(0, 105)
plt.grid(axis="y", alpha=0.3)
for bar, value in zip(bars, comparison_values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1.3,
        f"{value:.2f}%",
        ha="center",
        fontsize=8,
    )
plt.tight_layout()
plt.savefig(
    PROJECT_DIR / "Task4_Defense_Performance_Comparison.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# Show clean and controlled-noise versions used by Option B.
example_images, example_labels = next(iter(test_loader))
noise_example_generator = torch.Generator().manual_seed(SEED + 900)
noise = torch.randn(
    example_images[:8].shape,
    generator=noise_example_generator,
    dtype=example_images.dtype,
) * NOISE_INJECTION_STD
noisy_examples = torch.clamp(example_images[:8] + noise, 0.0, 1.0)

plt.figure(figsize=(14, 4))
for i in range(8):
    plt.subplot(2, 8, i + 1)
    plt.imshow(example_images[i].squeeze(), cmap="gray")
    plt.title(f"Clean {example_labels[i].item()}", fontsize=8)
    plt.axis("off")

    plt.subplot(2, 8, i + 9)
    plt.imshow(noisy_examples[i].squeeze(), cmap="gray")
    plt.title(f"Noise std={NOISE_INJECTION_STD:.2f}", fontsize=8)
    plt.axis("off")
plt.suptitle("Task 4 Option B: Controlled Noise-Injection Examples")
plt.tight_layout()
plt.savefig(
    PROJECT_DIR / "Task4_Noise_Injection_Examples.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# Show retained and removed examples.
reference_confidences = torch.cat(reference_confidences).numpy()
image_roughness_scores = torch.cat(image_roughness_scores).numpy()
reference_predictions = torch.cat(reference_predictions).numpy()

example_kept = kept_indices[:5]
example_removed = removed_indices[:5]
example_indices = np.concatenate([example_kept, example_removed])

plt.figure(figsize=(13, 5))
for position, index in enumerate(example_indices):
    plt.subplot(2, 5, position + 1)
    plt.imshow(poisoned_dataset.images[index].squeeze(), cmap="gray")
    status = "Retained" if position < 5 else "Removed"
    plt.title(
        f"{status}\n"
        f"Label {poisoned_dataset.labels[index].item()}, "
        f"Ref {reference_predictions[index]}\n"
        f"Conf {reference_confidences[index]:.2f}, "
        f"R {image_roughness_scores[index]:.2f}",
        fontsize=8,
    )
    plt.axis("off")
plt.suptitle("Task 4 Option C: Examples Retained and Removed by Filtering")
plt.tight_layout()
plt.savefig(
    PROJECT_DIR / "Task4_Filtered_Sample_Examples.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# ============================================================
# 12. Save Task 4 text results
# ============================================================

with open(
    PROJECT_DIR / "Task4_Defense_Results.txt",
    "w",
    encoding="utf-8",
) as result_file:
    result_file.write("TASK 4: GROUP 5 DEFENSES (OPTION B + OPTION C)\n")
    result_file.write("=" * 70 + "\n")
    result_file.write("OPTION C - DATA FILTERING\n")
    result_file.write(f"Original samples: {len(poisoned_dataset)}\n")
    result_file.write(f"Retained samples: {len(kept_indices)}\n")
    result_file.write(f"Removed samples: {len(removed_indices)}\n")
    result_file.write(f"Filtering precision: {filter_precision:.2f}%\n")
    result_file.write(f"Filtering recall: {filter_recall:.2f}%\n")
    result_file.write(f"Remaining label purity: {remaining_label_purity:.2f}%\n\n")

    result_file.write("OPTION B - CONTROLLED NOISE INJECTION\n")
    result_file.write(f"Gaussian-noise standard deviation: {NOISE_INJECTION_STD:.2f}\n")
    result_file.write(
        f"Probability per training image: {NOISE_INJECTION_PROBABILITY:.0%}\n"
    )
    result_file.write(f"Training epochs: {DEFENSE_EPOCHS}\n\n")

    result_file.write("FINAL PERFORMANCE\n")
    result_file.write("-" * 70 + "\n")
    result_file.write(
        f"Baseline clean: Loss={baseline_clean_loss:.4f}, "
        f"Accuracy={baseline_clean_accuracy:.2f}%\n"
    )
    result_file.write(
        f"Baseline Gaussian noise: Loss={baseline_noisy_loss:.4f}, "
        f"Accuracy={baseline_noisy_accuracy:.2f}%\n"
    )
    result_file.write(
        f"Baseline FGSM: Loss={baseline_fgsm_loss:.4f}, "
        f"Accuracy={baseline_fgsm_accuracy:.2f}%\n"
    )
    result_file.write(
        f"Poisoned clean: Loss={poisoned_clean_loss:.4f}, "
        f"Accuracy={poisoned_clean_accuracy:.2f}%\n"
    )
    result_file.write(
        f"Filtering-only clean: Loss={filtered_clean_loss:.4f}, "
        f"Accuracy={filtered_clean_accuracy:.2f}%\n"
    )
    result_file.write(
        f"Filtering-only Gaussian noise: Loss={filtered_noisy_loss:.4f}, "
        f"Accuracy={filtered_noisy_accuracy:.2f}%\n"
    )
    result_file.write(
        f"Defended clean: Loss={defended_clean_loss:.4f}, "
        f"Accuracy={defended_clean_accuracy:.2f}%\n"
    )
    result_file.write(
        f"Defended Gaussian noise: Loss={defended_noisy_loss:.4f}, "
        f"Accuracy={defended_noisy_accuracy:.2f}%\n"
    )
    result_file.write(
        f"Defended FGSM: Loss={defended_fgsm_loss:.4f}, "
        f"Accuracy={defended_fgsm_accuracy:.2f}%\n\n"
    )
    result_file.write(
        f"Poisoning recovery: {poison_recovery:+.2f} percentage points\n"
    )
    result_file.write(
        f"Noisy-input change versus baseline: "
        f"{noise_robustness_change:+.2f} percentage points\n"
    )
    result_file.write(
        f"FGSM change versus baseline: "
        f"{fgsm_robustness_change:+.2f} percentage points\n"
    )
    result_file.write(
        f"Clean-accuracy difference versus baseline: "
        f"{-clean_accuracy_cost:+.2f} percentage points\n"
    )

print("\nTask 4 completed successfully for Group 5.")
print("Generated:")
print("1. mnist_filtered_model.pth")
print("2. mnist_defended_model.pth")
print("3. Task4_Defense_Results.txt")
print("4. Task4_Data_Filtering_Outcome.png")
print("5. Task4_Noise_Injection_Examples.png")
print("6. Task4_Defense_Training_Testing_Loss.png")
print("7. Task4_Defense_Training_Testing_Accuracy.png")
print("8. Task4_Defense_Performance_Comparison.png")
print("9. Task4_Filtered_Sample_Examples.png")


# ============================================================
# TASK 5 AND OPTIONAL BONUS EXTENSIONS
# ============================================================

# Task 5 uses the models produced above, so no second script is needed.
SELECTED_EPSILON = ATTACK_EVALUATION_EPSILON
PGD_STEPS = 10
EPSILON_VALUES = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

print("\n========== TASK 5 AND BONUS CONFIGURATION ==========")
print(f"Selected epsilon       : {SELECTED_EPSILON:.2f}")
print(f"PGD iterations         : {PGD_STEPS}")
print(f"Epsilon sweep          : {EPSILON_VALUES}")
print("Bonus 1                : PGD attack and FGSM comparison")
print("Bonus 2                : Perturbation-strength analysis")
print("====================================================")

# ============================================================
# 3. Metrics
# ============================================================


def confusion_matrix_from_predictions(labels, predictions, classes=10):
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for actual, predicted in zip(labels, predictions):
        matrix[int(actual), int(predicted)] += 1
    return matrix


def classification_metrics(labels, predictions):
    matrix = confusion_matrix_from_predictions(labels, predictions)
    true_positive = np.diag(matrix).astype(float)
    false_positive = matrix.sum(axis=0) - true_positive
    false_negative = matrix.sum(axis=1) - true_positive

    precision = np.divide(
        true_positive,
        true_positive + false_positive,
        out=np.zeros_like(true_positive),
        where=(true_positive + false_positive) != 0,
    )
    recall = np.divide(
        true_positive,
        true_positive + false_negative,
        out=np.zeros_like(true_positive),
        where=(true_positive + false_negative) != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    accuracy = 100.0 * true_positive.sum() / matrix.sum()

    return {
        "accuracy": accuracy,
        "precision": 100.0 * precision.mean(),
        "recall": 100.0 * recall.mean(),
        "f1": 100.0 * f1.mean(),
        "confusion_matrix": matrix,
    }


# ============================================================
# 4. Clean, FGSM, and PGD evaluation
# ============================================================


def fgsm_attack(model, images, labels, epsilon):
    if epsilon == 0:
        return images.detach()

    attack_images = images.detach().clone().requires_grad_(True)
    outputs = model(attack_images)
    loss = criterion(outputs, labels)
    gradients = torch.autograd.grad(loss, attack_images)[0]

    return torch.clamp(
        attack_images + epsilon * gradients.sign(),
        min=0.0,
        max=1.0,
    ).detach()


def pgd_attack(model, images, labels, epsilon, steps):
    if epsilon == 0:
        return images.detach()

    # The step size scales with epsilon and remains large enough at small eps.
    alpha = max(epsilon / 5.0, 0.005)

    # Random initialization inside the allowed L-infinity region.
    adversarial_images = images.detach() + torch.empty_like(images).uniform_(
        -epsilon,
        epsilon,
    )
    adversarial_images = torch.clamp(adversarial_images, 0.0, 1.0)

    for _ in range(steps):
        adversarial_images.requires_grad_(True)
        outputs = model(adversarial_images)
        loss = criterion(outputs, labels)
        gradients = torch.autograd.grad(loss, adversarial_images)[0]

        adversarial_images = adversarial_images.detach() + alpha * gradients.sign()
        perturbation = torch.clamp(
            adversarial_images - images,
            min=-epsilon,
            max=epsilon,
        )
        adversarial_images = torch.clamp(
            images + perturbation,
            min=0.0,
            max=1.0,
        ).detach()

    return adversarial_images


def evaluate_model(model, attack=None, epsilon=0.0, pgd_steps=10):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_labels = []
    all_predictions = []

    # Make PGD random starts reproducible for each evaluation.
    torch.manual_seed(SEED + int(epsilon * 1000) + pgd_steps)

    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        if attack == "fgsm":
            evaluated_images = fgsm_attack(
                model,
                images,
                labels,
                epsilon,
            )
        elif attack == "pgd":
            evaluated_images = pgd_attack(
                model,
                images,
                labels,
                epsilon,
                pgd_steps,
            )
        else:
            evaluated_images = images

        with torch.no_grad():
            outputs = model(evaluated_images)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        all_labels.append(labels.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())

    labels = np.concatenate(all_labels)
    predictions = np.concatenate(all_predictions)
    metrics = classification_metrics(labels, predictions)
    metrics["loss"] = total_loss / total_samples
    metrics["labels"] = labels
    metrics["predictions"] = predictions
    return metrics


def evaluate_noisy_model_metrics(model, noise_std):
    model.eval()
    generator = torch.Generator().manual_seed(SEED + 500)
    total_loss = 0.0
    total_samples = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            noise = torch.randn(
                images.shape,
                generator=generator,
                dtype=images.dtype,
            ) * noise_std
            noisy_images = torch.clamp(images + noise, 0.0, 1.0).to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(noisy_images)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())

    labels_array = np.concatenate(all_labels)
    predictions_array = np.concatenate(all_predictions)
    metrics = classification_metrics(labels_array, predictions_array)
    metrics["loss"] = total_loss / total_samples
    metrics["labels"] = labels_array
    metrics["predictions"] = predictions_array
    return metrics


# ============================================================
# 5. Task 5 required model comparison
# ============================================================

print("\nEvaluating the models required by Task 5...")

baseline_clean = evaluate_model(baseline_model)
baseline_fgsm = evaluate_model(
    baseline_model,
    attack="fgsm",
    epsilon=SELECTED_EPSILON,
)
poisoned_clean = evaluate_model(poisoned_model)
defended_clean = evaluate_model(defended_model)
defended_fgsm = evaluate_model(
    defended_model,
    attack="fgsm",
    epsilon=SELECTED_EPSILON,
)

# Option B is additionally evaluated against deterministic Gaussian noise.
baseline_noisy_metrics = evaluate_noisy_model_metrics(
    baseline_model,
    NOISE_INJECTION_STD,
)
defended_noisy_metrics = evaluate_noisy_model_metrics(
    defended_model,
    NOISE_INJECTION_STD,
)

required_results = [
    {
        "Model / Condition": "Baseline Model (Clean)",
        **baseline_clean,
    },
    {
        "Model / Condition": "Baseline under Gaussian Noise (std=0.20)",
        **baseline_noisy_metrics,
    },
    {
        "Model / Condition": "Baseline under FGSM (eps=0.20)",
        **baseline_fgsm,
    },
    {
        "Model / Condition": "Poisoned Model (Clean Test)",
        **poisoned_clean,
    },
    {
        "Model / Condition": "Defended Model B+C (Clean)",
        **defended_clean,
    },
    {
        "Model / Condition": "Defended Model B+C under Gaussian Noise (std=0.20)",
        **defended_noisy_metrics,
    },
    {
        "Model / Condition": "Defended Model B+C under FGSM (eps=0.20)",
        **defended_fgsm,
    },
]

# Report training loss separately because FGSM attack conditions are
# test-time evaluations, not separately trained models. Each value is measured
# after training on the model's own training distribution.
baseline_training_loss, baseline_training_accuracy = evaluate_clean(
    baseline_model,
    clean_train_eval_loader,
)
poisoned_training_loss, poisoned_training_accuracy = evaluate_clean(
    poisoned_model,
    poisoned_train_eval_loader,
)
filtered_training_loss, filtered_training_accuracy = evaluate_clean(
    filtered_model,
    filtered_eval_loader,
)
defended_training_loss, defended_training_accuracy = (
    evaluate_noise_injected_training_distribution(
        defended_model,
        filtered_eval_loader,
        NOISE_INJECTION_PROBABILITY,
        NOISE_INJECTION_STD,
    )
)

training_loss_summary = [
    {
        "Model": "Baseline Model",
        "Training Data": "Clean MNIST training set",
        "Training Loss": baseline_training_loss,
        "Training Accuracy": baseline_training_accuracy,
    },
    {
        "Model": "Poisoned Model",
        "Training Data": "Poisoned MNIST training set",
        "Training Loss": poisoned_training_loss,
        "Training Accuracy": poisoned_training_accuracy,
    },
    {
        "Model": "Filtering-only Model",
        "Training Data": "Filtered training set",
        "Training Loss": filtered_training_loss,
        "Training Accuracy": filtered_training_accuracy,
    },
    {
        "Model": "Group 5 Defended Model",
        "Training Data": "Filtered set; controlled noise applied to 50% of images",
        "Training Loss": defended_training_loss,
        "Training Accuracy": defended_training_accuracy,
    },
]

print("\n========== TASK 5 TRAINING LOSS SUMMARY ==========")
for row in training_loss_summary:
    print(
        f"{row['Model']:<25} | "
        f"Loss: {row['Training Loss']:.4f} | "
        f"Accuracy: {row['Training Accuracy']:.2f}%"
    )
print("Note: Each model is evaluated on its own training distribution.")
print("Attack conditions are omitted because they are test-time evaluations, not trained models.")
print("==================================================")

with open(
    PROJECT_DIR / "Task5_Training_Loss_Summary.csv",
    "w",
    newline="",
    encoding="utf-8",
) as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(
        [
            "Model",
            "Training Data Used",
            "Post-training Loss",
            "Post-training Accuracy (%)",
        ]
    )
    for row in training_loss_summary:
        writer.writerow(
            [
                row["Model"],
                row["Training Data"],
                f"{row['Training Loss']:.4f}",
                f"{row['Training Accuracy']:.2f}",
            ]
        )

plt.figure(figsize=(9, 5))
plt.bar(
    [row["Model"] for row in training_loss_summary],
    [row["Training Loss"] for row in training_loss_summary],
)
plt.ylabel("Post-training Loss")
plt.title("Task 5 Training Loss Summary")
plt.xticks(rotation=15, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(
    PROJECT_DIR / "Task5_Training_Loss_Summary.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

print("\n========== TASK 5 PERFORMANCE COMPARISON ==========")
for result in required_results:
    print(
        f"{result['Model / Condition']:<39} | "
        f"Loss: {result['loss']:.4f} | "
        f"Accuracy: {result['accuracy']:.2f}% | "
        f"Precision: {result['precision']:.2f}% | "
        f"Recall: {result['recall']:.2f}% | "
        f"F1: {result['f1']:.2f}%"
    )
print("===================================================")

# Save CSV table.
with open(
    PROJECT_DIR / "Task5_Performance_Comparison.csv",
    "w",
    newline="",
    encoding="utf-8",
) as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(
        [
            "Model / Condition",
            "Testing Loss",
            "Accuracy (%)",
            "Macro Precision (%)",
            "Macro Recall (%)",
            "Macro F1-score (%)",
        ]
    )
    for result in required_results:
        writer.writerow(
            [
                result["Model / Condition"],
                f"{result['loss']:.4f}",
                f"{result['accuracy']:.2f}",
                f"{result['precision']:.2f}",
                f"{result['recall']:.2f}",
                f"{result['f1']:.2f}",
            ]
        )

# Save report-readable text.
with open(
    PROJECT_DIR / "Task5_Performance_Results.txt",
    "w",
    encoding="utf-8",
) as result_file:
    result_file.write("TASK 5: PERFORMANCE EVALUATION AND ANALYSIS\n")
    result_file.write("=" * 75 + "\n")
    result_file.write("TRAINING LOSS SUMMARY\n")
    result_file.write("-" * 75 + "\n")
    for row in training_loss_summary:
        result_file.write(f"{row['Model']}\n")
        result_file.write(f"  Training data: {row['Training Data']}\n")
        result_file.write(
            f"  Post-training loss: {row['Training Loss']:.4f}\n"
        )
        result_file.write(
            f"  Post-training accuracy: "
            f"{row['Training Accuracy']:.2f}%\n\n"
        )
    result_file.write(
        "Note: Each loss is evaluated after training on the model's own "
        "training distribution. Noise/FGSM attack conditions are excluded because "
        "they are test-time evaluations, not separately trained models.\n\n"
    )
    result_file.write("TESTING PERFORMANCE COMPARISON\n")
    result_file.write("-" * 75 + "\n")
    for result in required_results:
        result_file.write(f"{result['Model / Condition']}\n")
        result_file.write(f"  Testing Loss: {result['loss']:.4f}\n")
        result_file.write(f"  Accuracy: {result['accuracy']:.2f}%\n")
        result_file.write(
            f"  Macro Precision: {result['precision']:.2f}%\n"
        )
        result_file.write(f"  Macro Recall: {result['recall']:.2f}%\n")
        result_file.write(f"  Macro F1-score: {result['f1']:.2f}%\n\n")

    result_file.write("KEY CHANGES\n")
    result_file.write("-" * 75 + "\n")
    result_file.write(
        f"FGSM reduced baseline accuracy by "
        f"{baseline_clean['accuracy'] - baseline_fgsm['accuracy']:.2f} points.\n"
    )
    result_file.write(
        f"Poisoning reduced clean accuracy by "
        f"{baseline_clean['accuracy'] - poisoned_clean['accuracy']:.2f} points.\n"
    )
    result_file.write(
        f"The Group 5 defended model changed clean accuracy by "
        f"{defended_clean['accuracy'] - poisoned_clean['accuracy']:+.2f} points "
        f"relative to the poisoned model.\n"
    )
    result_file.write(
        f"Under Gaussian noise, the defended model changed accuracy by "
        f"{defended_noisy_metrics['accuracy'] - baseline_noisy_metrics['accuracy']:+.2f} "
        f"points relative to the baseline.\n"
    )
    result_file.write(
        f"Under FGSM, the defended model changed accuracy by "
        f"{defended_fgsm['accuracy'] - baseline_fgsm['accuracy']:+.2f} points "
        f"relative to the baseline.\n"
    )

# ============================================================
# 6. Task 5 figures
# ============================================================

names = [result["Model / Condition"] for result in required_results]
short_names = [
    "Baseline\nClean",
    "Baseline\nNoisy",
    "Baseline\nFGSM",
    "Poisoned\nClean",
    "Defended\nClean",
    "Defended\nNoisy",
    "Defended\nFGSM",
]
accuracies = [result["accuracy"] for result in required_results]
losses = [result["loss"] for result in required_results]

plt.figure(figsize=(12, 6))
bars = plt.bar(short_names, accuracies)
plt.ylabel("Classification Accuracy (%)")
plt.title("Task 5 Accuracy Comparison")
plt.ylim(0, 105)
plt.grid(axis="y", alpha=0.3)
for bar, value in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1.3,
        f"{value:.2f}%",
        ha="center",
        fontsize=9,
    )
plt.tight_layout()
plt.savefig(
    PROJECT_DIR / "Task5_Accuracy_Comparison.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

plt.figure(figsize=(12, 6))
bars = plt.bar(short_names, losses)
plt.ylabel("Testing Loss")
plt.title("Task 5 Testing-Loss Comparison")
plt.grid(axis="y", alpha=0.3)
for bar, value in zip(bars, losses):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + max(losses) * 0.015,
        f"{value:.3f}",
        ha="center",
        fontsize=9,
    )
plt.tight_layout()
plt.savefig(
    PROJECT_DIR / "Task5_Loss_Comparison.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# Confusion matrices for the main model states.
confusion_items = [
    ("Baseline Clean", baseline_clean["confusion_matrix"]),
    ("Baseline FGSM", baseline_fgsm["confusion_matrix"]),
    ("Poisoned Clean", poisoned_clean["confusion_matrix"]),
    ("Defended B+C Clean", defended_clean["confusion_matrix"]),
]

fig, axes = plt.subplots(2, 2, figsize=(13, 11))
for axis, (title, matrix) in zip(axes.flat, confusion_items):
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_title(title)
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("True Label")
    axis.set_xticks(range(10))
    axis.set_yticks(range(10))
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
plt.suptitle("Task 5 Confusion-Matrix Comparison", fontsize=15)
plt.tight_layout()
plt.savefig(
    PROJECT_DIR / "Task5_Confusion_Matrices.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# ============================================================
# 7. BONUS 1: PGD attack and comparison with FGSM
# ============================================================

print("\nRunning Bonus 1: PGD attack...")

baseline_pgd = evaluate_model(
    baseline_model,
    attack="pgd",
    epsilon=SELECTED_EPSILON,
    pgd_steps=PGD_STEPS,
)
defended_pgd = evaluate_model(
    defended_model,
    attack="pgd",
    epsilon=SELECTED_EPSILON,
    pgd_steps=PGD_STEPS,
)

print("\n========== BONUS 1: FGSM VS PGD ==========")
print(f"Baseline FGSM accuracy : {baseline_fgsm['accuracy']:.2f}%")
print(f"Baseline PGD accuracy  : {baseline_pgd['accuracy']:.2f}%")
print(f"Defended FGSM accuracy : {defended_fgsm['accuracy']:.2f}%")
print(f"Defended PGD accuracy  : {defended_pgd['accuracy']:.2f}%")
print("==========================================")

attack_names = [
    "Baseline\nFGSM",
    "Baseline\nPGD",
    "Defended\nFGSM",
    "Defended\nPGD",
]
attack_values = [
    baseline_fgsm["accuracy"],
    baseline_pgd["accuracy"],
    defended_fgsm["accuracy"],
    defended_pgd["accuracy"],
]
plt.figure(figsize=(8, 6))
bars = plt.bar(attack_names, attack_values)
plt.ylabel("Classification Accuracy (%)")
plt.title("Bonus 1: FGSM and PGD Attack Comparison at Epsilon 0.20")
plt.ylim(0, 105)
plt.grid(axis="y", alpha=0.3)
for bar, value in zip(bars, attack_values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1.3,
        f"{value:.2f}%",
        ha="center",
    )
plt.tight_layout()
plt.savefig(
    PROJECT_DIR / "Bonus1_FGSM_vs_PGD_Comparison.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# ============================================================
# 8. BONUS 2: vary perturbation strength
# ============================================================

print("\nRunning Bonus 2: epsilon sensitivity analysis...")

bonus_rows = []
for epsilon in EPSILON_VALUES:
    print(f"  Evaluating epsilon={epsilon:.2f}...")

    baseline_fgsm_e = evaluate_model(
        baseline_model,
        attack="fgsm",
        epsilon=epsilon,
    )
    defended_fgsm_e = evaluate_model(
        defended_model,
        attack="fgsm",
        epsilon=epsilon,
    )
    baseline_pgd_e = evaluate_model(
        baseline_model,
        attack="pgd",
        epsilon=epsilon,
        pgd_steps=PGD_STEPS,
    )
    defended_pgd_e = evaluate_model(
        defended_model,
        attack="pgd",
        epsilon=epsilon,
        pgd_steps=PGD_STEPS,
    )

    bonus_rows.append(
        {
            "epsilon": epsilon,
            "baseline_fgsm": baseline_fgsm_e["accuracy"],
            "defended_fgsm": defended_fgsm_e["accuracy"],
            "baseline_pgd": baseline_pgd_e["accuracy"],
            "defended_pgd": defended_pgd_e["accuracy"],
        }
    )

with open(
    PROJECT_DIR / "Bonus2_Epsilon_Sensitivity_Results.csv",
    "w",
    newline="",
    encoding="utf-8",
) as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(
        [
            "Epsilon",
            "Baseline FGSM Accuracy (%)",
            "Defended FGSM Accuracy (%)",
            "Baseline PGD Accuracy (%)",
            "Defended PGD Accuracy (%)",
        ]
    )
    for row in bonus_rows:
        writer.writerow(
            [
                f"{row['epsilon']:.2f}",
                f"{row['baseline_fgsm']:.2f}",
                f"{row['defended_fgsm']:.2f}",
                f"{row['baseline_pgd']:.2f}",
                f"{row['defended_pgd']:.2f}",
            ]
        )

plt.figure(figsize=(9, 6))
epsilons = [row["epsilon"] for row in bonus_rows]
plt.plot(
    epsilons,
    [row["baseline_fgsm"] for row in bonus_rows],
    marker="o",
    label="Baseline - FGSM",
)
plt.plot(
    epsilons,
    [row["defended_fgsm"] for row in bonus_rows],
    marker="s",
    label="Defended - FGSM",
)
plt.plot(
    epsilons,
    [row["baseline_pgd"] for row in bonus_rows],
    marker="^",
    label="Baseline - PGD",
)
plt.plot(
    epsilons,
    [row["defended_pgd"] for row in bonus_rows],
    marker="D",
    label="Defended - PGD",
)
plt.xlabel("Perturbation Strength (Epsilon)")
plt.ylabel("Classification Accuracy (%)")
plt.title("Bonus 2: Accuracy versus Attack Strength")
plt.xticks(EPSILON_VALUES)
plt.ylim(0, 105)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(
    PROJECT_DIR / "Bonus2_Accuracy_vs_Epsilon.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# PGD original/adversarial examples using the baseline model.
example_images, example_labels = next(iter(test_loader))
example_images = example_images[:8].to(DEVICE)
example_labels = example_labels[:8].to(DEVICE)
example_pgd = pgd_attack(
    baseline_model,
    example_images,
    example_labels,
    SELECTED_EPSILON,
    PGD_STEPS,
)

with torch.no_grad():
    clean_predictions = baseline_model(example_images).argmax(1)
    pgd_predictions = baseline_model(example_pgd).argmax(1)

plt.figure(figsize=(14, 5))
for i in range(8):
    plt.subplot(2, 8, i + 1)
    plt.imshow(example_images[i].cpu().squeeze(), cmap="gray")
    plt.title(
        f"True {example_labels[i].item()}\n"
        f"Pred {clean_predictions[i].item()}",
        fontsize=8,
    )
    plt.axis("off")

    plt.subplot(2, 8, i + 9)
    plt.imshow(example_pgd[i].cpu().squeeze(), cmap="gray")
    plt.title(
        f"PGD\nPred {pgd_predictions[i].item()}",
        fontsize=8,
    )
    plt.axis("off")
plt.suptitle("Bonus 1: Original and PGD Adversarial Examples")
plt.tight_layout()
plt.savefig(
    PROJECT_DIR / "Bonus1_Original_vs_PGD_Examples.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# ============================================================
# 9. Save bonus text summary
# ============================================================

with open(
    PROJECT_DIR / "Bonus_Extensions_Results.txt",
    "w",
    encoding="utf-8",
) as result_file:
    result_file.write("OPTIONAL BONUS EXTENSIONS\n")
    result_file.write("=" * 70 + "\n")
    result_file.write("BONUS 1: PGD ATTACK AND COMPARISON WITH FGSM\n")
    result_file.write(
        f"Selected epsilon: {SELECTED_EPSILON:.2f}; "
        f"PGD steps: {PGD_STEPS}; "
        f"step size: max(epsilon/5, 0.005).\n"
    )
    result_file.write(
        f"Baseline FGSM accuracy: {baseline_fgsm['accuracy']:.2f}%\n"
    )
    result_file.write(
        f"Baseline PGD accuracy: {baseline_pgd['accuracy']:.2f}%\n"
    )
    result_file.write(
        f"Defended FGSM accuracy: {defended_fgsm['accuracy']:.2f}%\n"
    )
    result_file.write(
        f"Defended PGD accuracy: {defended_pgd['accuracy']:.2f}%\n\n"
    )

    result_file.write("BONUS 2: EFFECT OF VARYING PERTURBATION STRENGTH\n")
    result_file.write("-" * 70 + "\n")
    for row in bonus_rows:
        result_file.write(
            f"Epsilon {row['epsilon']:.2f}: "
            f"Baseline FGSM={row['baseline_fgsm']:.2f}%, "
            f"Defended FGSM={row['defended_fgsm']:.2f}%, "
            f"Baseline PGD={row['baseline_pgd']:.2f}%, "
            f"Defended PGD={row['defended_pgd']:.2f}%\n"
        )

print("\nTasks 4 and 5 for Group 5, including both bonus extensions, completed successfully.")
print("Generated:")
print("1. Task5_Performance_Comparison.csv")
print("2. Task5_Training_Loss_Summary.csv")
print("3. Task5_Training_Loss_Summary.png")
print("4. Task5_Performance_Results.txt")
print("5. Task5_Accuracy_Comparison.png")
print("6. Task5_Loss_Comparison.png")
print("7. Task5_Confusion_Matrices.png")
print("8. Bonus1_FGSM_vs_PGD_Comparison.png")
print("9. Bonus1_Original_vs_PGD_Examples.png")
print("10. Bonus2_Epsilon_Sensitivity_Results.csv")
print("11. Bonus2_Accuracy_vs_Epsilon.png")
print("12. Bonus_Extensions_Results.txt")
