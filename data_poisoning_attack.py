from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
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
# 2. Task 3 poisoning configuration
# ============================================================

LABEL_FLIP_PERCENTAGE = 60
NOISE_PERCENTAGE = 60
NOISE_STANDARD_DEVIATION = 1.50

BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

print("\n========== POISONING CONFIGURATION ==========")
print(f"Label-flip percentage       : {LABEL_FLIP_PERCENTAGE}%")
print(f"Noisy-image percentage      : {NOISE_PERCENTAGE}%")
print(f"Noise standard deviation    : {NOISE_STANDARD_DEVIATION:.2f}")
print("=============================================")


# ============================================================
# 3. Poisoned MNIST training dataset
# ============================================================

class PoisonedMNIST(Dataset):
    """Create a fixed, reproducible poisoned copy of MNIST training data."""

    def __init__(
        self,
        root,
        label_flip_percentage,
        noise_percentage,
        noise_standard_deviation,
        seed,
    ):
        clean_dataset = datasets.MNIST(
            root=root,
            train=True,
            download=True,
        )

        # Convert raw images from [60000, 28, 28] uint8 to
        # [60000, 1, 28, 28] floating-point tensors in [0, 1].
        self.clean_images = (
            clean_dataset.data.float().unsqueeze(1) / 255.0
        )
        self.images = self.clean_images.clone()

        self.clean_labels = clean_dataset.targets.clone()
        self.labels = self.clean_labels.clone()

        number_of_samples = len(self.labels)
        rng = np.random.default_rng(seed)

        number_of_flipped_labels = int(
            number_of_samples * label_flip_percentage / 100
        )
        number_of_noisy_images = int(
            number_of_samples * noise_percentage / 100
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

        # Flip selected labels to systematically incorrect classes.
        original_selected_labels = self.labels[
            self.label_flip_indices
        ]

        # 0 -> 1, 1 -> 2, ..., 9 -> 0
        flipped_labels = (
            original_selected_labels + 1
        ) % 10

        self.labels[self.label_flip_indices] = flipped_labels

        # Add fixed Gaussian noise to the selected training images.
        noise_generator = torch.Generator().manual_seed(seed + 1)
        noise = torch.randn(
            (
                number_of_noisy_images,
                1,
                28,
                28,
            ),
            generator=noise_generator,
        ) * noise_standard_deviation

        poisoned_images = (
            self.images[self.noise_indices] + noise
        )

        self.images[self.noise_indices] = torch.clamp(
            poisoned_images,
            min=0.0,
            max=1.0,
        )

        self.poisoned_indices = np.union1d(
            self.label_flip_indices,
            self.noise_indices,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


poisoned_train_dataset = PoisonedMNIST(
    root=DATA_DIR,
    label_flip_percentage=LABEL_FLIP_PERCENTAGE,
    noise_percentage=NOISE_PERCENTAGE,
    noise_standard_deviation=NOISE_STANDARD_DEVIATION,
    seed=SEED,
)

# The test set remains completely clean.
test_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=False,
    transform=transforms.ToTensor(),
    download=True,
)

train_generator = torch.Generator().manual_seed(SEED)

poisoned_train_loader = DataLoader(
    poisoned_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=train_generator,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

print("\n========== DATASET INFORMATION ==========")
print("Training images               :", len(poisoned_train_dataset))
print("Clean testing images          :", len(test_dataset))
print(
    "Number of flipped labels     :",
    len(poisoned_train_dataset.label_flip_indices),
)
print(
    "Number of noisy images       :",
    len(poisoned_train_dataset.noise_indices),
)
print(
    "Unique poisoned samples      :",
    len(poisoned_train_dataset.poisoned_indices),
)
print("=========================================")


# ============================================================
# 4. Define the same neural network used for the baseline
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


criterion = nn.CrossEntropyLoss()


# ============================================================
# 5. Training and evaluation functions
# ============================================================

def train_one_epoch(model, data_loader, optimizer):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)

    average_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples

    return average_loss, accuracy


def evaluate_model(model, data_loader, return_predictions=False):
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


def load_model_weights(model, model_path):
    """Load weights while remaining compatible with older PyTorch versions."""
    try:
        state_dict = torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=True,
        )
    except TypeError:
        state_dict = torch.load(
            model_path,
            map_location=DEVICE,
        )

    model.load_state_dict(state_dict)


# ============================================================
# 6. Load and evaluate the existing baseline model
# ============================================================

baseline_model_path = PROJECT_DIR / "mnist_baseline_model.pth"

if not baseline_model_path.exists():
    raise FileNotFoundError(
        "mnist_baseline_model.pth was not found. "
        "Run model.py first before running Task 3."
    )

baseline_model = SimpleNN().to(DEVICE)
load_model_weights(baseline_model, baseline_model_path)
baseline_model.eval()

baseline_test_loss, baseline_test_accuracy = evaluate_model(
    baseline_model,
    test_loader,
)

print("\n========== BASELINE MODEL RESULTS ==========")
print(f"Clean Testing Loss        : {baseline_test_loss:.4f}")
print(f"Classification Accuracy  : {baseline_test_accuracy:.2f}%")
print("============================================")


# ============================================================
# 7. Retrain a new model using the poisoned training dataset
# ============================================================

# Reset the seed so the poisoned model starts reproducibly.
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

poisoned_model = SimpleNN().to(DEVICE)

optimizer = optim.Adam(
    poisoned_model.parameters(),
    lr=LEARNING_RATE,
)

training_losses = []
testing_losses = []
training_accuracies = []
testing_accuracies = []

print("\nTraining the model using poisoned data...\n")

for epoch in range(NUM_EPOCHS):
    train_loss, train_accuracy = train_one_epoch(
        poisoned_model,
        poisoned_train_loader,
        optimizer,
    )

    test_loss, test_accuracy = evaluate_model(
        poisoned_model,
        test_loader,
    )

    training_losses.append(train_loss)
    testing_losses.append(test_loss)
    training_accuracies.append(train_accuracy)
    testing_accuracies.append(test_accuracy)

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] | "
        f"Poisoned Training Loss: {train_loss:.4f} | "
        f"Poisoned Training Accuracy: {train_accuracy:.2f}% | "
        f"Clean Testing Loss: {test_loss:.4f} | "
        f"Clean Testing Accuracy: {test_accuracy:.2f}%"
    )


# ============================================================
# 8. Final evaluation and comparison
# ============================================================

(
    poisoned_test_loss,
    poisoned_test_accuracy,
    clean_test_labels,
    poisoned_test_predictions,
) = evaluate_model(
    poisoned_model,
    test_loader,
    return_predictions=True,
)

accuracy_reduction = (
    baseline_test_accuracy - poisoned_test_accuracy
)

loss_increase = poisoned_test_loss - baseline_test_loss

print("\n========== TASK 3 FINAL COMPARISON ==========")
print(f"Baseline Test Accuracy       : {baseline_test_accuracy:.2f}%")
print(f"Poisoned Model Accuracy      : {poisoned_test_accuracy:.2f}%")
print(f"Accuracy Reduction           : {accuracy_reduction:.2f} percentage points")
print(f"Baseline Testing Loss        : {baseline_test_loss:.4f}")
print(f"Poisoned Model Testing Loss  : {poisoned_test_loss:.4f}")
print(f"Testing Loss Increase        : {loss_increase:.4f}")
print("=============================================")


# ============================================================
# 9. Save the poisoned model
# ============================================================

poisoned_model_path = PROJECT_DIR / "mnist_poisoned_model.pth"
torch.save(poisoned_model.state_dict(), poisoned_model_path)

print("\nPoisoned model saved as mnist_poisoned_model.pth")


# ============================================================
# 10. Plot training and clean-testing loss
# ============================================================

epochs = range(1, NUM_EPOCHS + 1)

plt.figure(figsize=(8, 5))
plt.plot(
    epochs,
    training_losses,
    marker="o",
    label="Poisoned Training Loss",
)
plt.plot(
    epochs,
    testing_losses,
    marker="s",
    label="Clean Testing Loss",
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Poisoned Model Training and Testing Loss")
plt.xticks(list(epochs))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(
    PROJECT_DIR / "Poisoning_Training_Testing_Loss.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# ============================================================
# 11. Plot training and clean-testing accuracy
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(
    epochs,
    training_accuracies,
    marker="o",
    label="Poisoned Training Accuracy",
)
plt.plot(
    epochs,
    testing_accuracies,
    marker="s",
    label="Clean Testing Accuracy",
)
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Poisoned Model Training and Testing Accuracy")
plt.xticks(list(epochs))
plt.ylim(0, 100)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(
    PROJECT_DIR / "Poisoning_Training_Testing_Accuracy.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# ============================================================
# 12. Plot baseline and poisoned-model comparison
# ============================================================

model_names = [
    "Baseline Model",
    "Poisoned Model",
]

model_accuracies = [
    baseline_test_accuracy,
    poisoned_test_accuracy,
]

plt.figure(figsize=(7, 5))
bars = plt.bar(model_names, model_accuracies)
plt.ylabel("Classification Accuracy (%)")
plt.title("Baseline and Poisoned Model Accuracy Comparison", pad=12)
plt.ylim(0, 105)

for bar, value in zip(bars, model_accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.2f}%",
        ha="center",
    )

plt.tight_layout()
plt.savefig(
    PROJECT_DIR / "Poisoning_Baseline_vs_Poisoned_Accuracy.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# ============================================================
# 13. Display original and poisoned training samples
# ============================================================

label_flip_set = set(
    poisoned_train_dataset.label_flip_indices.tolist()
)
noise_set = set(
    poisoned_train_dataset.noise_indices.tolist()
)

label_only_indices = sorted(label_flip_set - noise_set)[:2]
noise_only_indices = sorted(noise_set - label_flip_set)[:2]
both_attack_indices = sorted(label_flip_set & noise_set)[:2]

example_indices = (
    label_only_indices
    + noise_only_indices
    + both_attack_indices
)

plt.figure(figsize=(12, 12))

for row, sample_index in enumerate(example_indices):
    clean_image = poisoned_train_dataset.clean_images[
        sample_index
    ].squeeze().numpy()

    poisoned_image = poisoned_train_dataset.images[
        sample_index
    ].squeeze().numpy()

    clean_label = int(
        poisoned_train_dataset.clean_labels[sample_index]
    )

    poisoned_label = int(
        poisoned_train_dataset.labels[sample_index]
    )

    was_noisy = sample_index in noise_set
    was_flipped = sample_index in label_flip_set

    attack_description = []
    if was_noisy:
        attack_description.append("Noise")
    if was_flipped:
        attack_description.append("Label flip")

    plt.subplot(len(example_indices), 2, row * 2 + 1)
    plt.imshow(clean_image, cmap="gray", vmin=0, vmax=1)
    plt.title(f"Original: label {clean_label}")
    plt.axis("off")

    plt.subplot(len(example_indices), 2, row * 2 + 2)
    plt.imshow(poisoned_image, cmap="gray", vmin=0, vmax=1)
    plt.title(
        f"Poisoned: label {poisoned_label}\n"
        + " + ".join(attack_description)
    )
    plt.axis("off")

plt.tight_layout()
plt.savefig(
    PROJECT_DIR / "Poisoning_Original_vs_Poisoned_Samples.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# ============================================================
# 14. Create poisoned-model confusion matrix
# ============================================================

confusion_matrix = torch.zeros(
    (10, 10),
    dtype=torch.int64,
)

for actual_label, predicted_label in zip(
    clean_test_labels,
    poisoned_test_predictions,
):
    confusion_matrix[actual_label, predicted_label] += 1

plt.figure(figsize=(8, 7))
plt.imshow(confusion_matrix.numpy(), interpolation="nearest")
plt.title("Poisoned Model Confusion Matrix on Clean Test Data")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.xticks(range(10))
plt.yticks(range(10))
plt.colorbar()

threshold = confusion_matrix.max().item() / 2

for row in range(10):
    for column in range(10):
        value = confusion_matrix[row, column].item()
        plt.text(
            column,
            row,
            str(value),
            ha="center",
            va="center",
            color="white" if value > threshold else "black",
            fontsize=7,
        )

plt.tight_layout()
plt.savefig(
    PROJECT_DIR / "Poisoning_Confusion_Matrix.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# ============================================================
# 15. Save Task 3 results to a text file
# ============================================================

results_path = PROJECT_DIR / "Data_Poisoning_Task3_Results.txt"

with results_path.open("w", encoding="utf-8") as file:
    file.write("TASK 3: DATA POISONING ATTACK RESULTS\n")
    file.write("=" * 50 + "\n")
    file.write(
        f"Label-Flip Percentage: {LABEL_FLIP_PERCENTAGE}%\n"
    )
    file.write(
        f"Noisy-Image Percentage: {NOISE_PERCENTAGE}%\n"
    )
    file.write(
        "Noise Standard Deviation: "
        f"{NOISE_STANDARD_DEVIATION:.2f}\n"
    )
    file.write(
        "Number of Flipped Labels: "
        f"{len(poisoned_train_dataset.label_flip_indices)}\n"
    )
    file.write(
        "Number of Noisy Images: "
        f"{len(poisoned_train_dataset.noise_indices)}\n"
    )
    file.write(
        "Unique Poisoned Samples: "
        f"{len(poisoned_train_dataset.poisoned_indices)}\n\n"
    )

    file.write("EPOCH RESULTS\n")
    file.write("-" * 50 + "\n")

    for epoch_index in range(NUM_EPOCHS):
        file.write(
            f"Epoch {epoch_index + 1}: "
            f"Training Loss = {training_losses[epoch_index]:.4f}, "
            f"Training Accuracy = "
            f"{training_accuracies[epoch_index]:.2f}%, "
            f"Testing Loss = {testing_losses[epoch_index]:.4f}, "
            f"Testing Accuracy = "
            f"{testing_accuracies[epoch_index]:.2f}%\n"
        )

    file.write("\nFINAL COMPARISON\n")
    file.write("-" * 50 + "\n")
    file.write(
        f"Baseline Testing Loss: {baseline_test_loss:.4f}\n"
    )
    file.write(
        f"Baseline Test Accuracy: {baseline_test_accuracy:.2f}%\n"
    )
    file.write(
        f"Poisoned Model Testing Loss: {poisoned_test_loss:.4f}\n"
    )
    file.write(
        f"Poisoned Model Accuracy: {poisoned_test_accuracy:.2f}%\n"
    )
    file.write(
        "Accuracy Reduction: "
        f"{accuracy_reduction:.2f} percentage points\n"
    )
    file.write(
        f"Testing Loss Increase: {loss_increase:.4f}\n"
    )

print("\nTask 3 completed successfully.")
print("Generated files:")
print("1. mnist_poisoned_model.pth")
print("2. Poisoning_Training_Testing_Loss.png")
print("3. Poisoning_Training_Testing_Accuracy.png")
print("4. Poisoning_Baseline_vs_Poisoned_Accuracy.png")
print("5. Poisoning_Original_vs_Poisoned_Samples.png")
print("6. Poisoning_Confusion_Matrix.png")
print("7. Data_Poisoning_Task3_Results.txt")
