import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# 1. Reproducibility and device
# ============================================================

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device used:", device)


# ============================================================
# 2. Load MNIST test dataset
# ============================================================

transform = transforms.ToTensor()

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

print("Number of testing images:", len(test_dataset))
print("Number of testing batches:", len(test_loader))


# ============================================================
# 3. Define the same baseline model architecture
# ============================================================

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Flatten images from [batch, 1, 28, 28] to [batch, 784]
        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# ============================================================
# 4. Load the trained baseline model
# ============================================================

model = SimpleNN().to(device)

model_path = "mnist_baseline_model.pth"

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"{model_path} was not found. "
        "Please run model.py first to train and save the baseline model."
    )

model.load_state_dict(
    torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )
)

model.eval()

print("Baseline model loaded successfully.")


# ============================================================
# 5. Loss function
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# 6. FGSM attack function
# ============================================================

def fgsm_attack(image, epsilon, image_gradient):
    """
    Generate an adversarial image using FGSM.

    Parameters:
        image:
            Original clean image.

        epsilon:
            Perturbation strength.

        image_gradient:
            Gradient of the loss with respect to the image.

    Returns:
        perturbed_image:
            Adversarial image constrained to the pixel range [0, 1].
    """

    # Obtain the sign of the image gradient
    gradient_sign = image_gradient.sign()

    # Add the perturbation
    perturbed_image = image + epsilon * gradient_sign

    # Keep MNIST pixel values between 0 and 1
    perturbed_image = torch.clamp(
        perturbed_image,
        min=0,
        max=1
    )

    return perturbed_image


# ============================================================
# 7. Evaluate clean test accuracy
# ============================================================

def evaluate_clean_model(model, data_loader):
    model.eval()

    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += labels.size(0)

    average_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples

    return average_loss, accuracy


# ============================================================
# 8. Evaluate model under FGSM attack
# ============================================================

def evaluate_fgsm_attack(
    model,
    data_loader,
    epsilon,
    maximum_examples=10
):
    model.eval()

    total_correct = 0
    total_samples = 0
    total_adversarial_loss = 0.0

    adversarial_examples = []

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Input gradients are required for FGSM
        images.requires_grad = True

        # Clear previously stored gradients
        model.zero_grad()

        # Make predictions on clean images
        clean_outputs = model(images)

        # Calculate clean-image loss
        clean_loss = criterion(clean_outputs, labels)

        # Calculate gradients with respect to input images
        clean_loss.backward()

        image_gradients = images.grad.detach()

        # Generate adversarial images
        adversarial_images = fgsm_attack(
            images,
            epsilon,
            image_gradients
        )

        # Evaluate the model using adversarial images
        adversarial_outputs = model(adversarial_images)

        adversarial_loss = criterion(
            adversarial_outputs,
            labels
        )

        total_adversarial_loss += (
            adversarial_loss.item() * images.size(0)
        )

        clean_predictions = clean_outputs.argmax(dim=1)
        adversarial_predictions = adversarial_outputs.argmax(dim=1)

        total_correct += (
            adversarial_predictions == labels
        ).sum().item()

        total_samples += labels.size(0)

        # Store examples where the clean prediction was correct
        # but the adversarial prediction changed
        for index in range(images.size(0)):
            clean_prediction = clean_predictions[index].item()
            adversarial_prediction = (
                adversarial_predictions[index].item()
            )

            actual_label = labels[index].item()

            if (
                clean_prediction == actual_label
                and adversarial_prediction != actual_label
                and len(adversarial_examples) < maximum_examples
            ):
                original_image = (
                    images[index]
                    .detach()
                    .cpu()
                    .squeeze()
                    .numpy()
                )

                adversarial_image = (
                    adversarial_images[index]
                    .detach()
                    .cpu()
                    .squeeze()
                    .numpy()
                )

                perturbation = (
                    adversarial_images[index]
                    - images[index]
                )

                perturbation = (
                    perturbation
                    .detach()
                    .cpu()
                    .squeeze()
                    .numpy()
                )

                adversarial_examples.append(
                    {
                        "original_image": original_image,
                        "adversarial_image": adversarial_image,
                        "perturbation": perturbation,
                        "actual_label": actual_label,
                        "clean_prediction": clean_prediction,
                        "adversarial_prediction":
                            adversarial_prediction
                    }
                )

    average_loss = total_adversarial_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples

    return average_loss, accuracy, adversarial_examples


# ============================================================
# 9. Evaluate the clean baseline model
# ============================================================

clean_test_loss, clean_test_accuracy = evaluate_clean_model(
    model,
    test_loader
)

print("\n========== CLEAN TEST RESULTS ==========")
print(f"Testing Loss            : {clean_test_loss:.4f}")
print(f"Classification Accuracy : {clean_test_accuracy:.2f}%")
print("========================================")


# ============================================================
# 10. Evaluate different epsilon values
# ============================================================

epsilon_values = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30
]

fgsm_results = []
saved_examples = {}

print("\n========== FGSM ATTACK RESULTS ==========")

for epsilon in epsilon_values:
    fgsm_loss, fgsm_accuracy, examples = evaluate_fgsm_attack(
        model,
        test_loader,
        epsilon,
        maximum_examples=10
    )

    fgsm_results.append(
        {
            "epsilon": epsilon,
            "loss": fgsm_loss,
            "accuracy": fgsm_accuracy
        }
    )

    saved_examples[epsilon] = examples

    print(
        f"Epsilon: {epsilon:.2f} | "
        f"Adversarial Loss: {fgsm_loss:.4f} | "
        f"Adversarial Accuracy: {fgsm_accuracy:.2f}%"
    )

print("=========================================")


# ============================================================
# 11. Select a main epsilon for Task 2 comparison
# ============================================================

main_epsilon = 0.20

main_result = next(
    result
    for result in fgsm_results
    if result["epsilon"] == main_epsilon
)

main_fgsm_accuracy = main_result["accuracy"]
main_fgsm_loss = main_result["loss"]

accuracy_reduction = (
    clean_test_accuracy - main_fgsm_accuracy
)

print("\n========== TASK 2 FINAL COMPARISON ==========")
print(f"Selected Epsilon                : {main_epsilon:.2f}")
print(f"Clean Test Accuracy             : {clean_test_accuracy:.2f}%")
print(f"FGSM Adversarial Accuracy       : {main_fgsm_accuracy:.2f}%")
print(f"Accuracy Reduction              : {accuracy_reduction:.2f} percentage points")
print(f"Clean Testing Loss              : {clean_test_loss:.4f}")
print(f"FGSM Adversarial Testing Loss   : {main_fgsm_loss:.4f}")
print("=============================================")


# ============================================================
# 12. Plot accuracy against epsilon
# ============================================================

epsilons = [
    result["epsilon"]
    for result in fgsm_results
]

accuracies = [
    result["accuracy"]
    for result in fgsm_results
]

plt.figure(figsize=(8, 5))

plt.plot(
    epsilons,
    accuracies,
    marker="o"
)

plt.axhline(
    y=clean_test_accuracy,
    linestyle="--",
    label=f"Clean accuracy: {clean_test_accuracy:.2f}%"
)

plt.xlabel("Epsilon")
plt.ylabel("Classification Accuracy (%)")
plt.title("Effect of FGSM Perturbation Strength on Accuracy")
plt.xticks(epsilons)
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    "FGSM_Accuracy_vs_Epsilon.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 13. Plot clean and FGSM accuracy comparison
# ============================================================

comparison_names = [
    "Clean Test Data",
    f"FGSM Data\nε = {main_epsilon}"
]

comparison_accuracies = [
    clean_test_accuracy,
    main_fgsm_accuracy
]

plt.figure(figsize=(7, 5))

bars = plt.bar(
    comparison_names,
    comparison_accuracies
)

plt.ylabel("Classification Accuracy (%)")
plt.title("Clean and FGSM Adversarial Accuracy Comparison")
plt.ylim(0, 100)

for bar, value in zip(bars, comparison_accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "FGSM_Clean_vs_Adversarial_Accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 14. Display original and adversarial image examples
# ============================================================

examples = saved_examples[main_epsilon]

if len(examples) == 0:
    print(
        "No successful adversarial examples were stored "
        f"for epsilon {main_epsilon}."
    )

else:
    number_to_show = min(5, len(examples))

    plt.figure(figsize=(12, number_to_show * 3))

    for index in range(number_to_show):
        example = examples[index]

        # Original image
        plt.subplot(number_to_show, 3, index * 3 + 1)

        plt.imshow(
            example["original_image"],
            cmap="gray",
            vmin=0,
            vmax=1
        )

        plt.title(
            "Original Image\n"
            f"Actual: {example['actual_label']}, "
            f"Prediction: {example['clean_prediction']}"
        )

        plt.axis("off")

        # Perturbation
        plt.subplot(number_to_show, 3, index * 3 + 2)

        plt.imshow(
            example["perturbation"],
            cmap="gray"
        )

        plt.title(
            f"FGSM Perturbation\nε = {main_epsilon}"
        )

        plt.axis("off")

        # Adversarial image
        plt.subplot(number_to_show, 3, index * 3 + 3)

        plt.imshow(
            example["adversarial_image"],
            cmap="gray",
            vmin=0,
            vmax=1
        )

        plt.title(
            "Adversarial Image\n"
            f"Prediction: "
            f"{example['adversarial_prediction']}"
        )

        plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        "FGSM_Original_Perturbation_Adversarial_Examples.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 15. Save results to a text file
# ============================================================

with open(
    "FGSM_Task2_Results.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write("TASK 2: FGSM ATTACK RESULTS\n")
    file.write("=" * 45 + "\n")

    file.write(
        f"Clean Testing Loss: {clean_test_loss:.4f}\n"
    )

    file.write(
        f"Clean Test Accuracy: {clean_test_accuracy:.2f}%\n\n"
    )

    for result in fgsm_results:
        file.write(
            f"Epsilon: {result['epsilon']:.2f}, "
            f"Adversarial Loss: {result['loss']:.4f}, "
            f"Adversarial Accuracy: "
            f"{result['accuracy']:.2f}%\n"
        )

    file.write("\n")
    file.write(f"Selected Epsilon: {main_epsilon:.2f}\n")

    file.write(
        f"Selected FGSM Accuracy: "
        f"{main_fgsm_accuracy:.2f}%\n"
    )

    file.write(
        f"Accuracy Reduction: "
        f"{accuracy_reduction:.2f} percentage points\n"
    )

print("\nTask 2 results saved successfully.")
print("Generated files:")
print("1. FGSM_Accuracy_vs_Epsilon.png")
print("2. FGSM_Clean_vs_Adversarial_Accuracy.png")
print("3. FGSM_Original_Perturbation_Adversarial_Examples.png")
print("4. FGSM_Task2_Results.txt")