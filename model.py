import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Set random seeds
seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Select GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device used:", device)

#Task 1: Baseline Deep Learning Model

# Convert MNIST images into tensors
transform = transforms.ToTensor()

# Download the clean MNIST training dataset
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)

# Download the MNIST testing dataset
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True
)

print("Number of training images:", len(train_dataset))
print("Number of testing images:", len(test_dataset))

batch_size = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

print("Training batches:", len(train_loader))
print("Testing batches:", len(test_loader))

images, labels = next(iter(train_loader))

plt.figure(figsize=(10, 4))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images[i].squeeze(), cmap="gray")
    plt.title(f"Label: {labels[i].item()}")
    plt.axis("off")

plt.tight_layout()
plt.show()

#baseline nueral network

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 28 × 28 image pixels = 784 input features
        self.fc1 = nn.Linear(784, 128)

        # 10 output classes for digits 0 to 9
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Flatten each image from [batch, 1, 28, 28]
        # into [batch, 784]
        x = x.view(x.size(0), -1)

        # First fully connected layer with ReLU
        x = torch.relu(self.fc1(x))

        # Output raw scores for the 10 classes
        x = self.fc2(x)

        return x
    
model = SimpleNN().to(device)

print(model)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

#training loss
def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Clear gradients from the previous batch
        optimizer.zero_grad()

        # Forward propagation
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Accumulate the total loss
        total_loss += loss.item() * images.size(0)

        # Obtain predicted classes
        _, predicted = torch.max(outputs, dim=1)

        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    average_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples

    return average_loss, accuracy

#testing loss
def evaluate_model(model, data_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Disable gradient calculations during testing
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, dim=1)

            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    average_loss = total_loss / total_samples
    # Accuracy
    accuracy = 100.0 * total_correct / total_samples 

    return average_loss, accuracy

num_epochs = 5

training_losses = []
testing_losses = []

training_accuracies = []
testing_accuracies = []

for epoch in range(num_epochs):
    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    test_loss, test_accuracy = evaluate_model(
        model,
        test_loader,
        criterion,
        device
    )

    training_losses.append(train_loss)
    testing_losses.append(test_loss)

    training_accuracies.append(train_accuracy)
    testing_accuracies.append(test_accuracy)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] | "
        f"Training Loss: {train_loss:.4f} | "
        f"Training Accuracy: {train_accuracy:.2f}% | "
        f"Testing Loss: {test_loss:.4f} | "
        f"Testing Accuracy: {test_accuracy:.2f}%"
    )

final_training_loss = training_losses[-1]
final_testing_loss = testing_losses[-1]
final_testing_accuracy = testing_accuracies[-1]

print("\n========== BASELINE MODEL RESULTS ==========")
print(f"Final Training Loss       : {final_training_loss:.4f}")
print(f"Final Testing Loss        : {final_testing_loss:.4f}")
print(f"Classification Accuracy  : {final_testing_accuracy:.2f}%")
print("============================================")

epochs = range(1, num_epochs + 1)

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    training_losses,
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs,
    testing_losses,
    marker="s",
    label="Testing Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Baseline Model Training and Testing Loss")
plt.xticks(epochs)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "Baseline_Training_Testing_Loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    training_accuracies,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epochs,
    testing_accuracies,
    marker="s",
    label="Testing Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Baseline Model Training and Testing Accuracy")
plt.xticks(epochs)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "Baseline_Training_Testing_Accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

model.eval()

images, labels = next(iter(test_loader))
images = images.to(device)

with torch.no_grad():
    outputs = model(images)
    _, predictions = torch.max(outputs, dim=1)

images = images.cpu()
predictions = predictions.cpu()

plt.figure(figsize=(12, 6))

for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.imshow(images[i].squeeze(), cmap="gray")

    actual = labels[i].item()
    predicted = predictions[i].item()

    plt.title(
        f"Actual: {actual}\nPredicted: {predicted}"
    )

    plt.axis("off")

plt.tight_layout()

plt.savefig(
    "Baseline_Sample_Predictions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

torch.save(
    model.state_dict(),
    "mnist_baseline_model.pth"
)

print("Baseline model saved as mnist_baseline_model.pth")

loaded_model = SimpleNN().to(device)

loaded_model.load_state_dict(
    torch.load(
        "mnist_baseline_model.pth",
        map_location=device
    )
)

loaded_model.eval()

print("Baseline model loaded successfully.")