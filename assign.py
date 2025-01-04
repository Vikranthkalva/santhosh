import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# Data Simulation
def create_non_iid_data(dataset, num_clients):
    data_split = []
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    
    for i in range(num_clients):
        split = indices[i::num_clients]
        data_split.append(torch.utils.data.Subset(dataset, split))
    return data_split

# CNN Model
def build_model():
    return nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(64 * 6 * 6, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

# Training Local Models
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Federated Averaging
def federated_averaging(models, weights):
    global_model = models[0]
    for key in global_model.state_dict().keys():
        global_weights = torch.zeros_like(global_model.state_dict()[key])
        for i, model in enumerate(models):
            global_weights += weights[i] * model.state_dict()[key]
        global_model.state_dict()[key].copy_(global_weights)
    return global_model

# Validation
def validate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# Main Flow
if __name__ == "__main__":
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root="./data", train=False, transform=transform)

    # Simulate Non-IID Data
    num_clients = 5
    client_data = create_non_iid_data(train_data, num_clients)

    # Initialize Models and Optimizers
    models = [build_model().to(device) for _ in range(num_clients)]
    optimizers = [optim.SGD(model.parameters(), lr=0.01) for model in models]
    criterion = nn.CrossEntropyLoss()

    # Prepare Test Loader
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

    # Train Models Locally
    global_accuracies = []
    for i, data in enumerate(client_data):
        train_loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)
        train_model(models[i], train_loader, criterion, optimizers[i], device)

    # Federated Aggregation
    weights = [1 / num_clients] * num_clients  # Equal weighting for simplicity
    global_model = federated_averaging(models, weights)

    # Validate Global Model
    accuracy = validate_model(global_model, test_loader, device)
    global_accuracies.append(accuracy)

    # Visualization
    plt.plot(global_accuracies, label='Global Model Accuracy', color='blue')
    plt.xlabel('Training Rounds')
    plt.ylabel('Accuracy (%)')
    plt.title('Global Model Performance')
    plt.legend()
    plt.show()

    print("Federated Learning Complete")
