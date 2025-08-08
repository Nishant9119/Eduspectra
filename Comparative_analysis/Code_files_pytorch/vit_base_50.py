from timm import create_model
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import torch.nn as nn

# Data Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
checkpoint_path = r"/home/ied10015.21/vit_base_patch16_224.pth"
# Load Dataset
train_dataset = ImageFolder(r"/home/ied10015.21/train_images/", transform=transform)
test_dataset = ImageFolder(r"/home/ied10015.21/test_images/", transform=transform)


# DataLoader
batch_size = 128
train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(test_dataset, batch_size=batch_size) 

# Model, Loss Function, Optimizer
model = create_model('vit_base_patch16_224',pretrained =False,num_classes=2)
model.load_state_dict(torch.load(checkpoint_path))
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

print(f"Model vit_base_patch16_224 epoch 50---------------------------------------")
# Training Loop
num_epochs =50
print("Started Training----------------------------------------")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(train_dl):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i + 1) % 100 == 0:
            print(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}")
            running_loss = 0.0
print("Finished Training---------------------------------------")

# Validation and Metrics Calculation
model.eval()
correct = 0
total = 0
true_positives = 0
false_positives = 0
false_negatives = 0
true_negatives = 0

with torch.no_grad():
    for inputs, targets in val_dl:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        # Count total correct predictions
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

        # Confusion matrix components
        true_positives += ((predicted == 1) & (targets == 1)).sum().item()
        false_positives += ((predicted == 1) & (targets == 0)).sum().item()
        false_negatives += ((predicted == 0) & (targets == 1)).sum().item()
        true_negatives += ((predicted == 0) & (targets == 0)).sum().item()

# Compute metrics
accuracy = 100 * correct / total
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Confusion Matrix
confusion_matrix = torch.tensor([[true_negatives, false_positives],
                                 [false_negatives, true_positives]])

# Print Metrics
print(f"Accuracy on Preprocessed test set: {accuracy:.2f}%")
print(f"Recall: {recall:.2f}")
print(f"Precision: {precision:.2f}")
print(f"F1 Score: {f1:.2f}")
print("Confusion Matrix:")
print(confusion_matrix)
