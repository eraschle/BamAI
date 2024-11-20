import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Daten laden
points = np.load("data/processed/train_points.npy")
labels = np.load("data/processed/train_labels.npy")

# Konvertiere zu Tensoren
points_tensor = torch.tensor(points, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.long)

# Erstelle Dataset und DataLoader
dataset = TensorDataset(points_tensor, labels_tensor)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Einfaches Modell
model = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 4))

# Training
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for batch_points, batch_labels in data_loader:
        optimizer.zero_grad()
        outputs = model(batch_points)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}: Loss = {loss.item()}")  # pyright: ignore[reportPossiblyUnboundVariable]
