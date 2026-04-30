import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

from data.pipeline import DataIngestionPipeline
from models.cnn_model import BrainTumorCNN

def train_cnn(epochs=5, batch_size=32, learning_rate=0.001):
    print("="*50)
    print("CNN TRAINING PIPELINE")
    print("="*50)

    # 1. Load Data
    ingestion = DataIngestionPipeline(data_dir="data")
    print("Loading training data...")
    X_train, y_train = ingestion.load_data(subset="Training")
    
    # Convert to Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    
    dataset = TensorDataset(X_train_t, y_train_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. Initialize Model, Loss, and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = BrainTumorCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 3. Training Loop
    model.train()
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (i + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        print(f"-> Epoch {epoch+1} Complete. Avg Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    # 4. Save the Model
    save_path = "best_model.pth"
    torch.save(model.state_dict(), save_path)
    print("="*50)
    print(f"SUCCESS: Model weights saved to '{save_path}'")
    print("="*50)

if __name__ == "__main__":
    # Check if data exists before running
    if os.path.exists("data/Training"):
        train_cnn(epochs=3) # 3 epochs is usually enough for a baseline in this task
    else:
        print("Error: 'data/Training' directory not found. Cannot train.")
