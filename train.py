import torch
import torch.nn as nn
import torch.optim as optim

from collections import Counter
from dataset import get_dataloaders
from model import EmotionClassificationNN

# Training function for one epoch
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        # Move data to GPU / CPU
        images = images.to(device)
        labels = labels.to(device)

        # Clear old gradients -> forward pass
        optimizer.zero_grad()
        outputs = model(images)

        # Calculate loss and backpropagate
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    # Calculate epoch loss and accuracy and return
    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def main():

    train_path = "TrainData"
    test_path = "TestData"
    best_val_acc = 0.0

    batch_size = 64
    epochs = 30
    learning_rate = 1e-3
    weight_decay = 1e-4

    # Use Cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, train_dataset = get_dataloaders( train_path, batch_size=batch_size, shuffle=True )
    val_loader, val_dataset = get_dataloaders( test_path, batch_size=batch_size, shuffle=False )

    # Dealing with class imbalance
    class_names = train_dataset.classes
    num_classes = len(class_names)
    class_counts = Counter(train_dataset.targets)

    # Normalize weights (keeps loss scale stable)
    class_weights = torch.tensor( [1.0 / class_counts[i] for i in range(num_classes)], dtype=torch.float )
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = class_weights.to(device)

    # Model
    model = EmotionClassificationNN( num_classes=num_classes )
    model.to( device )

    # Training setup
    criterion = nn.CrossEntropyLoss( label_smoothing=0.1 )
    optimizer = optim.Adam( model.parameters(), lr=learning_rate, weight_decay=weight_decay )

    # Training loop
    for epoch in range( epochs ):

        train_loss, train_acc = train_one_epoch( model, train_loader, criterion, optimizer, device )
        val_loss, val_acc = validate( model, val_loader, criterion, device )

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save( model.state_dict(), "Models/best_emotion_model.pth" )

    print( f"Best Validation Accuracy: {best_val_acc:.4f}" )


if __name__ == "__main__":
    main()