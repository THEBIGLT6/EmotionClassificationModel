import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

from collections import Counter
from dataset import get_dataloaders
from model import EmotionClassificationNN, EmotionClassificationAttentionNN

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


# Function for validating the model, returning loss, accuracy and F1 score
@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    epoch_f1 = f1_score( all_labels.numpy(), all_preds.numpy(), average="macro" )

    return epoch_loss, epoch_acc, epoch_f1


# Main training loop
def train_model( model, model_name, train_loader, val_loader, criterion, optimizer, device, epochs ):
    best_val_acc = 0.0
    best_f1_score = 0.0

    for epoch in range( epochs ):

        train_loss, train_acc = train_one_epoch( model, train_loader, criterion, optimizer, device )
        val_loss, val_acc, val_f1 = validate( model, val_loader, criterion, device )

        print( f"[{model_name}] Epoch [{epoch+1}/{epochs}] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}" )

        if val_f1 > best_f1_score:
            best_f1_score = val_f1
            torch.save( model.state_dict(), f"Models/best_{model_name}.pth" )

    print( f"[{model_name}] Best Val F1 Score: {best_f1_score:.4f}" )
    return best_f1_score


if __name__ == "__main__":

    train_path = "TrainData"
    test_path = "TestData"

    batch_size = 64
    epochs = 30
    learning_rate = 1e-3
    weight_decay = 1e-4

    # Use Cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, train_dataset = get_dataloaders( train_path, batch_size=batch_size, shuffle=True, is_train=True )
    val_loader, val_dataset = get_dataloaders( test_path, batch_size=batch_size, shuffle=False, is_train=False )

    # Dealing with class imbalance
    class_names = train_dataset.classes
    num_classes = len(class_names)
    class_counts = Counter(train_dataset.targets)

    # Normalize weights (keeps loss scale stable)
    class_weights = torch.tensor( [1.0 / class_counts[i] for i in range(num_classes)], dtype=torch.float )
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = class_weights.to(device)

    # Model training
    criterion = nn.CrossEntropyLoss( weight=class_weights, label_smoothing=0.1 )

    baseline_model = EmotionClassificationNN(num_classes=num_classes).to(device)
    baseline_optimizer = optim.Adam( baseline_model.parameters(), lr=learning_rate, weight_decay=weight_decay )
    
    baseline_acc = train_model(
        model=baseline_model,
        model_name="baseline",
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=baseline_optimizer,
        device=device,
        epochs=epochs
    )

    attention_model = EmotionClassificationAttentionNN(num_classes=num_classes).to(device)
    attention_optimizer = optim.Adam( attention_model.parameters(), lr=learning_rate, weight_decay=weight_decay )

    attention_acc = train_model(
        model=attention_model,
        model_name="attention",
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=attention_optimizer,
        device=device,
        epochs=epochs
    )