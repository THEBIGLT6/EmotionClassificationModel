# evaluate.py
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from dataset import get_dataloaders
from model import EmotionClassificationNN, EmotionClassificationAttentionNN


@torch.no_grad()
def evaluate_model( model, loader, device, class_names, title, save_path ):
    model.eval()

    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    plt.figure( figsize=(8, 6) )
    sns.heatmap( cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names )

    plt.xlabel( "Predicted Label" )
    plt.ylabel( "True Label" )
    plt.title( title )
    plt.tight_layout()

    os.makedirs( os.path.dirname(save_path), exist_ok=True )
    plt.savefig( save_path, dpi=300 )
    plt.close()

    print( f"Saved {title} to {save_path}" )


if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    baseline_model = "Models/best_baseline.pth"
    attention_model = "Models/best_attention.pth"
    test_path = "TestData"
    batch_size = 64

    class_names = [ "angry", "disgust", "fear", "happy", "neutral", "sad", "surprised" ]

    # Data loader
    loader, _ = get_dataloaders( test_path, batch_size=batch_size, shuffle=False )

    # Baseline Model 
    baseline = EmotionClassificationNN( num_classes=len(class_names) )
    baseline.load_state_dict( torch.load(baseline_model, map_location=device) )
    baseline.to( device )

    evaluate_model( baseline, loader, device, class_names, title="Baseline CNN Normalized Confusion Matrix", save_path="Results/confusion_baseline.png" )

    # Attention Model
    attention = EmotionClassificationAttentionNN( num_classes=len(class_names) )
    attention.load_state_dict( torch.load( attention_model, map_location=device) )
    attention.to( device )

    evaluate_model( attention, loader, device, class_names, title="Attention CNN Normalized Confusion Matrix", save_path="Results/confusion_attention.png" )