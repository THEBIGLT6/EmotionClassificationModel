import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from dataset import get_dataloaders
from model import EmotionClassificationNN, EmotionClassificationAttentionNN


# ----- Confusion Matrix Evaluation -----

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

# ----- t-SNE -----

@torch.no_grad()
def extract_features( model, loader, device ):
    model.eval()

    features = []
    labels = []

    for images, targets in loader:
        images = images.to( device )

        x = model.features( images )          
        x = torch.flatten( x, start_dim=1 )  

        features.append( x.cpu().numpy() )
        labels.append( targets.numpy() )

    features = np.concatenate( features, axis=0 )
    labels = np.concatenate( labels, axis=0 )

    return features, labels


def run_tsne( features, perplexity=30 ):
    tsne = TSNE( n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=42 )
    return tsne.fit_transform( features )


def plot_tsne( embeddings, labels, class_names, title, save_path ):
    plt.figure( figsize=(8, 6) )

    for idx, name in enumerate( class_names ):
        mask = labels == idx
        plt.scatter( embeddings[mask, 0], embeddings[mask, 1], label=name, s=15, alpha=0.7 )

    plt.legend( markerscale=2 )
    plt.title( title )
    plt.xlabel( "t-SNE Dim 1" )
    plt.ylabel( "t-SNE Dim 2" )
    plt.tight_layout()

    os.makedirs( os.path.dirname(save_path), exist_ok=True )
    plt.savefig( save_path, dpi=300 )
    plt.close()

    print( f"Saved {title} to {save_path}" )

# ---- Main -----

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

    baseline_feats, labels = extract_features( baseline, loader, device )
    baseline_tsne = run_tsne( baseline_feats )
    plot_tsne( baseline_tsne,labels, class_names, title="Baseline CNN t-SNE", save_path="Results/tsne_baseline.png" )

    # Attention Model
    attention = EmotionClassificationAttentionNN( num_classes=len(class_names) )
    attention.load_state_dict( torch.load( attention_model, map_location=device) )
    attention.to( device )

    evaluate_model( attention, loader, device, class_names, title="Attention CNN Normalized Confusion Matrix", save_path="Results/confusion_attention.png" )

    attention_feats, _ = extract_features( attention, loader, device )
    attention_tsne = run_tsne( attention_feats )
    plot_tsne( attention_tsne, labels, class_names, title="Attention CNN t-SNE", save_path="Results/tsne_attention.png" )