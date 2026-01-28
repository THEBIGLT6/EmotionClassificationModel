import torch
import os
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

from model import EmotionClassificationNN, EmotionClassificationAttentionNN
from gradcam import GradCAM


# Load in both models
def load_models(baseline_path, attention_path, num_classes, device):

    baseline_model = EmotionClassificationNN(num_classes=num_classes)
    attention_model = EmotionClassificationAttentionNN(num_classes=num_classes)

    baseline_model.load_state_dict(torch.load(baseline_path, map_location=device))
    attention_model.load_state_dict(torch.load(attention_path, map_location=device))

    baseline_model.to(device).eval()
    attention_model.to(device).eval()

    return baseline_model, attention_model


# Predict an image on both models
@torch.no_grad()
def predict_both(image_tensor, baseline_model, attention_model, class_names):

    baseline_out = baseline_model(image_tensor)
    attention_out = attention_model(image_tensor)

    baseline_pred = baseline_out.argmax(dim=1).item()
    attention_pred = attention_out.argmax(dim=1).item()

    return { "baseline": class_names[baseline_pred], "attention": class_names[attention_pred] }


# Image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# Helper function to load and preprocess image
def load_image( image_path, device ):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    return image


# Run Grad-CAM on a given image and model
def run_gradcam( image_tensor, model, target_layer ):
    gradcam = GradCAM( model, target_layer )
    cam = gradcam.generate( image_tensor )
    return cam


# Produce the grad-cam overlay on the image and save in it's own directory
def overlay_cam(image_path, cam, output_path):

    image = cv2.imread(image_path)
    image = cv2.resize(image, (cam.shape[1], cam.shape[0])) if False else image

    # Normalize CAM
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    # Resize CAM to image size
    cam = cv2.resize(cam, (image.shape[1], image.shape[0]))

    # Convert CAM to heatmap
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, overlay)


# Main functionailty 
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    images_dir = "Images"
    gradcam_output_dir = "GradCAM"
    baseline_model_path = "Models/best_baseline.pth"
    attention_model_path = "Models/best_attention.pth"

    valid_exts = ( ".jpg", ".jpeg", ".png", ".bmp", ".gif" )
    class_names = [ "angry", "disgust", "fear", "happy", "neutral", "sad", "surprised"]

    baseline_model, attention_model = load_models( baseline_model_path, attention_model_path, num_classes=len(class_names),device=device)

    for filename in os.listdir( images_dir ):

        if filename.lower().endswith( valid_exts ):
            image_path = os.path.join( images_dir, filename )

            image_tensor = load_image(image_path, device)
            preds = predict_both( image_tensor, baseline_model, attention_model, class_names )

            baseline_cam = run_gradcam( image_tensor, baseline_model, baseline_model.features[-1] )
            attention_cam = run_gradcam( image_tensor, attention_model, attention_model.features[-1] )

            overlay_cam( image_path, baseline_cam, f"{gradcam_output_dir}/baseline_{filename}" )
            overlay_cam( image_path, attention_cam, f"{gradcam_output_dir}/attention_{filename}" )

            print( f"{filename} | Baseline: {preds['baseline']} | Attention: {preds['attention']}" )