import torch
import os
from torchvision import transforms
from PIL import Image

from model import EmotionClassificationNN

def predict_image(image_path, model_path):

    class_names = [
        "angry",
        "disgust",
        "fear",
        "happy",
        "neutral",
        "sad",
        "surprised"
    ]

    device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )

    # Load model
    model = EmotionClassificationNN( num_classes=len(class_names) )
    model.load_state_dict( torch.load(model_path, map_location=device) )
    model.to( device )
    model.eval()

    # Same transforms as training
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # Load and preprocess image
    image = Image.open( image_path ).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # add batch dimension
    image = image.to(device)

    # Inference
    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)

    predicted_class = class_names[pred.item()]
    return predicted_class

# Test out all images in the Images folder
if __name__ == "__main__":

    images_dir = "Images"
    model_path = "Models/best_emotion_model.pth"

    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

    for filename in os.listdir(images_dir):
        
        if filename.lower().endswith(valid_exts):
            image_path = os.path.join(images_dir, filename)
            prediction = predict_image(image_path, model_path)
            print(f"{filename}: {prediction}")