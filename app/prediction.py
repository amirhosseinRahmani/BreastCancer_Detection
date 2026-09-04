from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from app.model import CLASS_NAMES


IMAGE_SIZE = 400

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def predict_image(model, device, image_path):
    """Predict one breast ultrasound image."""
    image_path = Path(image_path)

    with Image.open(image_path) as image:
        image = image.convert("L")
        tensor = TRANSFORM(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    predicted_index = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_index].item()

    probabilities_dict = {
        CLASS_NAMES[index]: float(probabilities[index])
        for index in range(len(CLASS_NAMES))
    }

    return CLASS_NAMES[predicted_index], confidence, probabilities_dict
