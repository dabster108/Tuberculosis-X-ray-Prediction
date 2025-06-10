from PIL import Image
import torchvision.transforms as transforms
import torch

# Preprocessing pipeline (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_image(image: Image.Image, model, device):
    input_tensor = transform(image).unsqueeze(0).to(device)
    outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 1)
    classes = ['Normal', 'Tuberculosis']
    return classes[predicted.item()]
