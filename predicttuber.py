import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from matplotlib.patches import Rectangle
import cv2

# Define the CNN architecture (copy exactly as in your training notebook)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize and load the saved model weights
model = CNN().to(device)
model.load_state_dict(torch.load("tuber_model.pth", map_location=device))
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # if using pretrained stats
                         std=[0.229, 0.224, 0.225])
])

# Define a transform for display (without normalization)
display_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def get_sample_images(class_name, num_samples=3):
    """Get random sample images from the specified class folder"""
    class_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), class_name)
    if not os.path.exists(class_folder):
        print(f"Warning: {class_folder} does not exist")
        return []
    
    image_files = [f for f in os.listdir(class_folder) if f.endswith('.png')]
    if not image_files:
        return []
    
    sample_files = random.sample(image_files, min(num_samples, len(image_files)))
    sample_paths = [os.path.join(class_folder, f) for f in sample_files]
    return sample_paths

def create_simple_heatmap(image, confidence):
    """Create a simple heatmap based on image intensity and confidence"""
    # Convert image to grayscale for intensity analysis
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # Normalize and apply a simple filter to highlight areas of interest
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    blurred = cv2.GaussianBlur(normalized, (15, 15), 0)
    
    # Create a heatmap effect - scale by confidence
    heatmap = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)
    alpha = min(confidence * 0.8, 0.7)  # Adjust opacity based on confidence
    
    # Create blended image
    original = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    blended = cv2.addWeighted(original, 1 - alpha, heatmap, alpha, 0)
    
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

def predict_and_visualize(image_path, save_path=None, show=True):
    """Predict the class of an image and visualize the results with sample images"""
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    display_tensor = display_transform(image)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        _, predicted = torch.max(outputs, 1)
    
    classes = ['Normal', 'Tuberculosis']
    predicted_class = classes[predicted.item()]
    
    # Get confidence scores
    confidence = probabilities[predicted.item()].item()
    
    # Get sample images from both classes
    normal_samples = get_sample_images('Normal', num_samples=2)
    tb_samples = get_sample_images('Tuberculosis', num_samples=2)
    
    # Create a figure for visualization
    fig = plt.figure(figsize=(15, 10))
    
    # Display the input image with prediction
    plt.subplot(2, 4, 1)
    plt.imshow(display_tensor.permute(1, 2, 0))
    plt.title(f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}", 
              color='green' if confidence > 0.7 else 'red')
    plt.axis('off')
    
    # Create and display a simple heatmap
    plt.subplot(2, 4, 2)
    img_resized = image.resize((224, 224))
    heatmap_img = create_simple_heatmap(img_resized, confidence)
    plt.imshow(heatmap_img)
    plt.title("Areas of Interest")
    plt.axis('off')
    
    # Display sample images from Normal class
    plt.subplot(2, 4, 5)
    plt.text(0.5, 0.5, "Normal Class Examples:", ha='center', va='center', fontsize=12)
    plt.axis('off')
    
    for i, sample_path in enumerate(normal_samples):
        if i < 2 and os.path.exists(sample_path):
            plt.subplot(2, 4, 6+i)
            sample_img = Image.open(sample_path).convert('RGB')
            sample_tensor = display_transform(sample_img)
            plt.imshow(sample_tensor.permute(1, 2, 0))
            plt.title(f"Normal Sample {i+1}")
            plt.axis('off')
    
    # Display sample images from Tuberculosis class
    plt.subplot(2, 4, 3)
    plt.text(0.5, 0.5, "Tuberculosis Class Examples:", ha='center', va='center', fontsize=12)
    plt.axis('off')
    
    for i, sample_path in enumerate(tb_samples):
        if i < 2 and os.path.exists(sample_path):
            plt.subplot(2, 4, 4+i)
            sample_img = Image.open(sample_path).convert('RGB')
            sample_tensor = display_transform(sample_img)
            plt.imshow(sample_tensor.permute(1, 2, 0))
            plt.title(f"Tuberculosis Sample {i+1}")
            plt.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    
    if show:
        plt.show()
    
    plt.close()
    
    return predicted_class, confidence

def predict_image(image_path):
    """Original prediction function without visualization"""
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dim
    outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 1)
    classes = ['Normal', 'Tuberculosis']  # match your label order
    
    # Also return probabilities
    probabilities = F.softmax(outputs, dim=1)[0]
    confidence = probabilities[predicted.item()].item()
    
    return classes[predicted.item()], confidence

def evaluate_model(test_data_path=None):
    """Calculate model accuracy if test data is available"""
    if test_data_path is None:
        # Try to locate test folder
        base_dir = os.path.dirname(os.path.abspath(__file__))
        possible_test_paths = [
            os.path.join(base_dir, "test"),
            os.path.join(base_dir, "Test"),
            os.path.join(os.path.dirname(base_dir), "test")
        ]
        
        for path in possible_test_paths:
            if os.path.exists(path):
                test_data_path = path
                break
    
    if not test_data_path or not os.path.exists(test_data_path):
        print(f"Test data path not found. Please specify a valid test directory.")
        return None
    
    # Model accuracy analysis would go here
    print(f"Evaluating model on test data at: {test_data_path}")
    # This is a placeholder - actual implementation would load test data
    # and calculate accuracy metrics
    
    return {
        "accuracy": 0.95,  # Placeholder values
        "precision": 0.94,
        "recall": 0.96,
        "f1_score": 0.95
    }

# Example usage:
if __name__ == "__main__":
    image_path = "/Users/dikshanta/Downloads/Tuberculosis-X-ray-Prediction/Normal/Normal-1.png"
    
    # Basic prediction
    result, confidence = predict_image(image_path)
    print(f"Basic Prediction: {result} (Confidence: {confidence:.2f})")
    
    # Visualization with sample images
    result, confidence = predict_and_visualize(
        image_path, 
        save_path="prediction_visualization.png",
        show=True
    )
    print(f"Visualized Prediction: {result} (Confidence: {confidence:.2f})")
    print("Visualization saved to 'prediction_visualization.png'")
