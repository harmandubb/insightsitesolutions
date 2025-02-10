import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

def preprocess_image(image_path, output_size=(1024, 1024)):
    """
    Pre-process an image as described in Section 3.2 of the paper:
    1. Crop manually (not included since it's manual)
    2. Resize using bi-cubic interpolation
    3. Apply Gaussian blur (5x5 kernel)
    4. Perform Adaptive Histogram Equalization (8x8 block size)
    """
    
    # Load image (grayscale)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize using bi-cubic interpolation
    image = cv2.resize(image, output_size, interpolation=cv2.INTER_CUBIC)
    
    # Apply Gaussian blur (5x5 kernel)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Apply Adaptive Histogram Equalization (8x8 block size)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    
    # Convert to PyTorch tensor and normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    image = Image.fromarray(image)  # Convert NumPy array to PIL Image
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image

# Example Usage:
# image_tensor = preprocess_image("example.jpg")