import torch 
from multi_digit_cnn import MultiDigitCNN  # Import the CNN model
from PIL import Image
import torchvision.transforms as transforms

MODEL_PATH = "multi_digit_cnn_svhn.pth"
model = MultiDigitCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cuda")))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((1024, 1024)),  # Match model input size
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize (adjust as needed)
])

image_pth = "timestamps/timestamp_1.png"

image = Image.open(image_pth)
input_tensor = transform(image).unsqueeze(0)



with torch.no_grad():
    output=model(input_tensor)

# Get prediction
predicted_class = torch.argmax(output, dim=1).item()
print(f"Predicted class: {predicted_class}")

