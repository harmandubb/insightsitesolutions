import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from multi_digit_cnn import MultiDigitCNN  # Import the CNN model
from image_preprocessing import preprocess_image  # Import preprocessing function

def train_model(epochs, batch_size, learning_rate, save_path):
    # Check if CUDA is available, otherwise exit with an error message
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA device not found. This script requires a GPU to run.")
    
    # Load SVHN dataset with preprocessing pipeline
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((1024, 1024)),  # Resize as per the paper
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1,1]
    ])
    
    trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
    # testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

        
    # Get one batch of images and labels
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    show_images(images, labels, batch_size)


    for inputs, labels in train_loader: 
        print("Inputs shape:", inputs.shape)
        print("Labels shape:", labels.shape)
        print("Labels dtype:", labels.dtype)
        print("Labels:", labels)
        break
    
    # Model Initialization
    # num_classes = 18  # As described in the paper
    # model = MultiDigitCNN(num_classes)
    # model.to(device)
    
    # Loss Function and Optimizer
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Enable Mixed Precision Training for RTX 3060
    # scaler = torch.amp.GradScaler('cuda')


    # Training Loop
    # train_losses = []
    # test_losses = []
    # train_accuracies = []
    # test_accuracies = []
    
    # for epoch in range(epochs):
    #     model.train()
    #     running_loss = 0.0
    #     correct = 0
    #     total = 0

    #     # Progress bar for training
    #     loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
    #     for images, labels in train_loader:
    #         images, labels = images.to(device), labels.to(device)
    #         optimizer.zero_grad()

    #         with torch.amp.autocast('cuda'):
    #             outputs = model(images)
    #             loss = criterion(outputs, labels)

    #         # Backpropagation with AMP
    #         scaler.scale(loss).backward()
    #         scaler.step(optimizer)
    #         scaler.update()

    #         # loss.backward()
    #         # optimizer.step()
            
    #         running_loss += loss.item()
    #         _, predicted = torch.max(outputs, 1)
    #         correct += (predicted == labels).sum().item()
    #         total += labels.size(0)

    #         # Update progress bar with loss info
    #         loop.set_postfix(loss=loss.item())
        
    #     train_losses.append(running_loss / len(train_loader))
    #     train_accuracies.append(100 * correct / total)
        
    #     # Evaluate on test set
    #     model.eval()
    #     test_loss = 0.0
    #     correct = 0
    #     total = 0
    #     with torch.no_grad():
    #         for images, labels in test_loader:
    #             images, labels = images.to(device), labels.to(device)
    #             outputs = model(images)
    #             loss = criterion(outputs, labels)
    #             test_loss += loss.item()
    #             _, predicted = torch.max(outputs, 1)
    #             correct += (predicted == labels).sum().item()
    #             total += labels.size(0)
        
    #     test_losses.append(test_loss / len(test_loader))
    #     test_accuracies.append(100 * correct / total)
        
    #     print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.2f}%, Test Acc: {test_accuracies[-1]:.2f}%")
    
    # # Save Model
    # torch.save(model.state_dict(), save_path)
    # print(f"Model saved successfully at {save_path}.")
    
    # # Plot Loss and Accuracy
    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    # plt.plot(range(1, epochs + 1), test_losses, label="Test Loss")
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.title("Training & Testing Loss")
    # plt.legend()
    
    # plt.subplot(1, 2, 2)
    # plt.plot(range(1, epochs + 1), train_accuracies, label="Train Accuracy")
    # plt.plot(range(1, epochs + 1), test_accuracies, label="Test Accuracy")
    # plt.xlabel("Epochs")
    # plt.ylabel("Accuracy (%)")
    # plt.title("Training & Testing Accuracy")
    # plt.legend()
    
    # plt.show()

def show_images(images, labels, batch_size):
    fig, axes = plt.subplots(1, batch_size, figsize=(10, 2))
    for i in range(batch_size):
        ax = axes[i]
        img = images[i].squeeze(0)  # Remove channel dimension
        ax.imshow(img, cmap='gray')  # Display image in grayscale
        ax.set_title(f"Label: {labels[i].item()}")  # Display label
        ax.axis("off")  # Hide axes
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MultiDigitCNN on SVHN Dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--save_path", type=str, default="multi_digit_cnn_svhn.pth", help="Path to save the trained model")
    
    args = parser.parse_args()
    
    train_model(args.epochs, args.batch_size, args.lr, args.save_path)

