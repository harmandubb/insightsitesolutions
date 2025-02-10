import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MultiDigitCNN(nn.Module):
    def __init__(self, num_classes=18, dropout_rate=0.5):
        super(MultiDigitCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 48, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(48)
        self.pool1 = nn.MaxPool2d(2, 2, padding=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        
        self.conv2 = nn.Conv2d(48, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2, padding=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2, padding=1)
        self.dropout3 = nn.Dropout(p=dropout_rate)
        
        self.conv4 = nn.Conv2d(128, 160, kernel_size=5, stride=1, padding=2)
        self.bn4 = nn.BatchNorm2d(160)
        self.pool4 = nn.MaxPool2d(2, 2, padding=1)
        self.dropout4 = nn.Dropout(p=dropout_rate)
        
        self.conv5 = nn.Conv2d(160, 192, kernel_size=5, stride=2, padding=2)
        self.bn5 = nn.BatchNorm2d(192)
        self.pool5 = nn.MaxPool2d(2, 2, padding=1)
        self.dropout5 = nn.Dropout(p=dropout_rate)
        
        self.conv6 = nn.Conv2d(192, 192, kernel_size=5, stride=1, padding=2)
        self.bn6 = nn.BatchNorm2d(192)
        self.pool6 = nn.MaxPool2d(2, 2, padding=1)
        self.dropout6 = nn.Dropout(p=dropout_rate)
        
        self.conv7 = nn.Conv2d(192, 192, kernel_size=5, stride=2, padding=2)
        self.bn7 = nn.BatchNorm2d(192)
        self.pool7 = nn.MaxPool2d(2, 2, padding=1)
        self.dropout7 = nn.Dropout(p=dropout_rate)
        
        self.conv8 = nn.Conv2d(192, 192, kernel_size=5, stride=1, padding=2)
        self.bn8 = nn.BatchNorm2d(192)
        self.pool8 = nn.MaxPool2d(2, 2, padding=1)
        self.dropout8 = nn.Dropout(p=dropout_rate)

        self.flatten_size = self._get_flatten_size()
        
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, num_classes)  # Output layer for classification

    def forward(self, x):
        x = self.dropout1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        x = self.dropout3(self.pool3(F.relu(self.bn3(self.conv3(x)))))
        x = self.dropout4(self.pool4(F.relu(self.bn4(self.conv4(x)))))
        x = self.dropout5(self.pool5(F.relu(self.bn5(self.conv5(x)))))
        x = self.dropout6(self.pool6(F.relu(self.bn6(self.conv6(x)))))
        x = self.dropout7(self.pool7(F.relu(self.bn7(self.conv7(x)))))
        x = self.dropout8(self.pool8(F.relu(self.bn8(self.conv8(x)))))
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    def _get_flatten_size(self):
        """Helper function to determine the correct flatten size dynamically."""
        with torch.no_grad():
            x = torch.zeros(1, 1, 1024, 1024)  # Simulated input image
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = self.pool3(F.relu(self.bn3(self.conv3(x))))
            x = self.pool4(F.relu(self.bn4(self.conv4(x))))
            x = self.pool5(F.relu(self.bn5(self.conv5(x))))
            x = self.pool6(F.relu(self.bn6(self.conv6(x))))
            x = self.pool7(F.relu(self.bn7(self.conv7(x))))
            x = self.pool8(F.relu(self.bn8(self.conv8(x))))
            return x.numel()

# Model instantiation
num_classes = 18  # According to the paper
dropout_rate = 0.5  # Default value, can be adjusted
model = MultiDigitCNN(num_classes, dropout_rate)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
