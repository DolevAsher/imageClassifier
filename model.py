import torch
import torch.nn as nn

class ResBlock(nn.Module):
    """A standard Residual Block to prevent vanishing gradients in deep networks."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # Skip Connection
        return self.relu(out)


class ModelArchitecture(nn.Module):
    def __init__(self, num_classes: int = 20):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1 -> Output: 32 x 112 x 112
            nn.Conv2d(3, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResBlock(32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2 -> Output: 64 x 56 x 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResBlock(64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3 -> Output: 128 x 28 x 28
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResBlock(128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4 -> Output: 256 x 14 x 14
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResBlock(256),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5 -> Output: 512 x 7 x 7
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            ResBlock(512),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Pools the 7x7 spatial dimension down to 1x1
            nn.AdaptiveAvgPool2d(1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024), # Added for stability in the fully connected layer
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args: x (batch of images)
        Returns: logits for 20 classes
        """
        x = self.features(x)
        x = self.classifier(x)
        return x