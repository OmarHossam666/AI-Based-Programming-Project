import torch
import torch.nn as nn

class BrainTumorCNN(nn.Module):
    """
    A simple CNN architecture for Brain Tumor MRI classification.
    Designed to produce 7200 features [8 * 30 * 30] before the classifier.
    """
    def __init__(self):
        super(BrainTumorCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 250 -> 125
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 125 -> 62
            
            nn.Conv2d(32, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((30, 30)) # Forces 30x30 spatial features
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 30 * 30, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
