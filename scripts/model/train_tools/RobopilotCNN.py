import torch.nn as nn

class RobopilotCNN(nn.Module):
    def __init__(self, output_size=4, dropout_rate=0.3):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),                          
            nn.Dropout(dropout_rate),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, output_size),
            nn.Sigmoid()                           
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
