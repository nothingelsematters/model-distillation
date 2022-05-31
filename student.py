import torch.nn as nn
from torch.nn.functional import log_softmax

class CNN_Student(nn.Module):
    def __init__(self):
        super(CNN_Student, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=3),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layers = nn.Sequential(nn.Linear(1152, 10))

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        logits = self.fc_layers(x)
        y_pred = log_softmax(logits, dim=1)
        return y_pred, logits
