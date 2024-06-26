import torch.nn as nn
import torch
import torchvision
#git
from timeit import default_timer as timer

from tqdm import tqdm

class_names = ['fractured', 'not fractured']

class FractureModel(nn.Module):

    # Output shape should be 2 because of binary classification
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()

        # hyperparameters
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 0,
                      out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        print(x.shape)
        x = self.conv_block_2(x)
        print(x.shape)
        x = self.classifier(x)

        return x


torch.manual_seed(42)
device = torch.device("cpu")
model = FractureModel(input_shape=1,
                      hidden_units=2,
                      output_shape=len(class_names)).to(device)

loss = nn.BCELoss()
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.1)

