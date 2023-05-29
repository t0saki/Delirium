import torch
import torch.nn as nn
import torchvision.models as models

# Define the ResNet+BiLSTM model


class ResNetBiLSTM(nn.Module):
    def __init__(self, input_channels, input_length, hidden_size, num_layers, num_classes):
        super(ResNetBiLSTM, self).__init__()
        # Use ResNet18 as the feature extractor
        self.resnet = models.resnet18(pretrained=True)
        # Remove the last fully connected layer
        self.resnet.fc = nn.Identity()
        # Use a bidirectional LSTM to capture temporal dependencies
        self.lstm = nn.LSTM(input_channels*512, hidden_size,
                            num_layers, batch_first=True, bidirectional=True)
        # Use a linear layer to map the hidden state to the output classes
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        # x shape: (batch_size, input_channels, input_length)
        # Reshape x to (batch_size*input_channels, 3, 224, 224) and pass through ResNet
        batch_size = x.size(0)
        # Padding
        # if x.size(2) < 224:
        #     x = torch.cat((x, torch.zeros(
        #         batch_size, x.size(1), 224-x.size(2))), dim=2)
        x = x.view(-1, 3, 224, 224)
        x = self.resnet(x)
        # Reshape x to (batch_size, input_channels*512) and pass through LSTM
        x = x.view(batch_size, -1)
        x = x.unsqueeze(1)  # add a dummy sequence dimension
        x, _ = self.lstm(x)
        # Take the last hidden state of each sequence and pass through FC layer
        x = x[:, -1, :]
        x = self.fc(x)
        return x
