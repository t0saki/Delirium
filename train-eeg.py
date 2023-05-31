# Import libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np

from models.CLAM import RNNAttentionModel, RNNModel

# Set the seed for reproducibility
torch.manual_seed(42)


# Define the dataset class
class EEGDataset(Dataset):
    def __init__(self, data_file, label, time_len):
        # Load the data from the csv file
        self.data = pd.read_csv(data_file).values
        # Assign the label to the data
        self.label = label
        # Assign the time length to the data
        self.time_len = time_len

    def __len__(self):
        # Return the number of samples in the data divided by the time length
        return len(self.data) // self.time_len

    def __getitem__(self, index):
        # Return the sample and the label at the given index
        # The sample is a segment of the data with the given time length
        return self.data[index * self.time_len: (index + 1) * self.time_len], self.label


# Define the hyperparameters
input_size = 4  # The number of input channels
hid_size = 96  # The hidden size of the RNN layer
rnn_type = 'lstm'  # The type of the RNN layer
bidirectional = True  # Whether to use bidirectional RNN or not
n_classes = 2  # The number of output classes
kernel_size = 5  # The kernel size of the convolution layer
batch_size = 128  # The batch size for training and validation
num_epochs = 100  # The number of epochs for training
learning_rate = 0.00001  # The learning rate for the optimizer
# The device to use for training and inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the time length for each sample
time_len = 2048

# Create the dataset objects for each class
data1 = EEGDataset('datasets/data-nd.csv', 0., time_len)  # The data of class 0
data2 = EEGDataset('datasets/data-d.csv', 1., time_len)  # The data of class 1

# Concatenate the datasets into one
data = torch.utils.data.ConcatDataset([data1, data2])

# Split the dataset into training and validation sets with a ratio of 0.8:0.2
train_size = int(0.8 * len(data))
val_size = len(data) - train_size
train_data, val_data = torch.utils.data.random_split(
    data, [train_size, val_size])

# Create the data loaders for training and validation sets
# Use collate_fn to stack the samples and labels into batches
# Convert the numpy arrays to tensors using torch.from_numpy
# Transpose the samples using numpy.transpose
# Convert the tensors to float using torch.float
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                          collate_fn=lambda x: [torch.stack([torch.from_numpy(np.transpose(s[0])).float() for s in x]), torch.tensor([s[1] for s in x])])
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda x: [torch.stack([torch.from_numpy(np.transpose(s[0])).float() for s in x]), torch.tensor([s[1] for s in x])])


# Create the model object and move it to the device
model = RNNAttentionModel(input_size=input_size, hid_size=hid_size, rnn_type=rnn_type,
                          bidirectional=bidirectional, n_classes=n_classes, kernel_size=kernel_size).to(device)


# model = RNNModel(input_size=input_size, hid_size=hid_size, rnn_type=rnn_type,
#                  bidirectional=bidirectional, n_classes=n_classes, kernel_size=kernel_size).to(device)

# Create the loss function and the optimizer
# Give data2 a higher weight
pos_weight = torch.tensor([len(data) / len(data2)]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
# Use Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

model = torch.compile(model)

# Define a function to compute the accuracy of the model on a given data loader


def accuracy(loader):
    # Set the model to evaluation mode
    model.eval()
    # Initialize the number of correct predictions and the total number of samples
    correct = 0
    total = 0
    # Initialize the number of true positives, true negatives, false positives, and false negatives
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    # Iterate through the batches in the loader
    with torch.no_grad():  # No need to compute gradients for validation
        for x, y in loader:
            # Move the inputs and labels to the device
            x = x.to(device)
            y = y.to(device)
            # Forward pass the inputs through the model and get the outputs
            outputs = model(x)
            # Get the predicted labels by taking the argmax of the outputs
            preds = outputs > 0.5
            # Update the total number of samples
            total += y.size(0)
            # Update the number of correct predictions
            correct += (preds == y).sum().item()
            # Update the number of true positives, true negatives, false positives, and false negatives
            tp += ((preds == 1) & (y == 1)).sum().item()
            tn += ((preds == 0) & (y == 0)).sum().item()
            fp += ((preds == 1) & (y == 0)).sum().item()
            fn += ((preds == 0) & (y == 1)).sum().item()
    # Compute the sensitivity and specificity
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    # Return the accuracy, sensitivity, and specificity
    return correct / total, sensitivity, specificity


# Train the model for a given number of epochs
for epoch in range(num_epochs):
    # Set the model to training mode
    model.train()
    # Initialize the running loss and accuracy for this epoch
    running_loss = 0.0
    running_acc = 0.0
    # Create a progress bar object to track the training progress
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    # Iterate through the batches in the train loader
    for x, y in pbar:
        # Move the inputs and labels to the device
        x = x.to(device)
        y = y.to(device)
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward pass the inputs through the model and get the outputs
        outputs = model(x)
        # Compute the loss using the criterion function
        loss = criterion(outputs, y)
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        # Get the predicted labels by taking the argmax of the outputs
        preds = outputs > 0.5
        # Update the running loss and accuracy for this epoch
        running_loss += loss.item()
        running_acc += (preds == y).sum().item() / y.size(0)
        # Update the progress bar with the average loss and accuracy for this epoch so far
        pbar.set_postfix({'loss': running_loss / (pbar.n + 1),
                         'acc': running_acc / (pbar.n + 1)})

    # Compute and print the validation accuracy for this epoch
    val_acc, sensitivity, specificity = accuracy(val_loader)
    print(
        f'Validation accuracy: {val_acc:.4f}, sensitivity: {sensitivity:.4f}, specificity: {specificity:.4f}')

    # Save the whole model ckeckpoint
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f'pths/clam-{epoch}.pth')
