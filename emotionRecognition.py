# Imports
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
import torchvision
import torch.nn.functional as F
import os
import pandas as pd
from PIL import Image
from torch.utils.data import (
    Dataset,
    DataLoader,
)  # Gives easier dataset managment and creates mini batches
from customDataset import EmotionDataset

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Convolutional Neural Network
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, in_channel = 1, num_classes = 10):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1)) #keeps size
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride =(2,2)) #cuts size in half
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1)) #keeps size
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1)) #keeps size
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # keeps size
        self.fc1 = nn.Linear(64*12*12, num_classes)

    def forward(self, x):
        x = F.relu((self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

def save_model(state, filename = "saved_model.pth.tar"):
    print("Saving Model")
    torch.save(state, filename)

def load_model(checkpoint):
    print("Loading Model")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# Hyperparameters
in_channel = 1
num_classes = 7
learning_rate = 0.01
batch_size = 64
num_epochs = 11

# Load Data
data_set = EmotionDataset(root_dir = "CK", transform = transforms.ToTensor(), train=True)
train_set, test_set = torch.utils.data.random_split(data_set, [775, 70])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=0)

#init network
model = ConvolutionalNeuralNetwork().to(device)
#model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=False)

#loss - cost function
criterion = nn.CrossEntropyLoss()

#learning algorithm, like gradient descent
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

def train_model():
    print("Training Model")
    for epoch in range(num_epochs):
        losses = []
        for batch_idx, (data, targets) in enumerate(train_loader):
            # get data in cpu
            data = data.to(device=device)
            targets = targets.to(device=device)

            # gets correct shape
            # data = data.reshape(data.shape[0], -1) #for NN

            # forward
            scores = model(data)
            loss = criterion(scores, targets)

            losses.append(loss.item())

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

        print(f"Cost at epoch {epoch} is {sum(losses) / len(losses)}")

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        # if loader.dataset.train:
        #     print("Checking accuracy on training data")
        # else:
        #     print("Checking accuracy on test data")
        for x, y in loader:
            x = x.to(device = device)
            y = y.to(device=device)
            #x = x.reshape(x.shape[0], -1) #for NN

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

    model.train()

if __name__ == "__main__":
    #torch.set_num_threads(10)
    #training model
    train_model()

    saved_model = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    save_model(saved_model)

    #check accuracy
    check_accuracy(train_loader, model)
    check_accuracy(test_loader, model)
