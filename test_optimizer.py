#Packages
from medmnist import RetinaMNIST
import pandas as pd
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader
import torch.nn as nn
import dlordinal
from dlordinal.metrics import amae, mmae, accuracy_off1
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

mean =[np.float32(-0.0039968463), np.float32(-0.0029370212), np.float32(-0.002002367)]
std = [np.float32(0.0019575264), np.float32(0.0028890595), np.float32(0.0034698865)]

# return dataloaders for the RetinaMNIST dataset, given the mean and std for normalization and the batch size.
def get_dataloaders(mean, std, batch_size=32): #  feed 32 images at a time
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    train_data = RetinaMNIST(split='train', download=True, transform=transform)
    val_data   = RetinaMNIST(split='val',   download=True, transform=transform)
    test_data  = RetinaMNIST(split='test',  download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True) # shuffle the training data to help the model generalize better
    val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False) # no need to shuffle validation and test data, because we only evaluate the model on them, not train on them
    test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# Wrap in Pytorch dataloader objects to enable batching and shuffling
train_loader, val_loader, test_loader = get_dataloaders(mean=mean, std=std)


# Define CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*4*4,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# the amount of passes through the train_loader
epochs = 20

# test different learning rates to find the best one for optimizer Adam
for lr in [0.01, 0.001, 0.0001, 0.00001]:
    # Reinitialise model and optimizer for each lr
    net = Net()
    criterion = nn.CrossEntropyLoss()                   # Define loss function 
    optimizer = optim.Adam(net.parameters(), lr=lr)     # Define optimizer with the current learning rate
    min_validation_loss = np.inf                        # initialises a loss variable for validation loss

    for e in range(epochs):
        #Instantiate running loss
        train_loss_epoch = 0.0
        net.train() #Puts model in training mode
        for i, data in enumerate(train_loader, 0):
            train_inputs, train_labels = data
            #Squeezes the image to 1D
            train_labels = train_labels.squeeze().long()
            #Clears the gradients
            optimizer.zero_grad()
            #Runs the forward pass
            train_outputs = net(train_inputs)
            #Finds the loss
            train_loss = criterion(train_outputs, train_labels)
            #Calculates gradients
            train_loss.backward()
            #Updates weights
            optimizer.step()
            #Sums all loss over the epoch and calculates an average loss per batch
            train_loss_epoch += train_loss.item()


        train_loss_epoch /= len(train_loader)

        net.eval() # Puts model in evaluation mode
        validation_loss = 0.0 # Instantiates a validation loss
        correct = 0 # Counting correct predictions
        total = 0
        with torch.no_grad():
            #For each batch, predict a label and calculate the loss
            for val_inputs, val_labels in val_loader:
                val_labels = val_labels.squeeze().long()
                val_outputs = net(val_inputs)
                val_loss = criterion(val_outputs, val_labels)
                #Sums all loss over the epoch and calculates an average loss per batch
                validation_loss += val_loss.item()
                _, predicted = torch.max(val_outputs, 1)
                # Counts the number of correct predictions and total predictions for accuracy calculation
                correct += (predicted == val_labels).sum().item()
                total += val_labels.size(0)
        
        validation_loss /= len(val_loader) # Average validation loss over the batches
        val_accuracy = correct / total # Calculate validation accuracy

        #Saves a model if its validation loss is lower than the currently lowest validation loss
        if min_validation_loss > validation_loss:
            min_validation_loss = validation_loss
            torch.save(net.state_dict(), f'saved_model_lr{lr}.pth')

    # prints the best validation loss and final validation accuracy for the current learning rate
    print(f"lr={lr} — best val loss: {min_validation_loss:.4f}, final val accuracy: {val_accuracy:.4f}")

