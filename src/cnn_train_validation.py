#Packages
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from neural_net import Net
from utils import normalize_dataset, get_dataloaders

#Normalize the dataset and get mean and std values
mean, std = normalize_dataset()
# Wrap in Pytorch dataloader objects to enable batching and shuffling
train_loader, val_loader, test_loader, train_class_counts = get_dataloaders(mean=mean, std=std)

# the amount of passes through the train_loader
epochs = 20

# test different learning rates to find the best one for optimizer Adam
for dropout in [0.1, 0.2, 0.3, 0.4, 0.5]:
    for lr in [0.01, 0.001, 0.0001, 0.00001]:
        # Reinitialise model and optimizer for each lr

        net = Net()

        # Define class weights to handle class imbalance in the dataset
        weights = 1.0 / train_class_counts                # Inverse of class counts to give more weight to underrepresented classes
        weights = weights / weights.sum()                 # normalise so they sum to 1
        criterion = nn.CrossEntropyLoss(weight=weights)   # Assigns weights to the loss function to handle class imbalance in the dataset

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
        print(f"dropout={dropout}, lr={lr} — best val loss: {min_validation_loss:.4f}, final val accuracy: {val_accuracy:.4f}\n")

