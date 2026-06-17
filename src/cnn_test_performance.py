# Evaluation on the test set 
#Import packages
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch.nn as nn
from dlordinal.metrics import amae, mmae, accuracy_off1
from neural_net import Net
from src.utils import normalize_dataset, get_dataloaders

#Normalize the dataset and get mean and std values
mean, std = normalize_dataset()
# Wrap in Pytorch dataloader objects to enable batching and shuffling
train_loader, val_loader, test_loader = get_dataloaders(mean=mean, std=std)

net = Net()
val_preds  = []
val_all_labels = []
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001) # define optimizer with best learning rate found in test_optimizer.py 

# Load the best saved model
best_model = net
best_model.load_state_dict(torch.load('saved_model.pth'))
best_model.eval()

all_preds  = []
all_labels = []

# turn off the gradient for evaluation, because we don't need to calculate gradients when evaluating the model, and it saves memory and makes it faster
with torch.no_grad():
    # loops through the test set in batches of 32 (set in get_dataloaders) 
    for test_inputs, test_labels in test_loader:
        #Squeezes the image to 1D
        test_labels = test_labels.squeeze().long()
        # Runs the images through the CNN. The output is a tensor of shape (batch_size, num_classes)
        test_outputs = best_model(test_inputs)
        # The predicted class is the one with the highest score. torch.max returns the maximum value and its index 
        _, predicted = torch.max(test_outputs, 1)
        # converting the bartch of predicted labels and true labels from PyTorch tensors to numpy arrays and adds them to the lists
        all_preds.extend(predicted.numpy())
        all_labels.extend(test_labels.numpy())
        # after this loop the lists all_preds and all_labels contain the predictions and true labels for all images in the test set

# Metrics
test_acc   = accuracy_score(all_labels, all_preds)
test_qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
test_off1   = accuracy_off1(all_labels, all_preds)
test_amae   = amae(all_labels, all_preds)
test_mmae   = mmae(all_labels, all_preds)

print(f"Test accuracy:            {test_acc:.4f}")
print(f"Quadratic weighted kappa: {test_qwk:.4f}")
print(f"One-off accuracy:         {test_off1:.4f}")
print(f"AMAE:                     {test_amae:.4f}")
print(f"MMAE:                     {test_mmae:.4f}")

# Plots confusion matrix showing predicted labels (x-axis) against true labels (y-axis)
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1,2,3,4])
disp.plot()
plt.title("CNN — test set")
plt.show()     