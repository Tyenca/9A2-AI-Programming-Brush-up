#Import packages
from medmnist.dataset import RetinaMNIST
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dlordinal.metrics import amae, mmae, accuracy_off1
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
import torch

def normalize_dataset():
    train_data_raw = RetinaMNIST(split="train", download=True)
    # Select all images from the training set 
    all_images = np.stack([np.array(train_data_raw[i][0]) for i in range(len(train_data_raw))])

    channel_names = ['R', 'G', 'B']
    channel_means = []
    channel_stds  = []

    # Calculate mean and std for each channel
    for c, name in enumerate(channel_names):
        ch = all_images[:, :, :, c]
        channel_means.append(ch.mean())
        channel_stds.append(ch.std())

    mean = [channel_means[0]/255, channel_means[1]/255, channel_means[2]/255]
    std  = [channel_stds[0]/255, channel_stds[1]/255, channel_stds[2]/255]

    return mean, std

def get_dataloaders(mean, std, batch_size=32): #feed 32 images at a time
    # converts image to tensor and scales to [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    #Apply the transformation to the datasets
    train_data = RetinaMNIST(split='train', download=True, transform=transform)
    val_data   = RetinaMNIST(split='val',   download=True, transform=transform)
    test_data  = RetinaMNIST(split='test',  download=True, transform=transform)
    #Transform into dataloader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True) # shuffle the training data to help the model generalize better
    val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False) # no need to shuffle validation and test data, because we only evaluate the model on them, not train on them
    test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False)

    # Count the number of samples in each class in the training set, needed for criterion weights
    train_labels = np.array([train_data[i][1].item() for i in range(len(train_data))])
    train_class_counts = torch.tensor(np.bincount(train_labels), dtype=torch.float)

    return train_loader, val_loader, test_loader, train_class_counts


# return dataloaders for the RetinaMNIST dataset, given the mean and std for normalization and the batch size.
def get_datasets(mean, std, batch_size=32): #  feed 32 images at a time
    # converts image to tensor and scales to [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    #Apply the transformation to the datasets
    train_data = RetinaMNIST(split='train', download=True, transform=transform)
    val_data   = RetinaMNIST(split='val',   download=True, transform=transform)
    test_data  = RetinaMNIST(split='test',  download=True, transform=transform)

    return train_data, val_data, test_data

# Flatten the images, because sklearn models expects 1D feature vectors, not 2D images
def prepare_sklearn_data(dataset) :
    images = np.stack([np.array(dataset[i][0]) for i in range(len(dataset))])
    labels = np.array([dataset[i][1].item() for i in range(len(dataset))])
    # flatten each image from (3, 28, 28) to (2352)
    images = images.reshape(len(dataset), -1)
    # normalize to [0, 1]
    images = images / 255.0
    return images, labels