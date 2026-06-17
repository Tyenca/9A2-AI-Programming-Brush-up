#Import packages
from medmnist.dataset import RetinaMNIST
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dlordinal.metrics import amae, mmae, accuracy_off1
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay

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

    return train_loader, val_loader, test_loader


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
def prepare_sklearn_data(dataset):
    mean, std = normalize_dataset()
    get_datasets
    # flatten each image from (3, 28, 28) to (2352,)
    images = np.stack([dataset[i][0].numpy().reshape(-1) for i in range(len(dataset))])
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    return images, labels


#def print_results(true, pred):
#    acc = accuracy_score(true, pred)
#    print(f"LR val accuracy: {acc:.4f}")
#    print(f"One-off accuracy: {accuracy_off1(true, pred):.4f}")
#    print(f"Average mean absolute error (AMAE): {amae(true, pred):.4f}")
#    print(f"Maximum mean absolute error (MMAE): {mmae(true, pred):.4f}")
#    print(f"Quadratic weighted kappa: {cohen_kappa_score(true, pred, weights='quadratic'):.4f}\n")
