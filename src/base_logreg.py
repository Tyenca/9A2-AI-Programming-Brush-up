#Import packages
from utils import normalize_dataset, get_datasets, prepare_sklearn_data
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch.nn as nn
from dlordinal.metrics import amae, mmae, accuracy_off1
from sklearn.linear_model import LogisticRegression
import numpy as np

#Calculate mean and std values
mean, std = normalize_dataset()
#Finish normalizing the data
train_data, val_data, test_data = get_datasets(mean=mean, std=std)
#Flatten the images, because sklearn models expects 1D feature vectors, not 2D images
#Create new flat datasets with normalized pixel values
X_train, y_train = prepare_sklearn_data(train_data)
X_val,   y_val   = prepare_sklearn_data(val_data)
X_test,  y_test  = prepare_sklearn_data(test_data)

#Train a logistic regression model 
#For hyperparameter tuning, tried adding class_weight "balanced"
model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced') # class_weight='balanced' to handle class imbalance in the dataset
model.fit(X_train, y_train)

#validation
lr_val_preds = model.predict(X_val)
lr_val_acc = accuracy_score(y_val, lr_val_preds)
print("VALIDATION")
print(f"LR val accuracy: {lr_val_acc:.4f}")
print(f"One-off accuracy: {accuracy_off1(y_val, lr_val_preds):.4f}")
print(f"Average mean absolute error (AMAE): {amae(y_val, lr_val_preds):.4f}")
print(f"Maximum mean absolute error (MMAE): {mmae(y_val, lr_val_preds):.4f}")
print(f"Quadratic weighted kappa: {cohen_kappa_score(y_val, lr_val_preds, weights='quadratic'):.4f}\n")

cm = confusion_matrix(y_val, lr_val_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1,2,3,4])
disp.plot()
plt.title("Logistic Regression — validation set")
plt.show()

#testing
lr_test_preds = model.predict(X_test)
lr_test_acc = accuracy_score(y_test, lr_test_preds)
print("TEST")
print(f"LR test accuracy: {lr_test_acc:.4f}")
print(f"One-off accuracy: {accuracy_off1(y_test, lr_test_preds):.4f}")
print(f"Average mean absolute error (AMAE): {amae(y_test, lr_test_preds):.4f}")
print(f"Maximum mean absolute error (MMAE): {mmae(y_test, lr_test_preds):.4f}")
print(f"Quadratic weighted kappa: {cohen_kappa_score(y_test, lr_test_preds, weights='quadratic'):.4f}\n")

cm = confusion_matrix(y_test, lr_test_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1,2,3,4])
disp.plot()
plt.title("Logistic Regression — test set")
plt.show()