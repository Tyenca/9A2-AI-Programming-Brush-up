#Import packages
from utils import normalize_dataset, get_datasets, prepare_sklearn_data
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch.nn as nn
from dlordinal.metrics import amae, mmae, accuracy_off1
from sklearn.neighbors import KNeighborsClassifier

#Calculate mean and std values
mean, std = normalize_dataset()
#Finish normalizing the data
train_data, val_data, test_data = get_datasets(mean=mean, std=std)
#Flatten the images, because sklearn models expects 1D feature vectors, not 2D images
#Create new flat datasets with normalized pixel values
X_train, y_train = prepare_sklearn_data(train_data)
X_val,   y_val   = prepare_sklearn_data(val_data)
X_test,  y_test  = prepare_sklearn_data(test_data)

# Train a k-nearest neighbors model
k_values = [1, 3, 5, 7, 9, 11, 15] #hyperparameter tuning values
best_k, best_knn_acc = None, -1.0 # initialize variables to assure first loop will score the lowest k value. next loop will always update because it will score higher than -1

# Find best performing k for k-nearest neighbors model
for k in k_values:
    # Training model with various k values and saving the best
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance', random_state=42) # weights='balanced' to handle class imbalance in the dataset
    knn.fit(X_train, y_train)
    acc = accuracy_score(y_val, knn.predict(X_val))
    if acc > best_knn_acc:
        best_knn_acc, best_k = acc, k

#print the best K
print(f"Best k: {best_k}")

#  Retrain with best k
knn_model = KNeighborsClassifier(n_neighbors=best_k, weights='distance', random_state=42)
knn_model.fit(X_train, y_train)

#validation
knn_val_preds = knn_model.predict(X_val)
knn_val_acc = accuracy_score(y_val, knn_val_preds)
print("VALIDATION")
print(f"KNN val accuracy: {knn_val_acc:.4f}")
print(f"One-off accuracy: {accuracy_off1(y_val, knn_val_preds):.4f}")
print(f"Average mean absolute error (AMAE): {amae(y_val, knn_val_preds):.4f}")
print(f"Maximum mean absolute error (MMAE): {mmae(y_val, knn_val_preds):.4f}")
print(f"Quadratic weighted kappa: {cohen_kappa_score(y_val, knn_val_preds, weights='quadratic'):.4f}\n")

cm = confusion_matrix(y_val, knn_val_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1,2,3,4])
disp.plot()
plt.title("KNN — validation set")
plt.show()

#testing
knn_test_preds = knn_model.predict(X_test)
knn_test_acc = accuracy_score(y_test, knn_test_preds)
print("TEST")
print(f"KNN test accuracy: {knn_test_acc:.4f}")
print(f"One-off accuracy: {accuracy_off1(y_test, knn_test_preds):.4f}")
print(f"Average mean absolute error (AMAE): {amae(y_test, knn_test_preds):.4f}")
print(f"Maximum mean absolute error (MMAE): {mmae(y_test, knn_test_preds):.4f}")
print(f"Quadratic weighted kappa: {cohen_kappa_score(y_test, knn_test_preds, weights='quadratic'):.4f}\n")

cm = confusion_matrix(y_test, knn_test_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1,2,3,4])
disp.plot()
plt.title("KNN — test set")
plt.show()