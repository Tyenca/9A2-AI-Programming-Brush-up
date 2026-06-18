#Import packages
from utils import normalize_dataset, get_datasets, prepare_sklearn_data
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch.nn as nn
from dlordinal.metrics import amae, mmae, accuracy_off1
from sklearn.svm import LinearSVC

#Calculate mean and std values
mean, std = normalize_dataset()
#Finish normalizing the data
train_data, val_data, test_data = get_datasets(mean=mean, std=std)
#Flatten the images, because sklearn models expects 1D feature vectors, not 2D images
#Create new flat datasets with normalized pixel values
X_train, y_train = prepare_sklearn_data(train_data)
X_val,   y_val   = prepare_sklearn_data(val_data)
X_test,  y_test  = prepare_sklearn_data(test_data)


#Train a linear SVM model 
C_values = [0.001, 0.01, 0.1, 1.0, 10.0] #Hyperparameter tuning values
best_C, best_svc_acc = None, -1.0  # initialize variables to assure first loop will score the lowest c value. next loop will always update because it will score higher than -1

#Find best performing Regularization parameter C for linear SVM model
for C in C_values:
    # Training model with various c values and saving the best
    svc = LinearSVC(C=C, max_iter=7000, random_state=42, class_weight='balanced') # class_weight='balanced' to handle class imbalance in the dataset
    svc.fit(X_train, y_train)
    acc = accuracy_score(y_val, svc.predict(X_val))
    if acc > best_svc_acc:
        best_svc_acc, best_C = acc, C

#Prints the best performing C
print(f"Best C: {best_C}")

# Retrain with best C
svc_model = LinearSVC(C=best_C, max_iter=7000, random_state=42, class_weight='balanced') # class_weight='balanced' to handle class imbalance in the dataset
svc_model.fit(X_train, y_train)

#validation
svc_val_preds = svc_model.predict(X_val) 
svc_val_acc = accuracy_score(y_val, svc_val_preds)
print("VALIDATION")
print(f"SVC val accuracy: {svc_val_acc:.4f}")
print(f"One-off accuracy: {accuracy_off1(y_val, svc_val_preds):.4f}")
print(f"Average mean absolute error (AMAE): {amae(y_val, svc_val_preds):.4f}")
print(f"Maximum mean absolute error (MMAE): {mmae(y_val, svc_val_preds):.4f}") 
print(f"Quadratic weighted kappa: {cohen_kappa_score(y_val, svc_val_preds, weights='quadratic'):.4f}\n")

cm = confusion_matrix(y_val, svc_val_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1,2,3,4])
disp.plot()
plt.title("SVC — validation set")
plt.savefig(f"svc_val_confusion_matrix.png")
plt.close()

#testing
svc_test_preds = svc_model.predict(X_test) 
svc_test_acc = accuracy_score(y_test, svc_test_preds)
print("TEST")
print(f"SVC test accuracy: {svc_test_acc:.4f}")
print(f"One-off accuracy: {accuracy_off1(y_test, svc_test_preds):.4f}")
print(f"Average mean absolute error (AMAE): {amae(y_test, svc_test_preds):.4f}")
print(f"Maximum mean absolute error (MMAE): {mmae(y_test, svc_test_preds):.4f}") 
print(f"Quadratic weighted kappa: {cohen_kappa_score(y_test, svc_test_preds, weights='quadratic'):.4f}\n")

cm = confusion_matrix(y_test, svc_test_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1,2,3,4])
disp.plot()
plt.title("SVC — test set")
plt.savefig(f"svc_test_confusion_matrix.png")
plt.close()