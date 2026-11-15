# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 11:18:11 2024

@author: anbre
"""

import os
import glob
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Define the path to your masks
mask_folder_path = "BME6938_Project_Dataset/temp_test/masks"
# Define path to original predictions, NOT visualized predictions
predictions_folder_path = "outputs/Original_Predictions"

# Initialize empty lists to store the masks and predictions
masks = []
predictions = []

# Loop over the mask files in the directory
# glob returns files in arbitrary order; pair masks and predictions by sorted name.
mask_paths = sorted(glob.glob(os.path.join(mask_folder_path, "*.tif")))
pred_paths = sorted(glob.glob(os.path.join(predictions_folder_path, "*.tif")))
if len(mask_paths) != len(pred_paths):
    raise ValueError(f"{len(mask_paths)} masks but {len(pred_paths)} predictions; cannot pair them")
for mask_path, pred_path in zip(mask_paths, pred_paths):
    # Load the mask and its corresponding prediction
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  
    mask = cv2.resize(mask, (256, 192), interpolation=cv2.INTER_NEAREST)
    mask_array = mask.flatten()   # Adjusting the ground truth values to match the predictions

    pred = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)  # Use IMREAD_UNCHANGED to keep the original depth
    if pred.dtype == np.float32:  # Check if it is a 32-bit image
        # Normalize to [0, 4] range
        pred = cv2.normalize(pred, None, alpha=0, beta=4, norm_type=cv2.NORM_MINMAX)
        # Convert to 8-bit
        pred = pred.astype(np.uint8)
    
    pred = cv2.resize(pred, (256, 192), interpolation=cv2.INTER_NEAREST)
    pred_array = pred.flatten()

    # Add the ground truth and prediction to their respective lists
    masks.extend(mask_array)
    predictions.extend(pred_array)
    
# Convert the lists to numpy arrays
masks = np.array(masks)
predictions = np.array(predictions)

# Convert to integer type
masks = masks.astype(int)
predictions = predictions.astype(int)
# Generate confusion matrix
conf_mat = confusion_matrix(masks, predictions)

# Normalize the confusion matrix to get the percentage
normalized_conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis] * 100
# Modified matrix to remove empty space in array

# Calculate Precision, Recall, F1 score, and Accuracy for each class
precision = precision_score(masks, predictions, average=None)
recall = recall_score(masks, predictions, average=None)
f1 = f1_score(masks, predictions, average=None)
accuracy = accuracy_score(masks, predictions)
class_accuracies = conf_mat.diagonal() / conf_mat.sum(axis=1)

# Print the confusion matrix and class-specific accuracies
print("\nAccuracy for each class:")
for i, acc in enumerate(class_accuracies):
    print(f"Class {i}: {acc:.2f}")

print("\nPrecision for each class:", precision)
print("Recall for each class:", recall)
print("F1 Score for each class:", f1)
print("Overall Accuracy:", accuracy)

