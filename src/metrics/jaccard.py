# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 10:08:26 2024

@author: anbre
"""

  
import os
import glob
import cv2
import numpy as np
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt
import seaborn as sns
 
# Define the paths to your masks and predictions
mask_folder_path = "BME6938_Project_Dataset/temp_test/masks"
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
        pred = cv2.normalize(pred, None, alpha=0, beta=5, norm_type=cv2.NORM_MINMAX)
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
 
# Initialize dictionaries to store results
jaccard_indices = {}
jaccard_distances = {}

# Loop through each class (pixel value from 0 to 4)
for label in range(5):
    # Create binary masks for the current label
    y_true_binary = (masks == label).astype(int)
    y_pred_binary = (predictions == label).astype(int)
    
    # Calculate the Jaccard index for the current label
    jaccard = jaccard_score(y_true_binary, y_pred_binary)
    jaccard_distance = 1 - jaccard  # Jaccard distance
    
    # Store the result
    jaccard_indices[label] = jaccard
    jaccard_distances[label] = jaccard_distance

# Output the results
print("Jaccard Indices per pixel value:", jaccard_indices)
print("Jaccard Distances per pixel value:", jaccard_distances)