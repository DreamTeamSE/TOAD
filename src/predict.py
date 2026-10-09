# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 14:51:11 2024

@author: justinjoseph
"""

from keras.models import load_model
import os
import glob
import cv2
import numpy as np
from config import load_config

CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "configs", "baseline.yml"
)

cfg = load_config(CONFIG_PATH)

model_path = str(cfg["data_paths"]["weights_in"])
input_directory = cfg["data_paths"]["input_dir"]
output_directory = cfg["paths"]["output_dir"]

IMG_HEIGHT = cfg["data"]["img_height"]
IMG_WIDTH = cfg["data"]["img_width"]

def predict_images(input_directory, output_directory, model):
    original_pred_folder = os.path.join(output_directory, "Original_Predictions")
    visualized_pred_folder = os.path.join(output_directory, "Visualized_Predictions")

    os.makedirs(original_pred_folder, exist_ok=True)
    os.makedirs(visualized_pred_folder, exist_ok=True)

    for img_path in glob.glob(os.path.join(input_directory, "*.tif*")):
        img = cv2.imread(img_path, 1)
        if img is None:
            print(f"Skipping {img_path}: Could not read image.")
            continue
        if img.shape[2] != 3:
            print(f"Skipping {img_path}: Not an RGB image.")
            continue
        if img.shape[:2] != (IMG_HEIGHT, IMG_WIDTH):
            print(f"Resizing {img_path} to {IMG_WIDTH}x{IMG_HEIGHT}")
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

        img = (img / 255.0).astype(np.float32)
        img_input = np.expand_dims(img, 0)
        prediction = model.predict(img_input)
        predicted_img = np.argmax(prediction, axis=-1)[0, :, :]

        name = os.path.basename(img_path)
        cv2.imwrite(os.path.join(original_pred_folder, f"prediction_{name}"), predicted_img)

        pred_max = np.max(predicted_img)
        normalized_img = (
            (predicted_img / pred_max * 255) if pred_max > 0 else np.zeros_like(predicted_img, dtype=np.uint8)
        ).astype(np.uint8)
        cv2.imwrite(
            os.path.join(visualized_pred_folder, f"visualized_prediction_{name}"),
            normalized_img,
        )

if __name__ == "__main__":
    model = load_model(model_path)
    predict_images(input_directory, output_directory, model)
