import os
import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from shutil import copyfile
from collections import defaultdict
from tqdm import tqdm

# Define the list of emotion labels
emotion_labels = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Define the path to the root directory where emotion-specific subdirectories are located
data_root = '/home/darsith/repvgg/emotion-research-master/data'

# Define the output directory to save the balanced dataset
output_dir = '/home/darsith/repvgg/emotion-research-master/agumented_dataset'

# Create the subfolders for each emotion category
for label in emotion_labels:
    label_dir = os.path.join(output_dir, label)
    os.makedirs(label_dir, exist_ok=True)

# Create a dictionary to store the desired number of samples per class
desired_samples_per_class = {
    'anger': 5000,
    'contempt': 5000,
    'disgust': 5000,
    'fear': 5000,
    'happy': 5000,
    'neutral': 5000,
    'sad': 5000,
    'surprise': 5000
}

# Calculate the maximum augmentation ratio based on the desired number
max_augmentation_ratio = 0.5  # You can adjust this value

# Data augmentation using Albumentations
augmentation_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.RandomRotate90(p=0.5),
    A.GaussNoise(p=0.2),
    A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
])

# Loop through each emotion class
for label in emotion_labels:
    # Get a list of image file paths for the current class
    class_dir = os.path.join(data_root, label)
    image_files = os.listdir(class_dir)

    # Calculate the desired number of samples for the class
    desired_samples = desired_samples_per_class[label]

    # Calculate the number of images to augment for this class
    num_images_to_augment = max(0, desired_samples - len(image_files))

    # Calculate the augmentation ratio
    augmentation_ratio = min(1.0, num_images_to_augment / len(image_files))

    # Limit the augmentation ratio to the predefined maximum
    augmentation_ratio = min(augmentation_ratio, max_augmentation_ratio)

    # Initialize tqdm progress bar
    progress_bar = tqdm(range(num_images_to_augment), desc=f"Augmenting {label} class")

    # Loop through the original images in the class
    for i in progress_bar:
        # Randomly select an image from the class
        selected_image_file = np.random.choice(image_files)
        image_path = os.path.join(class_dir, selected_image_file)

        # Load the selected original image
        original_image = cv2.imread(image_path)

        # Apply data augmentation based on the calculated augmentation ratio
        augmented = augmentation_transform(image=original_image)
        augmented_image = augmented['image']

        # Save the augmented image to the corresponding emotion subfolder
        augmented_image_path = os.path.join(output_dir, label, f"{label}_{i}.jpg")
        cv2.imwrite(augmented_image_path, augmented_image)

        # Update the tqdm progress bar
        progress_bar.set_postfix({"Augmented": i + 1})

# Print the counts of samples in each class
for label in emotion_labels:
    class_dir = os.path.join(output_dir, label)
    num_samples = len(os.listdir(class_dir))
    print(f"Class '{label}': {num_samples} samples")
