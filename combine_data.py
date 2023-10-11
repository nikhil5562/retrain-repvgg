import os
import random
import shutil

# Define the directories for the original and augmented datasets
original_data_dir = '/home/darsith/repvgg/emotion-research-master/data'
augmented_data_dir = '/home/darsith/repvgg/emotion-research-master/augmented_dataset'
output_data_dir = '/home/darsith/repvgg/emotion-research-master/new_dataset'


# Define the desired count for each class
desired_count = 5000

# List of classes
classes = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Create the output directory if it doesn't exist
os.makedirs(output_data_dir, exist_ok=True)

# Function to copy a specified number of random original images to the output directory
def copy_random_original_images(class_dir, num_to_copy, output_class_dir):
    original_images = os.listdir(class_dir)
    selected_images = random.sample(original_images, num_to_copy)
    for image in selected_images:
        src = os.path.join(class_dir, image)
        dest = os.path.join(output_class_dir, image)
        shutil.copy(src, dest)

# Iterate through each class
for class_name in classes:
    # Create a directory for the current class in the output dataset
    os.makedirs(os.path.join(output_data_dir, class_name), exist_ok=True)
    
    # Calculate how many samples are needed to reach 5000
    class_dir_augmented = os.path.join(augmented_data_dir, class_name)
    class_dir_original = os.path.join(original_data_dir, class_name)
    existing_count_augmented = len(os.listdir(class_dir_augmented))
    num_to_copy_original = max(0, desired_count - existing_count_augmented)
    
    # Copy augmented images to reach 5000 samples
    shutil.copytree(class_dir_augmented, os.path.join(output_data_dir, class_name), dirs_exist_ok=True)
    
    # If needed, copy random original images to reach 5000 samples
    if num_to_copy_original > 0:
        copy_random_original_images(class_dir_original, num_to_copy_original, os.path.join(output_data_dir, class_name))
