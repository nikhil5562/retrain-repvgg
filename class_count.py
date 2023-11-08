import os

# Directory containing subfolders for each emotion
data_dir = "data"

# Initialize a dictionary to store the number of images for each class
class_image_counts = {}

# Iterate through subfolders
for emotion in os.listdir(data_dir):
    emotion_dir = os.path.join(data_dir, emotion)
    if os.path.isdir(emotion_dir):
        # Count the number of files (images) in the subfolder
        num_images = len([f for f in os.listdir(emotion_dir) if os.path.isfile(os.path.join(emotion_dir, f))])
        class_image_counts[emotion] = num_images

# Print the number of images in each class
print("Number of Images in Each Class:")
for class_name, count in class_image_counts.items():
    print(f"{class_name}: {count} images")
