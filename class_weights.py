import os

# Directory containing subfolders for each emotion
data_dir = "data"

# Initialize a dictionary to store the class samples
class_samples = {}

# Iterate through subfolders
for emotion in os.listdir(data_dir):
    emotion_dir = os.path.join(data_dir, emotion)
    if os.path.isdir(emotion_dir):
        # Count the number of files (images) in the subfolder
        num_images = len([f for f in os.listdir(emotion_dir) if os.path.isfile(os.path.join(emotion_dir, f))])
        class_samples[emotion] = num_images

# Calculate the total number of samples in the dataset
total_samples = sum(class_samples.values())

# Calculate class weights using Method 3
class_weights = {}
for class_name, count in class_samples.items():
    class_weights[class_name] = 1 - (count / total_samples)

# Print class weights
print("Class Weights:")
for class_name, weight in class_weights.items():
    print(f"{class_name}: {weight:.4f}")
