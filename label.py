import os
import pandas as pd

# Define the folder path
folder_path = "C:\\Users\\nikhi\\Desktop\\emotion-research-master\\data"

# Initialize empty lists to store file paths and labels
paths = []
labels = []

# Define the label names
label_names = [
    "anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"
]

# Loop through each label folder
for label in label_names:
    label_folder = os.path.join(folder_path, label)
    
    # Check if the label folder exists
    if os.path.exists(label_folder):
        # Get a list of all files in the label folder
        files = os.listdir(label_folder)
        
        # Append the paths and labels to the respective lists
        for file in files:
            # Replace backslashes with forward slashes in the file path
            file_path = os.path.join(label, file).replace("\\", "/")
            paths.append(file_path)
            labels.append(label)

# Create a DataFrame from the lists
data = {"pth": paths, "label": labels}
df = pd.DataFrame(data)

# Define the CSV file path
csv_file = "labels.csv"

# Check if the CSV file already exists
if os.path.exists(csv_file):
    # If it exists, append the data to the existing file without writing headers
    df.to_csv(csv_file, mode='a', index=False, header=False)
    print(f"Data appended to '{csv_file}' successfully.")
else:
    # If it doesn't exist, create a new file with headers
    df.to_csv(csv_file, index=False)
    print(f"CSV file '{csv_file}' created successfully.")

# Read the CSV file
df = pd.read_csv(csv_file)

# Replace forward slashes with backslashes in the 'pth' column
df['pth'] = df['pth'].str.replace('\\', '/')

# Define the destination folder path
destination_folder = "C:\\Users\\nikhi\\Desktop\\emotion-research-master\\data"

# Define the destination file path
destination_file = os.path.join(destination_folder, csv_file)

# Check if the destination folder exists, and if not, create it
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Save the modified DataFrame to the destination folder
df.to_csv(destination_file, index=False)

print(f"CSV file saved to '{destination_file}'.")
