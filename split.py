import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV file into a DataFrame
df = pd.read_csv("data/labels.csv")

# Split the DataFrame into training, testing, and validation sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Save the split datasets to separate CSV files
train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)
val_df.to_csv("data/val.csv", index=False)
