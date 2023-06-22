import pandas as pd

# Read the "train.csv" file into a pandas DataFrame
data = pd.read_csv('train.csv')

# Select the columns to be hot encoded
columns_to_encode = ['Sex', 'Embarked']

# Perform hot encoding with binary encoding
encoded_data = pd.get_dummies(data, columns=columns_to_encode, drop_first=True)

# Drop the unnecessary columns
columns_to_drop = ['Name', 'Ticket', 'Cabin']
encoded_data = encoded_data.drop(columns_to_drop, axis=1)

# Handle missing values
encoded_data = encoded_data.fillna(0)  # Replace missing values with 0

# Convert encoded columns to binary (0s and 1s)
encoded_data = encoded_data.astype(int)

# Save the encoded data to a new CSV file named "encoded_train.csv"
encoded_data.to_csv('encoded_train.csv', index=False)
