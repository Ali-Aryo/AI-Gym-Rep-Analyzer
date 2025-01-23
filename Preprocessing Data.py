import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv('press_data.csv')

# Remove any rows with missing values
df.dropna(inplace=True)

# Extract features and labels
X = df.drop(columns=['label'])
y = df['label']

# Normalize feature values
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Save the processed data
processed_df = pd.DataFrame(X_scaled, columns=X.columns)
processed_df['label'] = y
processed_df.to_csv('processed_data1.csv', index=False)

print("Data preprocessing complete.")
