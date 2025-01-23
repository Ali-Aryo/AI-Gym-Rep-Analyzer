import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Prompt user for the exercise name
exercise_name = input("Enter the exercise name (e.g., squat, deadlift, bench_press): ").strip().lower()

# Construct file names based on the exercise
data_file = f'processed_{exercise_name}_data.csv'
model_file = f'{exercise_name}_classifier.pkl'

try:
    # Load the processed data
    df = pd.read_csv(data_file)
    print(f"Successfully loaded {data_file}")

    # Split data into features and labels
    X = df.drop(columns=['label'])
    y = df['label']

    # Encode labels dynamically based on the provided exercise
    y = y.map({'no_movement': 0, exercise_name: 1})

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained successfully for '{exercise_name}' with Accuracy: {accuracy:.2f}")

    # Save the trained model
    joblib.dump(model, model_file)
    print(f"Model saved as '{model_file}'")

except FileNotFoundError:
    print(f"Error: File '{data_file}' not found. Please ensure the correct file exists.")
except Exception as e:
    print(f"An error occurred: {e}")
