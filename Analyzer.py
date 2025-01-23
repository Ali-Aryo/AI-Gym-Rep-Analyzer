import cv2
import mediapipe as mp
import joblib
import numpy as np
import pandas as pd

# Load trained model
#model = joblib.load('squat_classifier.pkl')
model = joblib.load('press_classifier.pkl')


# Load feature names from the training dataset
#column_names = pd.read_csv('processed_squat_data.csv').drop(columns=['label']).columns.tolist()
column_names = pd.read_csv('processed_press_data.csv').drop(columns=['label']).columns.tolist()


# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB for MediaPipe processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Default label
        label = "No Movement"

        if results.pose_landmarks:
            # Extract pose landmarks
            landmarks = results.pose_landmarks.landmark
            row = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]).flatten()

            # Convert to DataFrame with proper feature names
            row_df = pd.DataFrame([row], columns=column_names)

            # Predict movement using the trained model
            prediction = model.predict(row_df)

            # Display result on image
            label = "Squat" if prediction[0] == 1 else "No Movement"

            # Draw pose landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # Display the movement status on the screen
        cv2.putText(image, label, (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Squat Detection', image)

        # Exit the program when 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
