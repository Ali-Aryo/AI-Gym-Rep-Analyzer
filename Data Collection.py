import cv2
import mediapipe as mp
import csv
import time

# Initialize Mediapipe Holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# Open CSV file to store data
csv_file = open('{exercise_label}.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)

# Write CSV header
csv_writer.writerow(['label'] + [f'landmark_{i}_{axis}' for i in range(33) for axis in ['x', 'y', 'z', 'visibility']])

# Ask the user for label input
exercise_label = input("Enter label (squat / no_movement): ")

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    start_time = time.time()
    duration = 30  # Collect data for 30 seconds

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            row = [exercise_label]  # Add label first
            for lm in landmarks:
                row.extend([lm.x, lm.y, lm.z, lm.visibility])
            csv_writer.writerow(row)

        cv2.putText(image, f'Collecting: {exercise_label}', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Data Collection', image)

        # Stop after duration time
        if time.time() - start_time > duration or cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
csv_file.close()
cv2.destroyAllWindows()
