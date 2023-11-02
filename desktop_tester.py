import cv2, os
import mediapipe as mp
import pandas as pd
import time
import numpy as np
import tensorflow as tf

# Initialize MediaPipe Hand Landmark model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Start video stream
cap = cv2.VideoCapture(0)

loaded_model = tf.keras.models.load_model("hand_model.h5")

labels = ["LEFT_ONE", "LEFT_TWO", "LEFT_THREE",
          "RIGHT_ONE", "RIGHT_TWO", "RIGHT_THREE"]

# Loop forever
start_time = time.time()
current_prediction = ""
while True:
    # Read frame from the video stream
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB and process it with MediaPipe Hand Landmark model
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)
    results = hands.process(frame)

    # Set up filler info
    landmark_ids = [str(i) for i in range(21)]
    coords = [0 for i in range(21)]
    data_row = dict(zip(landmark_ids, coords))

    cv2.putText(frame, current_prediction, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw landmarks on the frame if detected
    if results.multi_hand_landmarks:
        
        # Container of hand data
        data = {"LeftX": data_row.copy(), "RightX": data_row.copy(),
                "LeftY": data_row.copy(), "RightY": data_row.copy()}

        # Loop through each hand's landmarks
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            if hand_landmarks:
                # Get the handedness (left or right) for the current hand
                hand_label = handedness.classification[0].label
                color = (0, 0, 0)
                if hand_label == "Left":
                    color = (255, 0, 0)
                elif hand_label == "Right":
                    color = (0, 0, 255)

                # Visualize the landmarks
                for id, landmark in enumerate(hand_landmarks.landmark):
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 7, color, -1)

                    # Display landmark ID number next to the landmark point
                    cv2.putText(frame, str(id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                    # Save the normalized landmarks
                    data[hand_label+"X"][str(id)] = max(0, x / frame.shape[1])
                    data[hand_label+"Y"][str(id)] = max(0, y / frame.shape[0])

        if time.time() - start_time >= 0.5:
            # Convert the nested dict to a DataFrame
            dataframe = pd.DataFrame.from_dict({i: data[i] for i in data.keys()}, orient='index')
            dataframe.columns = dataframe.columns.astype(str)

            # Convert to tensor
            features = dataframe.iloc[:, :].values  # Exclude the last column as it's the label
            
            # Convert to NumPy arrays
            features_array = np.array(features)
            
            # Convert to TensorFlow tensors
            features_tensor = tf.convert_to_tensor(features_array, dtype=tf.float32)
            all_features = tf.concat([[features_tensor]], axis=0)

            predictions = loaded_model.predict(all_features)

            # Format the preductions
            predicted_indices = np.argmax(predictions, axis=-1)

            # Map the indices to class labels
            predicted_labels = [labels[index] for index in predicted_indices[0]]

            # 'predicted_labels' now contains the predicted class labels for each example in the batch
            if predicted_labels[-1].startswith("LEFT"):
                current_prediction = predicted_labels[0]
            else:
                current_prediction = predicted_labels[-1]
                
            start_time = time.time()

    # Display the frame in a window
    cv2.imshow('Hand Landmarks', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()