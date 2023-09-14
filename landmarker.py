import cv2
import mediapipe as mp

# Initialize MediaPipe Hand Landmark model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Start video stream
cap = cv2.VideoCapture(0)

# Loop indefinitely
while True:
    # Read frame from the video stream
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB and process it with MediaPipe Hand Landmark model
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame)

    # Draw landmarks on the frame if detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmark points as circles
            for id, landmark in enumerate(hand_landmarks.landmark):
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 7, (255, 0, 0), -1)

                # Display landmark ID number next to the landmark point
                cv2.putText(frame, str(id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Display the frame in a window
    cv2.imshow('Hand Landmarks', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()