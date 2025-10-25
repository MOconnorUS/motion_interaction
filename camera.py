"""
The camera file is reserved for all functions related to the camera recording, opencv, and mediapipe.

When you are programming in python be sure to use type hinting as it clears up confusion between co-developers
and readers. If you are not familiar with type hinting see the following example.

--- Type Hinting ---
To type hint you must provide the type of the input parameters of a function as well as the return type 
of the function.

Example:
def foo(x : int, s : str) -> None:

--- Documentation ---
For the first few lines within each function please use the triple quotes as seen in this comment to record 
documentation about each function.
"""
import cv2
import mediapipe as mp
import math
import ctypes

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# <<< 2. ADD A THRESHOLD VALUE FOR "CLOSENESS" >>>
# This is a normalized distance (0.0 to 1.0).
# You'll need to experiment with this value. Start with 0.05.
GESTURE_THRESHOLD = 0.05

# This will track if we are already in a "clicked" state
click_locked = False

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    image_height, image_width, _ = image.shape

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:

      # select first hand detected
      hand_landmarks = results.multi_hand_landmarks[0]

      lm4 = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
      lm10 = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
      distance = math.hypot(lm4.x - lm10.x, lm4.y - lm10.y)

      if distance < GESTURE_THRESHOLD and not click_locked:
        print("CLICKED")
        click_locked = True

      elif distance >= GESTURE_THRESHOLD:
        click_locked = False

      mp_drawing.draw_landmarks(
          image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
      
      
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
