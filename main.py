"""
The main file is reserved for bringing together all information from peripheral_interactions, camera, etc. 
This file can also contain any helper functions to get things to run but it should be kept clean and concise.

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
import time

from peripheral_interactions import get_screen_dimensions, move_cursor

cap = cv2.VideoCapture(0) # Example: capturing from the default camera
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def open_cap() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return("ERROR: COULD NOT OPEN  VIDEO SOURCE.")
    
    return cap


if __name__ == '__main__':
    cap = open_cap()
    
    # Get the frame width and height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    screen_width, screen_height = get_screen_dimensions()

    print(f"Frame Width: {frame_width}")
    print(f"Frame Height: {frame_height}")

    aspect_ratio_x = screen_width / frame_width
    aspect_ratio_y = screen_height / frame_height
    print(f'ASPECT RATIO X: {aspect_ratio_x}')
    print(f'ASPECT RATIO Y: {aspect_ratio_y}')

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
                lm = hand_landmarks.landmark

                lm8 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                lm5 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                
                if lm8.y < (lm5.y - 0.085):
                    x = (lm8.x * image_width) * aspect_ratio_x
                    y = (lm8.y * image_height) * aspect_ratio_y

                    bottom_ratio = float(y / screen_height)

                    y = y + (screen_height * (1 - (bottom_ratio + 0.045))) if bottom_ratio >= 0.67 else y
                    # print("CURSOR MOVING: ", x, y)

                    x = screen_width - x
                    print(f'POSITION ON SCREEN: {x}, {y}\n')
                    move_cursor(int(x), int(y))

                    # time.sleep(0.1)

    # Release the video capture object
    cap.release()


