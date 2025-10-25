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
from typing import Any
import cv2
import mediapipe as mp
import math
import ctypes
import time
from peripheral_interactions import left_click, ctrl_tab, alt_tab_cycle, alt_tab_release, alt_tab_start

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# --- Tunables ---
# Normalized distances in [0,1] (mediaPipe image coordinates)
GESTURE_THRESHOLD: float = 0.05         # thumb–pinky/other distances (existing)
FINGERS_TOGETHER_THRESH: float = 0.06   # how close index/middle tips must be for scroll lock
EXTENSION_MARGIN: float = 0.02          # tip must be above PIP by this amount to be "extended"
ORIENT_ANGLE_TOL_DEG: float = 25.0      # tolerance for "straight up" or "straight down"
wrist_threshold = 0.25

# States
click_locked: bool = False       # pre-existing
aTab_locked: bool = False        # pre-existing
scroll_locked: bool = False
scroll_up: bool = False
scroll_down: bool = False
alt_key_down = False


# --- Timer/State Variables ---
ALT_TAB_TIMEOUT: float = 2.0
GESTURE_COOLDOWN: float = 0.5
alt_key_down: bool = False
last_gesture_time: float = 0
last_valid_gesture_time: float = 0

def norm_dist(a: Any, b: Any) -> float:
    """Return normalized euclidean distance between two landmarks a,b (x,y)."""
    return math.hypot(a.x - b.x, a.y - b.y)

def is_extended(hand_lm: Any, tip_idx: int, pip_idx: int) -> bool:
    """
    A finger is 'extended' if its tip is above (smaller y) its PIP by EXTENSION_MARGIN.
    Image y grows downward, so "above" means a smaller y value.
    """
    tip = hand_lm.landmark[tip_idx]
    pip = hand_lm.landmark[pip_idx]
    return (pip.y - tip.y) > EXTENSION_MARGIN

def angle_to_vertical_deg(hand_lm: Any, pip_idx: int, tip_idx: int) -> float:
    """
    Returns angle (degrees) between finger vector (PIP->TIP) and vertical-up.
      0°   = perfectly up
      180° = perfectly down
      ~90° = sideways
    """
    tip = hand_lm.landmark[tip_idx]
    pip = hand_lm.landmark[pip_idx]
    vx = tip.x - pip.x
    vy = tip.y - pip.y
    mag = math.hypot(vx, vy) + 1e-9
    cos_theta = max(-1.0, min(1.0, (-vy) / mag))  # compare to up-vector (0,-1)
    return math.degrees(math.acos(cos_theta))

# # This will track if we are already in a "ctrl-tab" state
# cTab_locked = False

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
    
    current_time: float = time.time()
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
        print("CURSOR MOVING: ", lm8.x, lm8.y)

      lm4 = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
      lm10 = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
      click_distance = math.hypot(lm4.x - lm10.x, lm4.y - lm10.y)

      if click_distance < GESTURE_THRESHOLD and not click_locked:
        lm16 = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        lm12 = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        lm20 = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
        lm0 = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        d1 = math.hypot(lm16.x - lm0.x, lm16.y - lm0.y)
        d2 = math.hypot(lm12.x - lm0.x, lm12.y - lm0.y)
        d3 = math.hypot(lm20.x - lm0.x, lm20.y - lm0.y)
        if d3 < wrist_threshold and d2 < wrist_threshold and d1 < wrist_threshold:
          left_click(0,0)
          print("CLICKED")
          click_locked = True

      elif click_distance >= GESTURE_THRESHOLD:
        click_locked = False

# ---------- ALT-TAB LOGIC (Timer-Based) ----------
      lm12 = lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
      aTab_distance = norm_dist(lm4, lm12)

      # Check if gesture is made AND cooldown has passed
      if aTab_distance < GESTURE_THRESHOLD and (current_time - last_valid_gesture_time) > GESTURE_COOLDOWN:
          
          if not alt_key_down:
              # FIRST press: Call your start function
              alt_tab_start()
              alt_key_down = True
              print("ALT-TAB START (Alt Down)")
          else:
              # 'Alt' is already down: Call your cycle function
              alt_tab_cycle()
              print("...NEXT TAB")
          
          # Reset both timers
          last_gesture_time = current_time
          last_valid_gesture_time = current_time
      
      # --- This block runs EVERY frame to check for a timeout ---
      # If 'alt' is down AND it's been too long since the last gesture
      if alt_key_down and (current_time - last_gesture_time) > ALT_TAB_TIMEOUT:
          # Call your release function
          alt_tab_release()
          alt_key_down = False
          print("ALT-TAB TIMEOUT (Alt Up)")

      lm20 = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
      cTab_distance = math.hypot(lm4.x - lm20.x, lm4.y - lm20.y)

      if cTab_distance < GESTURE_THRESHOLD and not cTab_locked:
        ctrl_tab()
        print("CTRL-TAB")
        cTab_locked = True

      elif cTab_distance >= GESTURE_THRESHOLD:
        cTab_locked = False

      # ---------- SCROLL GESTURES ----------
      # Index/middle/ring/pinky tips & PIPs
      idx_tip = mp_hands.HandLandmark.INDEX_FINGER_TIP
      idx_pip = mp_hands.HandLandmark.INDEX_FINGER_PIP
      mid_tip = mp_hands.HandLandmark.MIDDLE_FINGER_TIP
      mid_pip = mp_hands.HandLandmark.MIDDLE_FINGER_PIP
      rng_tip = mp_hands.HandLandmark.RING_FINGER_TIP
      rng_pip = mp_hands.HandLandmark.RING_FINGER_PIP
      pnk_tip = mp_hands.HandLandmark.PINKY_TIP
      pnk_pip = mp_hands.HandLandmark.PINKY_PIP

      index_extended  = is_extended(hand_landmarks, idx_tip, idx_pip)
      middle_extended = is_extended(hand_landmarks, mid_tip, mid_pip)
      ring_extended   = is_extended(hand_landmarks, rng_tip, rng_pip)
      pinky_extended  = is_extended(hand_landmarks, pnk_tip, pnk_pip)

      # special rule: if only index is up -> force scroll lock OFF
      only_index_up = index_extended and not middle_extended and not ring_extended and not pinky_extended
      if only_index_up:
        if scroll_locked or scroll_up or scroll_down:
          print("SCROLL LOCK FORCED OFF (index-only)")
        scroll_locked = False
        scroll_up = False
        scroll_down = False
      else:
        # lock condition: index + middle extended together, tips close; ring & pinky not extended
        tips_close = norm_dist(lm[idx_tip], lm[mid_tip]) < FINGERS_TOGETHER_THRESH
        should_lock = index_extended and middle_extended and tips_close and (not ring_extended) and (not pinky_extended)

        if should_lock:
          if not scroll_locked:
            print("SCROLL LOCKED")
          scroll_locked = True

          # direction: UP when the two extended fingers point up
          idx_theta = angle_to_vertical_deg(hand_landmarks, idx_pip, idx_tip)
          mid_theta = angle_to_vertical_deg(hand_landmarks, mid_pip, mid_tip)
          avg_theta = 0.5 * (idx_theta + mid_theta)

          scroll_up = (avg_theta <= ORIENT_ANGLE_TOL_DEG)
          # do not set scroll_down here; down is controlled by fist gesture below
          if scroll_up:
            print("SCROLL UP")
          else:
            # not up; keep down state controlled elsewhere
            pass
        else:
          # not in lock pose
          if scroll_locked:
            print("SCROLL UNLOCKED")
          scroll_locked = False
          scroll_up = False
          # do not clear scroll_down yet; we evaluate fist next

        # scroll_down = fist (all four closed)
        fist = (not index_extended) and (not middle_extended) and (not ring_extended) and (not pinky_extended)
        if fist:
            # ensure continuous prints while fist is held
            if not scroll_down:
                print("SCROLL DOWN START")
            scroll_down = True
            print("SCROLL DOWN")
        else:
            if scroll_down:
                print("SCROLL DOWN OFF")
            scroll_down = False

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