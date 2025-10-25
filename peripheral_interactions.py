"""
The peripheral_interactions file is reserved for all functions related to simulating the mouse and keyboard 
actions. 

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
import ctypes
import time

# Set the process-default DPI awareness to system-DPI awareness.
ctypes.windll.user32.SetProcessDPIAware()

# Creating user to perform ctypes actions.
user32 = ctypes.windll.user32

# GLOBAL STATIC VARIABLES
UP_EVENT = 0x0002
MOUSE_DOWN_EVENT = 0x0002
MOUSE_UP_EVENT = 0x0004
ALT_KEY = 0x12
TAB_KEY = 0x09
CTRL_KEY = 0x11
MOUSE_WHEEL = 0x0800

# Wheel Speed
WHEEL_DELTA = 120

def left_click(x : int, y : int) -> None:
    """
    Performs a simulated left click with the mouse.

    @param x the x coordinate
    @param y the y coordinate
    """

    ### Unsure if we need this since cursor should already be set ###
    user32.SetCursorPos(x, y)

    # Left Down
    user32.mouse_event(MOUSE_DOWN_EVENT, 0, 0, 0, 0)

    # Left Up
    user32.mouse_event(MOUSE_UP_EVENT, 0, 0, 0, 0)

    return None

def alt_tab() -> None:
    """
    Perform a simulated alt tab with the keyboard.
    """
    # Press Alt + Tab
    user32.keybd_event(ALT_KEY, 0, 0, 0)
    user32.keybd_event(TAB_KEY, 0, 0, 0)

    # Release Alt + Tab
    user32.keybd_event(ALT_KEY, 0, UP_EVENT, 0)
    user32.keybd_event(TAB_KEY, 0, UP_EVENT, 0)

    return None

def ctrl_tab() -> None:
    """
    Perform a simulated ctrl tab with the keyboard.
    """
    # Press Ctrl + Tab
    user32.keybd_event(CTRL_KEY, 0, 0, 0)
    user32.keybd_event(TAB_KEY, 0, 0, 0)

    # Release Ctrl + Tab
    user32.keybd_event(CTRL_KEY, 0, UP_EVENT, 0)
    user32.keybd_event(TAB_KEY, 0, UP_EVENT, 0)

    return None

def scroll(up : bool, down : bool) -> None:
    """
    Scroll the current page up/down depending on the parameters passed in. You MUST pass through only one
    True and False value, anything else will return without a scroll.

    @param up a boolean to determine scrolling up
    @param down a boolean to determine scrolling down
    """
    if (up is True and down is True) or (up is False and down is False):
        print('RETURNING AS UP AND DOWN ARE BOTH EITHER TRUE OR FALSE')
        return None

    # Scroll Down
    if down is True:
        user32.mouse_event(MOUSE_WHEEL, 0, 0, -WHEEL_DELTA * 2, 0)

    # Scroll Up
    if up is True:
        user32.mouse_event(MOUSE_WHEEL, 0, 0, WHEEL_DELTA * 2, 0)

    return None



### For Testing Purposes Only ###
# alt_tab()
# left_click(1, 1)
# ctrl_tab()
# scroll(up=True, down=False)
