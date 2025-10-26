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
WHEEL_DELTA = 50

def move_cursor(x : int, y : int) -> None:
    """
    Moves the cursor to an x, y coordinate.
    
    @param x the x coordinate
    @param y the y coordinate
    """
    user32.SetCursorPos(x, y)

    return None

def left_click() -> None:
    """
    Performs a simulated left click with the mouse.
    """
    # Left Down
    user32.mouse_event(MOUSE_DOWN_EVENT, 0, 0, 0, 0)

    # Left Up
    user32.mouse_event(MOUSE_UP_EVENT, 0, 0, 0, 0)

    return None

def alt_tab_start() -> None:
    """
    Holds the Alt key down and presses Tab once to start the menu.
    """
    # Press Alt Key, Press & Release Tab Key
    user32.keybd_event(ALT_KEY, 0, 0, 0)      
    user32.keybd_event(TAB_KEY, 0, 0, 0)     
    user32.keybd_event(TAB_KEY, 0, UP_EVENT, 0)
    return None

def alt_tab_cycle() -> None:
    """
    Presses and releases the Tab key. Assumes Alt is already held.
    """
    # Press & Release Tab Key
    user32.keybd_event(TAB_KEY, 0, 0, 0)     
    user32.keybd_event(TAB_KEY, 0, UP_EVENT, 0) 
    return None

def alt_tab_release() -> None:
    """
    Releases the Alt key to select the window.
    """
    # Release Alt Key
    user32.keybd_event(ALT_KEY, 0, UP_EVENT, 0) 
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

def get_screen_dimensions() -> list[int, int]:
    """
    Calculates the midpoint of the primary screen using ctypes on Windows.

    @return tuple: A tuple containing the (x, y) coordinates of the midpoint.
    """
    user32 = ctypes.windll.user32
    
    # Get the screen width using SM_CXSCREEN (index 0)
    screen_width = user32.GetSystemMetrics(0)
    
    # Get the screen height using SM_CYSCREEN (index 1)
    screen_height = user32.GetSystemMetrics(1)
    
    return screen_width, screen_height

### For Testing Purposes Only ###
# alt_tab()
# left_click(1, 1)
# ctrl_tab()
# scroll(up=True, down=False)

# width, height = get_screen_dimensions()
# print(f'WIDTH: {width}, HEIGHT: {height}')
