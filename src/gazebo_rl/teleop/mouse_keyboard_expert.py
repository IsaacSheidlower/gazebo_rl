import threading
import numpy as np
from typing import Tuple
from pynput import mouse, keyboard
from ui_interface import UIInterface

# mouse_keyboard_expert.py
from screeninfo import get_monitors
import numpy as np
from threading import Lock
from pynput import mouse, keyboard

# Define the UIInterface
class UIInterface:
    def get_action(self):
        raise NotImplementedError

# Implement the MouseKeyboardExpert class using pynput
class MouseKeyboardExpert(UIInterface):
    def __init__(self):
        self.action = np.zeros(6)  # Assuming 6 axes for the Joy message
        self.buttons = {}          # Dictionary to hold button states
        self.lock = Lock()         # To handle concurrent access

        self.spacebar_pressed = False  # Deadman switch state

        self.w_pressed = False

        # Start the mouse and keyboard listeners
        self.mouse_listener = mouse.Listener(
            on_move=self.on_move,
            on_click=self.on_click,
            on_scroll=self.on_scroll)
        self.keyboard_listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)

        self.mouse_listener.start()
        self.keyboard_listener.start()

        # Get screen dimensions
        monitor = get_monitors()[0]
        self.screen_width = monitor.width
        self.screen_height = monitor.height

        self.deadzone_thresh = 0.02
        self.max_output = 0.075

        # Initialize previous mouse position for relative movement
        self.prev_x = None
        self.prev_y = None

    def on_move(self, x, y):
        with self.lock:
            if self.spacebar_pressed:
                if self.prev_x is None:
                    self.prev_x = x; self.prev_y = y
                    return
                else:
                    dx, dy = self.prev_x - x, self.prev_y - y

                    normalized_x = (dx / (self.screen_width))
                    normalized_y = (dy / (self.screen_height))


                    # normalized_x = np.clip(normalized_x, -self.max_output, self.max_output); normalized_y = np.clip(normalized_y, -self.max_output, self.max_output)

                    # print(f"{dx:1.2f} {dy:1.2f} {normalized_x:1.2f} {normalized_y:1.2f}")

                    if abs(normalized_x) < self.deadzone_thresh: normalized_x = 0
                    if abs(normalized_y) < self.deadzone_thresh: normalized_y = 0
              
                if self.w_pressed:
                    self.action[3] = normalized_x
                    self.action[0] = 0; self.action[1] = 0
                    if normalized_x < -self.max_output: normalized_x = -0.1
                    elif normalized_x > 0.1: normalized_x = 0.1
                    if normalized_y < -0.1: normalized_y = -0.1
                    elif normalized_y > 0.1: normalized_y = 0.1
                else:
                    if normalized_x < -self.max_output: normalized_x = -self.max_output
                    elif normalized_x > self.max_output: normalized_x = self.max_output
                    if normalized_y < -self.max_output: normalized_y = -self.max_output
                    elif normalized_y > self.max_output: normalized_y = self.max_output
                    # Use center of screen as 0 (rather than dx,dy)
                    # Calculate normalized x and y positions scaled from -1 to 1
                    # normalized_x = (x / self.screen_width) * 2 - 1
                    # normalized_y = (y / self.screen_height) * 2 - 1
                    
                    # Flip y-axis if necessary (optional)
                    # normalized_y = -normalized_y  # Comment this line if not needed
                    
                    # Update the action with normalized positions
                    self.action[0] = normalized_y
                    self.action[1] = normalized_x
                    self.action[3] = 0
            else:
                # Reset action if spacebar is not pressed
                self.action[0] = 0
                self.action[1] = 0
                self.action[2] = 0
                self.prev_x = None
                self.prev_y = None  # Reset previous positions to avoid jumps when resuming

    def on_click(self, x, y, button, pressed):
        with self.lock:
            # Update buttons dictionary based on mouse clicks
            button_name = f'mouse_{button.name}'
            print(button_name)
            self.buttons[button_name] = int(pressed)

    def on_scroll(self, x, y, dx, dy):
        with self.lock:
            if self.spacebar_pressed:
                # Update the action based on scroll wheel movement
                # For example, map scroll to axis 2
                # scaling_factor = 0.1  # Adjust as needed
                # self.action[2] += dy * scaling_factor
                self.action[2] = 0.05 if dy > 0 else -0.05
            else:
                # Reset action if spacebar is not pressed
                self.action[2] = 0

    def on_press(self, key):
        with self.lock:
            try:
                key_name = key.char
            except AttributeError:
                key_name = key.name
            self.buttons[key_name] = 1  # Key is pressed

            if key_name == 'space' and not self.spacebar_pressed:
                print('ACTIVE')
                self.spacebar_pressed = True
                # Reset previous mouse position when spacebar is pressed
                self.prev_x = None
                self.prev_y = None

            if key_name == 'w' and not self.w_pressed:
                print("ANGLE ACTIVE")
                self.w_pressed = True

    def on_release(self, key):
        with self.lock:
            try:
                key_name = key.char
            except AttributeError:
                key_name = key.name
            self.buttons[key_name] = 0  # Key is released

            if key_name == 'space':
                print(f"...inactive...")
                self.spacebar_pressed = False
                # Reset action when spacebar is released
                self.action[0] = 0
                self.action[1] = 0
                self.action[2] = 0
                self.prev_x = None
                self.prev_y = None

            if key_name == 'w': 
                self.w_pressed = False
                self.action[3] = 0
                print("....angle inactive....")

    def get_action(self):
        with self.lock:
            # Return a copy of the current action and buttons
            action_copy = self.action.copy()
            buttons_copy = self.buttons.copy()
        return action_copy, buttons_copy

def main():
    import time

    mk_expert = MouseKeyboardExpert()
    print("Mouse and keyboard listener started. Move the mouse or press keys.")

    try:
        while True:
            action, buttons = mk_expert.get_action()
            print(f"Action: {action}, Buttons: {buttons}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting...")
        # Stop the listeners
        mk_expert.mouse_listener.stop()
        mk_expert.keyboard_listener.stop()


if __name__ == "__main__":
    main()
