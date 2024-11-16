import threading
import numpy as np
from typing import Tuple
from pynput import mouse, keyboard


class MouseKeyboardExpert:
    """
    This class provides an interface to mouse and keyboard input.
    It continuously reads mouse and keyboard state and provides
    a "get_action" method to get the latest action and button state.
    """

    def __init__(self):
        self.state_lock = threading.Lock()
        self.latest_data = {
            "action": np.zeros(6),
            "buttons": {
                "mouse": {"left": False, "right": False, "middle": False},
                "keyboard": set(),
            },
        }
        self.prev_mouse_position = None

        # Start listeners for mouse and keyboard
        self.mouse_listener = mouse.Listener(
            on_move=self._on_move,
            on_click=self._on_click,
            on_scroll=self._on_scroll,
        )
        self.keyboard_listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )

        self.mouse_listener.start()
        self.keyboard_listener.start()

    # Mouse event handlers
    def _on_move(self, x, y):
        with self.state_lock:
            if self.prev_mouse_position is not None:
                dx = x - self.prev_mouse_position[0]
                dy = y - self.prev_mouse_position[1]
            else:
                dx = dy = 0
            self.latest_data["action"][0] += dx
            self.latest_data["action"][1] += dy
            self.prev_mouse_position = (x, y)

    def _on_click(self, x, y, button, pressed):
        with self.state_lock:
            if button == mouse.Button.left:
                self.latest_data["buttons"]["mouse"]["left"] = pressed
            elif button == mouse.Button.right:
                self.latest_data["buttons"]["mouse"]["right"] = pressed
            elif button == mouse.Button.middle:
                self.latest_data["buttons"]["mouse"]["middle"] = pressed

    def _on_scroll(self, x, y, dx, dy):
        with self.state_lock:
            self.latest_data["action"][2] += dy

    # Keyboard event handlers
    def _on_press(self, key):
        with self.state_lock:
            try:
                self.latest_data["buttons"]["keyboard"].add(key.char)
            except AttributeError:
                self.latest_data["buttons"]["keyboard"].add(str(key))

    def _on_release(self, key):
        with self.state_lock:
            try:
                self.latest_data["buttons"]["keyboard"].discard(key.char)
            except AttributeError:
                self.latest_data["buttons"]["keyboard"].discard(str(key))

    def get_action(self) -> Tuple[np.ndarray, dict]:
        """Returns the latest action and button state."""
        with self.state_lock:
            # Copy the data to prevent external modification
            action = np.copy(self.latest_data["action"])
            buttons = {
                "mouse": self.latest_data["buttons"]["mouse"].copy(),
                "keyboard": self.latest_data["buttons"]["keyboard"].copy(),
            }
            # Optionally reset action after reading
            self.latest_data["action"] = np.zeros(6)
            return action, buttons


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
