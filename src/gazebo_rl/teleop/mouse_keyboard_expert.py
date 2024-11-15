import threading
import numpy as np
import pygame
from typing import Tuple
import time

class MouseKeyboardExpert:
    """
    This class provides an interface to read mouse and keyboard input.
    It continuously reads the mouse and keyboard state and provides
    a "get_action" method to get the latest action, button state, and key presses.
    """
    def __init__(self):
        pygame.init()
        self.state_lock = threading.Lock()
        self.latest_data = {"action": np.zeros(2), "buttons": [0, 0, 0], "keys": []}
        self.thread = threading.Thread(target=self._read_input)
        self.thread.daemon = True
        self.thread.start()

    def _read_input(self):
        while True:
            pygame.event.pump()
            events = pygame.event.get()
            with self.state_lock:
                mouse_x, mouse_y = pygame.mouse.get_rel()
                self.latest_data["action"] = np.array([-mouse_y, mouse_x])
                self.latest_data["buttons"] = list(pygame.mouse.get_pressed())
                self.latest_data["keys"] = [event.key for event in events if event.type == pygame.KEYDOWN]

    def get_action(self) -> Tuple[np.ndarray, list, list]:
        """Returns the latest action, button state, and key presses of the mouse and keyboard."""
        with self.state_lock:
            return self.latest_data["action"], self.latest_data["buttons"], self.latest_data["keys"]

def main():
    print("Starting input test. Move mouse and press keys to see input.")
    print("Press ESC to exit.")
    
    # Initialize the expert
    expert = MouseKeyboardExpert()

    for _ in range(100):
        pygame.event.pump()
        print(pygame.mouse.get_rel())
    
    try:
        for _ in range(100):
            # Get the latest action
            action, buttons, keys = expert.get_action()
            
            # Only print if there's actual input
            if np.any(action != 0) or any(buttons) or keys:
                print("\nCurrent Input State:")
                print(f"Mouse Movement (x, y, z): {action[:3]}")
                print(f"Rotation (roll, pitch, yaw): {action[3:]}")
                print(f"Mouse Buttons: {buttons}")
                
                # Convert key codes to readable format
                if keys:
                    key_names = [pygame.key.name(key) for key in keys]
                    print(f"Keys Pressed: {key_names}")
                
                # Check for ESC key to exit
                if pygame.K_ESCAPE in keys:
                    break
            
            # Small sleep to prevent excessive CPU usage
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        expert.thread.join()

if __name__ == "__main__":
    main()