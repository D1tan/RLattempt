import threading
import keyboard
import time

paused = False
force_continue = False  # New variable to manage forced continuation

def check_for_pause_and_continue():
    global paused, force_continue
    while True:
        # Check for 'p' press to toggle pause
        if keyboard.is_pressed('p'):
            paused = not paused
            if paused:
                print("Paused. Press 'p' again to resume or 'l' to force continue.")
            else:
                print("Resumed.")
            while keyboard.is_pressed('p'):  # Wait for 'p' to be released
                time.sleep(1)
        
        # Check for 'l' press to force continue
        if keyboard.is_pressed('l'):
            if paused or force_continue:  # Only act if paused or already forcing continuation
                force_continue = True
                paused = False
                print("Force continuing...")
            while keyboard.is_pressed('l'):  # Wait for 'l' to be released
                time.sleep(1)


def start_pause_listener():
    pause_thread = threading.Thread(target=check_for_pause_and_continue)
    pause_thread.daemon = True
    pause_thread.start()

def is_paused():
    return paused

def should_force_continue():
    global force_continue
    if force_continue:  # If force_continue is True, reset it to False and return True
        force_continue = False
        return True
    return False
