from pynput.keyboard import Key, Listener
import time

key_events = []

def on_press(key):
    key_events.append(('press', key, time.time()))

def on_release(key):
    key_events.append(('release', key, time.time()))
    if key == Key.esc:
        # Stop listener
        return False
time.sleep(3)
# Start listening to keyboard inputs
with Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()

# Process the recorded key events to simulate timing with pydirectinput
prev_time = None
for event in key_events:
    action, key, event_time = event
    if prev_time is not None:
        sleep_time = event_time - prev_time
        print(f'time.sleep({sleep_time})')
    if action == 'press':
        if hasattr(key, 'char'):  # This is a character key
            code_line = f'pydirectinput.keyDown("{key.char}")'
        else:  # This is a special key
            key_name = key.name if hasattr(key, 'name') else str(key)
            code_line = f'pydirectinput.keyDown("{key_name}")'
    elif action == 'release':
        if hasattr(key, 'char'):  # This is a character key
            code_line = f'pydirectinput.keyUp("{key.char}")'
        else:  # This is a special key
            key_name = key.name if hasattr(key, 'name') else str(key)
            code_line = f'pydirectinput.keyUp("{key_name}")'
    print(code_line)
    prev_time = event_time
