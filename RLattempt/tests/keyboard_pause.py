import keyboard

def on_key_event(event):
    if event.name == 'q':
        print("You pressed 'q'. Exiting now.")
        exit()
    else:
        print(f"You pressed '{event.name}'.")

# Hook the keyboard to listen to all keypresses
keyboard.hook(on_key_event)

# Block the program so it doesn't exit immediately
keyboard.wait()
