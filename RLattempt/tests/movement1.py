import pydirectinput
import time
time.sleep(3)
pydirectinput.press('alt')
for i in range(10):
    pydirectinput.keyDown('space')
    time.sleep(0.002)
    pydirectinput.keyUp('space')
time.sleep(2)
pydirectinput.keyDown('a')
pydirectinput.keyDown('w')
time.sleep(0.8)
pydirectinput.keyUp('a')
time.sleep(1.5)
pydirectinput.keyDown('d')
time.sleep(2.3)
pydirectinput.keyUp('d')
pydirectinput.keyDown('shift')
time.sleep(3.8)
pydirectinput.keyUp('shift')
pydirectinput.keyUp('w')
pydirectinput.press('tab')

