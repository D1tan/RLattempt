import ctypes
import time

SendInput = ctypes.windll.user32.SendInput


W = 0x11
A = 0x1E
S = 0x1F
D = 0x20

M = 0x32
J = 0x24
K = 0x25
LSHIFT = 0x2A
R = 0x13#用R代替识破
SPACE= 0x2F
Q = 0x10

I = 0x17
O = 0x18
P = 0x19
C = 0x2E
F = 0x21

up = 0xC8
down = 0xD0
left = 0xCB
right = 0xCD

esc = 0x01

# C struct redefinitions 
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Actuals Functions

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
    
    
def stop():
    ReleaseKey(W)
    ReleaseKey(S)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(LSHIFT)
    ReleaseKey(K)
    ReleaseKey(F)
    #time.sleep(0.1)
    

    
def go_forward():
    ReleaseKey(W)
    ReleaseKey(S)
    PressKey(W)
    
    
def go_back():
    ReleaseKey(W)
    ReleaseKey(S)
    PressKey(S)
    
    
def go_left():
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    PressKey(W)
    PressKey(A)
    
    
    
def go_right():
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    PressKey(W)
    PressKey(D)
    
def attack():
    PressKey(J)
    time.sleep(0.01)
    ReleaseKey(J)
    time.sleep(0.48)

def heavy():
    PressKey(K)
    time.sleep(0.01)
    ReleaseKey(K)
    time.sleep(0.64)
        
def jump_attack():
    PressKey(W)
    time.sleep(0.05)
    PressKey(F)
    time.sleep(0.05)
    ReleaseKey(F)
    time.sleep(0.2)
    PressKey(J)
    time.sleep(0.05)
    ReleaseKey(J)
    
    time.sleep(0.8)

def jump_heavy():
    PressKey(W)
    time.sleep(0.05)
    PressKey(F)
    time.sleep(0.05)
    ReleaseKey(F)
    time.sleep(0.1)
    PressKey(K)
    time.sleep(0.05)
    ReleaseKey(K)
    time.sleep(0.9)
    
def dodge_forward():#闪避
    PressKey(W)
    PressKey(LSHIFT)
    time.sleep(0.05)
    ReleaseKey(LSHIFT)
    time.sleep(0.4)

def dodge_backward():#闪避
    PressKey(S)
    PressKey(LSHIFT)
    time.sleep(0.05)
    ReleaseKey(LSHIFT)
    time.sleep(0.4)
    
def dodge_left():#闪避
    PressKey(W)
    PressKey(A)
    PressKey(LSHIFT)
    time.sleep(0.05)
    ReleaseKey(LSHIFT)
    time.sleep(0.4)

def dodge_right():#闪避
    PressKey(W)
    PressKey(D)
    PressKey(LSHIFT)
    time.sleep(0.05)
    ReleaseKey(LSHIFT)
    time.sleep(0.4)

def weapon_art():
    PressKey(Q)
    time.sleep(0.05)
    ReleaseKey(Q)
    time.sleep(0.9)
    PressKey(K)
    time.sleep(0.05)
    ReleaseKey(K)
    
    time.sleep(1.8)

def heal():#闪避
    PressKey(R)
    time.sleep(0.05)
    ReleaseKey(R)
    
    time.sleep(0.8)


if __name__ == '__main__':
    time.sleep(2)
    count=time.time()
    weapon_art()
    dodge_forward()
    print(time.time()-count)
    time.sleep(1)
    stop()
        
    
