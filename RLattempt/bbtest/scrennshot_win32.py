import cv2
import win32gui, win32ui, win32con, win32api
import numpy as np
import time

def grab_screen_shot():

        # 获取桌面
        hwin = win32gui.GetDesktopWindow()
        x = 238
        y = 50
        w = 510
        h = 286

        # 返回句柄窗口的设备环境、覆盖整个窗口，包括非客户区，标题栏，菜单，边框
        hwindc = win32gui.GetWindowDC(hwin)

        # 创建设备描述表
        srcdc = win32ui.CreateDCFromHandle(hwindc)

        # 创建一个内存设备描述表
        memdc = srcdc.CreateCompatibleDC()

        # 创建位图对象
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(srcdc, w, h)
        memdc.SelectObject(bmp)

        # 截图至内存设备描述表
        memdc.BitBlt((0, 0), (w, h), srcdc, (x, y), win32con.SRCCOPY)

        signedIntsArray = bmp.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (h, w, 4)

        # 内存释放
        srcdc.DeleteDC()
        memdc.DeleteDC()
        win32gui.ReleaseDC(hwin, hwindc)
        win32gui.DeleteObject(bmp.GetHandle())

        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
frame_number=0
#for i in range(1000):
#        frame = grab_screen_shot()
#        filename = f"frame_{frame_number}.png"

#        cv2.imwrite(filename,frame)
#        frame_number+=1
#        time.sleep(0.2)
frame = grab_screen_shot()
cv2.imwrite("square_screenshot.png",frame)
#IMG_WIDTH = 1021                                #Game capture resolution
#IMG_HEIGHT = 573                             
#MODEL_WIDTH = int(1021/2)                  #Ai vision resolution
#MODEL_HEIGHT = int(573/2)
#frame=cv2.resize(frame,(MODEL_HEIGHT,MODEL_WIDTH))
#observation =  cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#cv2.imshow('Screenshot', observation)
# Wait for a key press and then close the displayed window
