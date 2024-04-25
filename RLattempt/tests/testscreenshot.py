import cv2
import mss
import numpy as np
import pytesseract
import time
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\12470\AppData\Local\Tesseract-OCR\tesseract.exe'
def take_square_screenshot(file_path, x, y, size):
    # Initialize MSS
    with mss.mss() as sct:
        # Capture the entire screen
        screenshot = sct.shot(output=file_path)

    # Load the captured image using OpenCV
    frame = cv2.imread(file_path)

    # Crop the image to the specified square region
    
    #frame = frame[31:604, 10:1031] #Cutting the frame to the location of the text "next" (bottom left corner)
    frame = frame[461:468, 246:778]
    #frame = frame[24:31, 80:392]
    #frame=frame[567:584, 90:118]
    #image_hsv = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    #image_hsv = cv2.resize(frame, (400, 225))
    #image_hsv = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGRA)
    cv2.imwrite(file_path,frame)
    
    return file_path
def calculate_red_percentage(file_path):
    """
    This function calculates the percentage of red color in a health bar
    from a given image path. It assumes that the health bar is horizontal.
    
    Parameters:
    - image_path: The file path to the image containing the health bar.
    
    Returns:
    - A tuple containing the percentage of the red color in the health bar
      and the width of the red area in pixels.
    """
    # Load the image
    image = cv2.imread(file_path)
    
    # Convert the image to the HSV color space
         #Cut out the hp bar from the frame
   
    
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

     # Define the color range for red in HSV
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Create two masks for the two red ranges and combine them
    mask1 = cv2.inRange(image_hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(image_hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Assume the largest contour is the health bar
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        health_bar_width = w
    else:
        # No red color detected, health bar is empty or an error occurred
        health_bar_width = 0
    
    # Calculate the percentage of the health bar
    curr_hp = round((health_bar_width / image.shape[1]),2)
    print(curr_hp)
    return curr_hp, health_bar_width
# Specify the file path to save the screenshot
file_path = "square_screenshot.png"

# Specify the coordinates (x, y) and size of the square region
x_coordinate = 90
y_coordinate = 567
square_size = 500

# Take a square screenshot and save it to the specified file path
take_square_screenshot(file_path, x_coordinate, y_coordinate, square_size)
# Load and display the saved square screenshot using OpenCV (optional)
timecount=time.time()
calculate_red_percentage(file_path)

print(time.time()-timecount)
print('\n')
image = cv2.imread(file_path)
cv2.imshow("Square Screenshot", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
