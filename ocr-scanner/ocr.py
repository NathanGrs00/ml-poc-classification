import time
import pyautogui
from PIL import Image
import pytesseract

while True:
    screenshot = pyautogui.screenshot()
    screenshot.save('screenshot.png')
    text = pytesseract.image_to_string(Image.open('screenshot.png'))
    print('OCR Result:', text)
    time.sleep(10)
