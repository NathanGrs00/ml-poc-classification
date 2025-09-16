# ocr-scanner/ocr.py
from PIL import Image
import pytesseract

# Path to your image file
image_path = 'sample.png'

# Open the image and perform OCR
image = Image.open(image_path)
text = pytesseract.image_to_string(image)

print('Extracted text:')
print(text)