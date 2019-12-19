import cv2
import pytesseract
from pytesseract import Output

img = cv2.imread('../DetectText_ORC/crop_image/image2.jpg3.png')

d = pytesseract.image_to_data(img, output_type=Output.DICT)
text = str(((pytesseract.image_to_string(img, lang= 'vie', config = 'tesseract vietsample.tif output -l vie-t'))))
print(text)
