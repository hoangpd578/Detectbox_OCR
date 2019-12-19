import tempfile
import cv2
from PIL import Image
import numpy as np
from imutils import paths

IMAGE_SIZE = 1800
BINARY_THREHOLD = 120

size = None

def get_size(img):
    goble size
    if size is None:
        l_x, w_y = img.size
        factor = max(1, int(IMAGE_SIZE/l_x))
        size = factor * l_x, factor * l_y
      return size

def process_orc(imagePath):
	temp_filename = set_dpi(imagePath)
	im_new = remove_noise(temp_filename)

	return im_new

def set_dpi(imagePath):
	im = Image.open(imagePath)
	size = get_size(im)
	im_resize = im.resize(size, Image.ANTIALIAS)

	temp_file = tempfile.NamedTemporaryFile(delete = False, suffix = '.jpg')
	temp_filename = temp_file.name
	im_resize.save(temp_filename, dpi = (300, 300))
	return temp_filename

def image_smoothing(img):
    ret1, th1 = cv2.threshold(img, BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (1, 1), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2. THRESH_OTSU)
    return th3

def remove_noise(file_name):
	img = cv2.imread(file_name, 0)
	fillter = cv2.adaptiveThreshold(imag.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)
	kernel = np.ones((1, 1), dtype = 'uint8')
	opening = cv2.morphologyEx(fillter, cv2.MORPH_OPEN, kernel)
	closing = cv2.morphologyEx(fillter, cv2.MORPH_CLOSE, kernel)
	img = image_smoothing(img)
	or_img = cv2.bitwise_or(img, closing)
	return or_img


ap = argparse.ArgumentParser()
ap.add_argument('-p', '--paths', help = "path to image")
#ap.add_argument('-c', '--crop', help=" path save box")

args = vars(ap.parse_args())


imagePaths = list(paths.list_images(args['paths']))