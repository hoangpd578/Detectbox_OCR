import cv2
from pdf2image import convert_from_path
import numpy as np
from collections import Counter


def convert_pdf2image(path_image):
    """ conver pdf2image """
    pages = convert_from_path(path_image, 500)
    for i in range(len(pages)):
        pages[i].save("../images/{}.jpeg".format(i), "JPEG")


def detect_line_horizontal_morphological(path_image):
    """ using morphological to dectect line horizontal"""

    image = cv2.imread(path_image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
    gray = cv2.bitwise_not(gray_image)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                              cv2.THRESH_BINARY, 15, -2)

    # coppy image
    horizontal = np.copy(bw)
    vertical = np.copy(bw)

    cols = horizontal.shape[1]
    horizontal_size = int(cols / 30)

    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))

    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    cv2.imwrite("../images/line_horzontal.jpeg",horizontal)

    indexs = np.nonzero(horizontal)
    indexs = indexs[0]

    filter_line = []
    for i in range(len(indexs)):
        if i ==0:
            filter_line.append(indexs[indexs[i]])
        else:
            if np.abs(indexs[i] - indexs[i-1] >=20):
                filter_line.append(indexs[i])


    print(len((filter_line)))
    for i in range(len(filter_line)-1):
        image_croped = cv2.imwrite("../images/crop/{}.jpg".format(i), image[filter_line[i]:filter_line[i+1], :, :])


def detect_line_vertical_morphology(path_image):
    """ using morphological to dectect line horizontal"""
    pass



if __name__ == '__main__':
    # convert_pdf2image("../images/SOKA.pdf")
    detect_line_horizontal_morphological("../images/2.jpeg")