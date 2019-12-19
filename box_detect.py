import cv2
import numpy as np
import loadPath
import argparse
import os
from imutils import paths

def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

def box_extraction(imagePaths, cropped_dir_path):
    images, label = loadPath.load(imagePaths)
    for (i, image) in enumerate(images):
        img = cv2.cvtColor(image, cv2. COLOR_BGR2GRAY)  # Read the image
        (thresh, img_bin) = cv2.threshold(img, 128, 255,
                                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Thresholding the image
        img_bin = 255-img_bin  # Invert the image

        # tao 2 kernel
        kernel_length = np.array(img).shape[1]//20

        verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))

        hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
        # A kernel of (3 X 3) ones.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # Morphological detect line
        img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
        verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)

        img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
        horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
        cv2.imwrite("/home/dang-hoang/ATom/DetectText_ORC/image_line/" + str(label[i]) + ".png", horizontal_lines_img)

        alpha = 0.5
        beta = 1.0 - alpha

        # tao mot hinh anh moi ket hop tu hai anh
        img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
        img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
        (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Tim duong vien de detect cac box

        contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Xap xep tat ca cac duong vien theo thu tu tu tren xuong
        (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")

        idx = 0
        for c in contours:
            # tra ve vi tri chieu rong, cao cho moi duowng vien
            x, y, w, h = cv2.boundingRect(c)

            #crop and save
            if (w > 10 and h > 3) and w > 3*h:
                idx += 1
                new_img = img[y:y+h, :]
                cv2.imwrite(cropped_dir_path + str(label[i]) + str(idx) + '.png', new_img)

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--paths', help = "path to image")
#ap.add_argument('-c', '--crop', help=" path save box")

args = vars(ap.parse_args())


imagePaths = list(paths.list_images(args['paths']))
cropped_dir_path = "/home/dang-hoang/ATom/DetectText_ORC/crop_image/"
box_extraction(imagePaths, cropped_dir_path)
