import cv2
import os
from imutils import paths
def load(imagePaths):
    datas = []
    label_foders = []

    for (i, imagePath) in enumerate(imagePaths):
        image = cv2. imread(imagePath)
        label_foder = imagePath.split(os.path.sep)[-1]

        datas.append(image)
        label_foders.append(label_foder)

    return datas, label_foders
