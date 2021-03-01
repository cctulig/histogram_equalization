import cv2 as cv


def CLAHE(img):
    return CLAHE.clahe.apply(img)


CLAHE.clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
