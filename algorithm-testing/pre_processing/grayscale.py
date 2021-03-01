import cv2 as cv


def grayscale(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
