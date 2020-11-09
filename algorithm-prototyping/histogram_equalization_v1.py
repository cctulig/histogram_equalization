import numpy as np
import cv2 as cv
from PIL import Image


def histogram_equalization_v1(pil_img: Image):
    numpy_img = np.array(pil_img)
    image = cv.cvtColor(numpy_img, cv.COLOR_RGB2BGR)

    # convert to grayscale
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # create a CLAHE object (Arguments are optional).
    clahe: cv.CLAHE = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(gray_image)

    # Convert OpenCV image to PIL image
    im1 = Image.fromarray(cv.cvtColor(cl1, cv.COLOR_BGR2RGB))
    im2 = Image.fromarray(cv.cvtColor(gray_image, cv.COLOR_BGR2RGB))
    im3 = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    # im4 = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    # im5 = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))

    return [im1, im2, im3, im3, im3]
