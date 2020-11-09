import numpy as np
import cv2 as cv
from PIL import Image


def histogram_equalization_v2(pil_img: Image):
    numpy_img = np.array(pil_img)
    image = cv.cvtColor(numpy_img, cv.COLOR_RGB2BGR)

    # color histogram equalization
    eq = histogram_equalization(image)

    # Convert OpenCV image to PIL image
    im1 = Image.fromarray(cv.cvtColor(eq, cv.COLOR_BGR2RGB))
    im2 = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    # im3 = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    # im4 = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    # im5 = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))

    return [im1, im2, im2, im2, im2]


# function for color image equalization
# source: https://stackoverflow.com/questions/31998428/opencv-python-equalizehist-colored-image
def histogram_equalization(img_in):
    # segregate color streams
    b, g, r = cv.split(img_in)
    h_b, bin_b = np.histogram(b.flatten(), 256, [0, 256])
    h_g, bin_g = np.histogram(g.flatten(), 256, [0, 256])
    h_r, bin_r = np.histogram(r.flatten(), 256, [0, 256])
    # calculate cdf
    cdf_b = np.cumsum(h_b)
    cdf_g = np.cumsum(h_g)
    cdf_r = np.cumsum(h_r)

    # mask all pixels with value=0 and replace it with mean of the pixel values
    cdf_m_b = np.ma.masked_equal(cdf_b, 0)
    cdf_m_b = (cdf_m_b - cdf_m_b.min()) * 255 / (cdf_m_b.max() - cdf_m_b.min())
    cdf_final_b = np.ma.filled(cdf_m_b, 0).astype('uint8')

    cdf_m_g = np.ma.masked_equal(cdf_g, 0)
    cdf_m_g = (cdf_m_g - cdf_m_g.min()) * 255 / (cdf_m_g.max() - cdf_m_g.min())
    cdf_final_g = np.ma.filled(cdf_m_g, 0).astype('uint8')

    cdf_m_r = np.ma.masked_equal(cdf_r, 0)
    cdf_m_r = (cdf_m_r - cdf_m_r.min()) * 255 / (cdf_m_r.max() - cdf_m_r.min())
    cdf_final_r = np.ma.filled(cdf_m_r, 0).astype('uint8')
    # merge the images in the three channels
    img_b = cdf_final_b[b]
    img_g = cdf_final_g[g]
    img_r = cdf_final_r[r]

    img_out = cv.merge((img_b, img_g, img_r))
    # validation
    equ_b = cv.equalizeHist(b)
    equ_g = cv.equalizeHist(g)
    equ_r = cv.equalizeHist(r)
    equ = cv.merge((equ_b, equ_g, equ_r))
    # print(equ)
    # cv.imwrite('output_name.png', equ)
    return img_out