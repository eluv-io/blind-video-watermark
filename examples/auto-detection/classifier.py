import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('sec2.jpeg',0)
img2 = img.copy()
# All the 6 methods for comparison in a list
methods = ['cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR_NORMED']
a = "0123456789abcdef"
plt.rc('font', size=8)
for meth in methods:
    for i in range(16):
        template = cv.imread('nums/{}.jpeg'.format(a[i]), 0)
        w, h = template.shape[::-1]
        img = img2.copy()
        method = eval(meth)
        res = cv.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv.rectangle(img, top_left, bottom_right, 255, 2)
        plt.subplot(8, 4, i * 2 + 1),plt.imshow(res,cmap = 'gray')
        plt.title('{}, Score:{}'.format(a[i], max_val)), plt.xticks([]), plt.yticks([])
        plt.subplot(8, 4, i * 2 + 2),plt.imshow(img,cmap = 'gray')
        # plt.title('Detected #{}, Score: {}'.format(a[i], max_val)), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()
