import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread("./images/gc/bracelet.jpg", 0)
img2 = cv2.imread("./logos/gucc2.png", 0)
# img1 = template = cv2.Canny(img1, 50, 200)
# img2 = template = cv2.Canny(img2, 50, 200)

template = cv2.imread("./logos/g3.png")
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = cv2.Canny(template, 50, 200)
cv2.imshow("", template)
cv2.waitKey()
