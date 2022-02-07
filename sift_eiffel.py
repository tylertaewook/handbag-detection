import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread("./images/etc/n1.jpg")
img2 = cv2.imread("./logos/guc2.png")
# img1 = template = cv2.Canny(img1, 50, 200)
# img2 = template = cv2.Canny(img2, 50, 200)

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# sift
sift = cv2.SIFT_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

# feature matching
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(descriptors_1, descriptors_2)
matches = sorted(matches, key=lambda x: x.distance)

img3 = cv2.drawMatches(
    img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2
)
plt.imshow(img3), plt.show()

