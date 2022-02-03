import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread("./images/gc/bracelet.jpg", 0)
img2 = cv2.imread("./logos/gucc2.png", 0)
# img1 = template = cv2.Canny(img1, 50, 200)
# img2 = template = cv2.Canny(img2, 50, 200)

template = cv2.imread("./logos/gg1.png")
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = cv2.Canny(template, 50, 200)

# 입력 영상 밝기 50증가, 가우시안 잡음(sigma=10) 추가하여 약간의 변형을 줌
noise = np.zeros(template.shape, np.int32)

# cv2.randn은 가우시안 형태의 랜덤 넘버를 지정, 노이즈 영상에 평균이 50 시그마 10인 노이즈 추가
cv2.randn(noise, 50, 10)

# 노이즈를 입력 영상에 더함, 원래 영상보다 50정도 밝아지고 시그마 10정도 변형
template = cv2.add(template, noise, dtype=cv2.CV_8UC3)

cv2.imshow("", template)
cv2.waitKey()
