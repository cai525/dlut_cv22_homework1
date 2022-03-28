import cv2
import numpy as np
import matplotlib.pyplot as plt
import pdb
from utils import *


def Sobel(image):
    image_X = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)  # 计算梯度信息，使用高的数据类型
    image_Y = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    return image_X, image_Y


def Construct_cornerness_map(image_xx, image_xy, image_yx, image_yy, image):
    k = 0.05
    w, h = image.shape
    R = np.zeros((w, h))
    for i in range(w):
        for j in range(h):
            M = np.array([[image_xx[i][j], image_xy[i][j]], [image_yx[i][j], image_yy[i][j]]])
            R[i][j] = np.linalg.det(M) - k * np.trace(M) ** 2
    return R


def Non_max_suppression(R, image):
    PointList = []
    threshold = 0.05
    R_max = np.max(R)
    w, h = image.shape
    print(R.shape, w, h)

    # for i in range(w):
    #     for j in range(h):
    #         if R[i][j] > R_max * threshold and R[i][j] == np.max(R[max(0, j - 1):min(j + 2, h - 1), max(0, i - 1):min(i + 2, w - 1)]):
    #             confidence = R[i][j] / R_max
    #             PointList.append([i, j, confidence])
    for i in range(w):
        for j in range(h):
            # 除了进行进行阈值检测 还对3x3邻域内非极大值进行抑制(导致角点很小，会看不清)
            if R[i, j] > R_max * threshold and R[i, j] == np.max(
                    R[max(0, i - 1):min(i + 2, w - 1), max(0, j - 1):min(j + 2, h - 1)]):
                confidence = R[i][j] / R_max
                PointList.append([j, i, confidence])
    sorted(PointList, key=lambda k: k[2], reverse=True)
    return PointList

    #############################################################################
    # TODO: YOUR HARRIS CORNER DETECTOR CODE HERE                                                      #
    #############################################################################


scale_factor = 0.5
image1 = load_image('../data/Notre Dame/921919841_a30df938f2_o.jpg')
feature_width = 16  # width and height of each local feature, in pixels.
image1 = cv2.resize(image1, (0, 0), fx=scale_factor, fy=scale_factor)
image1_bw = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)

img = image1_bw
image_x, image_y = Sobel(image=img)
image_xx = image_x ** 2
image_xy = image_x * image_y
image_yx = image_y * image_x
image_yy = image_y ** 2

kernelsize = (15, 15)
sigma = 2

image_xx = cv2.GaussianBlur(image_xx, kernelsize, sigma)
image_xy = cv2.GaussianBlur(image_xy, kernelsize, sigma)
image_yx = cv2.GaussianBlur(image_yx, kernelsize, sigma)
image_yy = cv2.GaussianBlur(image_yy, kernelsize, sigma)

R = Construct_cornerness_map(image_xx, image_xy, image_yx, image_yy, img)

PointList = Non_max_suppression(R, img)

np.save('PointList.npy', PointList)
print(PointList)
x = len(PointList)
print(x)
radius = 2
color = (0, 255, 0)  # Green
thickness = 1

for i in range(x):
    col = PointList[i][0]
    row = PointList[i][1]
    cv2.circle(image1, (col, row), radius, color, thickness)

cv2.imshow("Corners", image1)
cv2.waitKey(0)
cv2.imwrite('out.png', image1)
cv2.destroyAllWindows()

# raise NotImplementedError('`get_interest_points` function in ' +
# '`student_harris.py` needs to be implemented')
