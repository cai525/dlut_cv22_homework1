## [imports]
import cv2 as cv
import sys

## [imports]
## [imread]
# opencv不能读取中文路径
img = cv.imread("example1.png")
## [imread]
## [empty]
if img is None:
    sys.exit("Could not read the image.")
## [empty]
## [imshow]

cv.namedWindow('Display window', 0)
cv.imshow("Display window", img)
k = cv.waitKey(0)
## [imshow]
## [imsave]
if k == ord("s"):
    cv.imwrite("starry_night.png", img)
## [imsave]
