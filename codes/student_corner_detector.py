import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pdb


def gradient(img):
    """
    实现numpy中的求梯度操作
    参考：https://docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html
    Args:
        img:图像

    Returns:(dy,dx):y和x向的导数
    """
    # scale = 1
    # delta = 0
    # ddepth = cv.CV_16S
    # # sobel的size可以修改以达到最优值
    # dx = cv.Sobel(img, ddepth, 1, 0, ksize=5, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    # dy = cv.Sobel(img, ddepth, 0, 1, ksize=5, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    dx = cv.Sobel(img, ddepth=cv.CV_16S, dx=1, dy=0, ksize=5)  # 计算梯度信息，使用高的数据类型
    dy = cv.Sobel(img, ddepth=cv.CV_16S, dx=0, dy=1, ksize=5)
    return dy, dx


def get_interest_points(img, feature_width=None):
    """
    For Harris corner detector:
        Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
        You can create additional interest point detector functions (e.g. MSER)
        for extra credit.（实现哈里斯角点检测。你可以创建额外的兴趣点检测函数来增加可靠性。）

        If you're finding spurious interest point detections near the boundaries,
        it is safe to simply suppress the gradients / corners near the edges of
        the image.(如果您在边界附近发现虚假的兴趣点，简单地抑制图像边缘附近的梯度/角点是更安全的。)

        Useful in this function in order to (a) suppress boundary interest
        points (where a feature wouldn't fit entirely in the image, anyway)
        or (b) scale the image filters being used. Or you can ignore it.
        (以下在函数中会很有用:a 抑制边界兴趣点(毕竟边界的角点不能代表图像) b 缩放滤波器的尺度（DOG+scale space）
        或者你可以忽略这步)
        By default you do not need to make scale and orientation invariant
        （默认你不需要做额外的尺度和方向不变操作）
        local features.

        Args:
        -   image: A numpy array of shape (m,n,c),
                    image may be grayscale of color (your choice)
        -   feature_width: (only for Harris) integer representing the local feature width in pixels.

        Returns:
        -   x: A numpy array of shape (N,) containing x-coordinates of interest points
        -   y: A numpy array of shape (N,) containing y-coordinates of interest points
        -   R_sort: A numpy array of shape (N,), descend sorted R,  correspondences to
                    x, y: R_sort[i] =   R[y[i],x[i]]

        please notice that cv2 returns the image with [h,w], which correspondes [y,x] dim respectively.
        ([vertically direction, horizontal direction])

    =================================================================================================
    For LoG detector :

        Implement the LoG corner detector to start with.

        Args:
        -   image: A numpy array of shape (m,n,c),
                    image may be grayscale of color (your choice)

        Returns:
        -   x_sort: A numpy array of shape (N,) containing x-coordinates of interest points
        -   y_sort: A numpy array of shape (N,) containing y-coordinates of interest points
        -   R_sort: A numpy array of shape (N,), descend sorted R,  correspondences to
                    x, y: R_sort[i] =   R[y[i],x[i]]

        The R_sort is the confidence for the selected interest point. If R_sort set to None, the code will
        automatically select the first 100 key points for evaluation,
        in this case, please return x_sort and y_sort with decreasing confidence.

        please notice that cv2 returns the image with [h,w], which correspondes [y,x] dim respectively.
        ([vertically direction, horizontal direction])
    """

    #############################################################################
    # TODO: YOUR HARRIS CORNER DETECTOR CODE HERE                                #
    # 判断是否为灰度图像，不是则转成灰度图
    if img.ndim == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # 高斯窗窗宽:feature_width
    if feature_width is None:
        feature_width = 15
    # 首先低通滤波除去噪声
    img = cv.GaussianBlur(img,(2*feature_width+1,feature_width*2+1),feature_width/3)

    # 求梯度
    dy, dx = gradient(img)

    # 归一化(防止溢出)
    # dy = dy.astype(float) / 255
    # dx = dx.astype(float) / 255

    # debug:
    plt.subplot(1, 2, 1)
    plt.imshow(dy)

    plt.subplot(1, 2, 2)
    plt.imshow(dx)
    plt.show()

    # 求Ixx,Iyy,Ixxy
    Ixx = dx ** 2
    Ixy = dy * dx
    Iyy = dy ** 2

    # 对每一点加高斯窗，计算M
    height = img.shape[0]
    width = img.shape[1]

    offset = feature_width // 2
    # alpha是计算R的参数，范围常取0.04-0.06，这里设为0.05
    alpha = 0.05
    # 设定threshold为结果R的阈值，小于阈值的点不要
    threshold = 1e-5
    # 存放兴趣点
    interestPoints = []
    kernel = cv.getGaussianKernel(feature_width, feature_width )
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            windowIxx = Ixx[y - offset:y + offset + 1, x - offset:x + offset + 1] * kernel
            windowIxy = Iyy[y - offset:y + offset + 1, x - offset:x + offset + 1] * kernel

            windowIyy = Ixy[y - offset:y + offset + 1, x - offset:x + offset + 1] * kernel
            # 求出M的三个部分的值
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()
            # 计算R
            det = (Sxx * Syy) - (Sxy ** 2)
            trace = Sxx + Syy
            r = det - alpha * (trace ** 2)
            if r > threshold:
                interestPoints.append((y, x, r))

    #############################################################################

    # raise NotImplementedError('`get_interest_points` function in ' +
    # '`student_harris.py` needs to be implemented')

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO: YOUR ADAPTIVE NON-MAXIMAL SUPPRESSION CODE HERE                     #
    # 排序
    f = lambda t: t[2]
    interestPoints.sort(key=f, reverse=True)
    i = 0
    radius = 30
    while i < len(interestPoints):
        j = i + 1
        while j < len(interestPoints):
            if np.linalg.norm(np.array(interestPoints[i][0:2]) - np.array(interestPoints[j][0:2])) < radius:
                interestPoints.pop(j)
            else:
                j = j + 1
        i = i + 1
    if len(interestPoints) > 100:
        interestPoints = interestPoints[:100]
    # return
    x_sort = []
    y_sort = []
    r_sort = []
    for point in interestPoints:
        x_sort.append(point[1])
        y_sort.append(point[0])
        r_sort.append(point[2])

    #############################################################################

    # raise NotImplementedError('adaptive non-maximal suppression in ' +
    # '`student_harris.py` needs to be implemented')

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    # return interestPoints
    return np.array(x_sort), np.array(y_sort), np.array(r_sort)
