import cv2
import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
from utils import *
from student_feature_matching import match_features
from student_sift import get_features
from student_corner_detector import get_interest_points

# ========================================register=====================================
# !!! Please replace the name & id with yours
name = 'goodLuck_cpf'
id = '201961204'  # student NO.

image2 = load_image('../data/checkerboard.png')
image2_bw = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
feature_width = 3  # width and height of each local feature, in pixels.
x2, y2, R2_confidence = get_interest_points(image2_bw, feature_width)
c2 = show_interest_points(image2, x2, y2)
plt.figure(3)
plt.imshow(c2)
plt.show()
