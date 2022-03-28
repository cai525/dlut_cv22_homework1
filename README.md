 # <font color= "FF9900">dlut-cv作业1-哈里斯角点检测</font>

 -by Pengfei Cai,DLUT,2022.3.28

## 代码说明

 代码实现了HW1（见pdf）中的Harris角点检测任务。
 
## 文件夹说明

- codes存有代码（作业中该文件夹名是code,但用code会和pycharm的debug功能冲突）
- reference/ HarrisCorner-master存有网上的参考代码[https://github.com/hughesj919/HarrisCorner]
- 其他参见hw1

## 调整说明

- 照片大小对效果影响极大，因为照片尺寸会影响累加数目进而影响R值，建议将照片尺度统一
- code/student_corner_detector.py文件里的可修改参数：
 1. 阈值threshold：比较R决定是否为角点
 2. alpha：一般取0.04-0.06
 3. 非极大值抑制半径radius 
- code/main中可修改参数：feature_width，即高斯滤波器的大小，或者说是窗的大小

## 性能说明：
- 受参数影响大，准确率一般在0.7-0.8之间
- 主要的干扰是行人和周围景物，关键点可能会检到它们上
