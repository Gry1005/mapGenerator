import glob
import os
import cv2
import numpy as np

image_list=glob.glob('../OS_10_80_Grey/*.jpg')

txt_dir='../OS_10_80_Grey/'

std_size=512

for image_path in image_list[0:5]:

    base_name = os.path.basename(image_path)

    print('base_name:',base_name)

    txt_path = txt_dir + base_name[0:len(base_name) - 4] + '.txt'

    image_x = cv2.imread(image_path)
    o_height, o_width = image_x.shape[0], image_x.shape[1]
    image_x = cv2.resize(image_x, (std_size, std_size))

    h_sca = std_size * 1. / o_height
    w_sca = std_size * 1. / o_width

    height, width = image_x.shape[0], image_x.shape[1]
    # print 'height,width',height,width
    #x = x / 255.
    #batch_X.append(x)
    # init variables
    prob_img = np.zeros((height, width, 3), np.uint8)  # regression target for y1 (segmentation task)
    border_img = np.zeros((height, width, 3), np.uint8)  # regression target for y1 (segmentation task)
    centerline_img = np.zeros((height, width, 3), np.uint8)  # regression target for y1 (segmentation task)

    # 加入一张新的结果图片：local height

    localheight_img = np.zeros((height, width, 3), np.uint8)

    inside_Y_regress = np.zeros((height, width, 6), np.float32)
    #border_p = self.border_percent
    #centers = []
    #thetas = []
    #whs = []

    # read and precess file: 读取数据要修改
    with open(txt_path, 'r') as f:
        data = f.readlines()

    bbox_idx = 0
    for line in data:

        bbox_idx += 1

        polyStr = line.split('/')[0]
        pointListStr = line.split('/')[1]
        localheight = line.split('/')[2].split('\n')[0]

        polyStr = polyStr.split(';')
        pointListStr = pointListStr.split(';')

        poly2 = []
        pointList2 = []

        for i in range(0, len(polyStr)):
            x = float(polyStr[i].split(',')[0])*h_sca
            y = float(polyStr[i].split(',')[1])*w_sca
            poly2.append([x, y])

        for i in range(0, len(pointListStr)):
            x1 = float(pointListStr[i].split(',')[0])*h_sca
            y1 = float(pointListStr[i].split(',')[1])*w_sca
            pointList2.append([x1, y1])

        # print(poly2)
        # print(pointList2)
        # print('localheight:'+localheight)

        poly2 = np.array([poly2], dtype=np.int32)

        pointList2 = np.array([pointList2], dtype=np.int32)

        # create mask image
        # points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        # print (points)
        # 填充多边形
        # cv2.fillConvexPoly(prob_img, points.astype(np.int), (0, bbox_idx, 0))
        cv2.fillPoly(image_x, poly2, (0, 255, 0))
        # centers.append([x_c, y_c])

        cv2.polylines(image_x, poly2, True, (255, 0, 0), 2)

        # cv2.line(border_img, points[0], points[1], (0, 1, 0), thickness)
        # cv2.line(border_img, points[1], points[2], (0, 1, 0), thickness)
        # cv2.line(border_img, points[2], points[3], (0, 1, 0), thickness)
        # cv2.line(border_img, points[3], points[0], (0, 1, 0), thickness)

        # create centerline image
        # cv2.line(centerline_img, left_pt, right_pt, (0, 1, 0), thickness * 2)
        cv2.polylines(image_x, pointList2, False, (0, 0, 255), 1)

        # create localheight image
        # cv2.line(localheight_img, left_pt, right_pt, (lh, 0, 0), thickness * 2)
        # print('localheight:',float(localheight))
        #cv2.polylines(localheight_img, pointList2, False, (float(localheight), 0, 0), 4)

        cv2.imshow('x:',image_x)

        cv2.waitKey()