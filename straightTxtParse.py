#把txt文件解析为pixel mask以及图片标注
import glob
import os
import cv2
import numpy as np

image_list=glob.glob('../concat_out_bigF_sChar/*.jpg')

txt_folder='../concat_out_bigF_sChar/'

outputFolder='testOutput/'

for image_path in image_list:

    base_name = os.path.basename(image_path)

    print('base_name:',base_name)

    #txt_path = 'original_size_OS_USGS/' + base_name[0:len(base_name) - 4] + '.txt'
    txt_path = txt_folder + base_name[0:len(base_name) - 4] + '.txt'

    if not os.path.exists(txt_path):
        continue

    image_x = cv2.imread(image_path)

    height=image_x.shape[0]

    width=image_x.shape[1]

    with open(txt_path, 'r') as f:
        data = f.readlines()

    #bbox_idx = 0

    polyList=[]

    for line in data:

        polyStr=line.split(',')

        poly=[]

        for i in range(0,len(polyStr)):
            if i%2==0:
                poly.append([float(polyStr[i]),float(polyStr[i+1])])

        polyList.append(poly)

    print('all: ',len(polyList))

    txt_pixel_result = np.zeros((height, width, 3), np.uint8)

    for i in range(0,len(polyList)):

        polyPoints=np.array([polyList[i]],dtype=np.int32)

        cv2.polylines(image_x, polyPoints, True, (0, 0, 255), 1)

        cv2.fillPoly(txt_pixel_result,polyPoints,(0,255,0))

        print('i: ',i)

    #cv2.imwrite(outputFolder+'parse_result_'+base_name[0:len(base_name) - 4] + '.jpg',image_x)
    cv2.imshow('result',image_x)
    cv2.waitKey()

    #cv2.imwrite(outputFolder+'masked_' + base_name[0:len(base_name) - 4] + '.jpg', txt_pixel_result)