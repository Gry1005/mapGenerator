# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import cv2
import argparse
import glob
import math
from generator_dynamic_zekun_gry import SynthMap_DataGenerator_Centerline_Localheight_Dynamic

print ('finished loading weights')
image_root_path = 'E:/Spatial Computing & Informatics Laboratory/CutTextArea/dataset/concat_out/'
fonts_path = 'fonts/'
GB_path="GB.txt"
train_datagen = SynthMap_DataGenerator_Centerline_Localheight_Dynamic(image_root_path = image_root_path, fonts_path=fonts_path,GB_path=GB_path,batch_size= 8,  seed = 3333, mode = 'training',overlap=False) # ,showPicDir='No'
X,Y = train_datagen.next()

from matplotlib import pyplot as plt
for x in X:
    plt.figure(figsize = (10,10))
    plt.imshow(x[:,:,::-1])
    plt.show()