import sys
import os
import numpy as np
from PIL import Image

input_dir = 'E:/Spatial Computing & Informatics Laboratory/CutTextArea/dataset/SynthNoText_OD_z14/'  # input directory
output_dir = '../od_z14_512/'
take_n_imgs = -1  # generate n output image. if -1, then generate all
image_size = 256
#neighbor_k = 3
neighbor_k = 2

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

all_imgs = os.listdir(input_dir)
all_imgs.sort()
print(len(all_imgs), 'images found in input directory')

if take_n_imgs < 0:
    take_n_imgs = len(all_imgs)

border_x_min = int(all_imgs[0].split('_')[1])
border_x_max = int(all_imgs[-1].split('_')[1])
border_y_min = int(all_imgs[0].split('_')[2])
#border_y_max = int(all_imgs[-1].split('_')[2])
border_y_max = 0

for i in range(0,len(all_imgs)):
    y=int(all_imgs[i].split('_')[2])
    border_y_max=max(border_y_max,y)



print('border_x_min',border_x_min)
print('border_x_max',border_x_max)
print('border_y_min',border_y_min)
print('border_y_max',border_y_max)

Arr = np.zeros((border_x_max - border_x_min + 1, border_y_max - border_y_min + 1))

#print(Arr.shape)

for img_name in all_imgs:
    idx_x = int(img_name.split('_')[1]) - border_x_min
    idx_y = int(img_name.split('_')[2]) - border_y_min

    #print(img_name.split('_')[1],img_name.split('_')[2])

    Arr[idx_x][idx_y] = 1

#assert np.count_nonzero(Arr) == len(all_imgs)

def paste_img(start_idx, start_jdx, k):
    '''provide starting index and concate range k'''
    new_img = Image.new('RGB', (image_size * k, image_size * k))
    for i in range(k):
        for j in range(k):
            #### specific format ####
            im = Image.open(input_dir + 'out_'+ str(start_idx + i) + '_' + str(start_jdx+j) + '_14.jpg')

            #图片是否存在
            if im!=None:
                new_img.paste(im, (i * image_size, j * image_size))
            else:
                return None

    return new_img

# for those has neighboring k blocks, concatenate
count = 0
break_flag = False
for idx in range(border_x_min, border_x_max + 1):
    if break_flag:
        break
    for jdx in range(border_y_min, border_y_max + 1):
        # check if all images exist to construct this patch
        arr_idx = idx - border_x_min
        arr_jdx = jdx - border_y_min
        if np.count_nonzero(Arr[arr_idx:arr_idx+neighbor_k,arr_jdx:arr_jdx+neighbor_k]) != neighbor_k*neighbor_k:
            continue
        else:
            new_img = paste_img(idx, jdx, neighbor_k)
            new_img.save(output_dir + '/concat'+str(neighbor_k) + '_'+ str(idx) + '_' + str(jdx)+ '.jpg')
            count += 1

            #
            print('count:',count)

            if count >= take_n_imgs:
                break_flag = True
                break


