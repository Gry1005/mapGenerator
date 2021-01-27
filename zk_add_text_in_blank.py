import os
import glob
from PIL import Image,ImageDraw,ImageFont,ImageFilter
import numpy as np
import random,math,copy
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import text as mtext
from matplotlib import font_manager
import cv2
import re

#map4_path = '/home/zekun/Documents/Dornsife/SynthMaps/concat_out/'
#save_path = '/home/zekun/Documents/Dornsife/SynthMaps/concat_out_text/'
#fonts_path = '/home/zekun/Documents/Dornsife/SynthMaps/fonts/'

map4_path = r'E:/Spatial Computing & Informatics Laboratory/CutTextArea/dataset/concat_out/'
save_path = r'../zekun_test/'
fonts_path = r'fonts/'

# load the words
word_set = set()
geoname_f = open("GB.txt", "r", encoding='UTF-8')
for line in geoname_f:
    cols = re.split(r'\t+', line.rstrip('\t'))
    words = cols[1].split(' ')
    for w in words:
        word_set.add(w.strip('()'))
geoname_f.close()
set_len = len(word_set)
print (len(word_set), ' words in total')
print ('eg:', list(word_set)[0:10])

# Notes for inserting text on images: 
# if no rotation or customized font is required, then cv2.putText and cv2.getTextsize would work perfectly
# But one reason NOT to use  cv2 to insert text is its incapability to insert customized font styles
# even loadFontData did not work for me in python


def fig2data ( fig ):
    # from http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
    """
    @brief Convert a Matplotlib figure to a 3D numpy array with RGB channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGB values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.frombuffer ( fig.canvas.tostring_rgb(), dtype=np.uint8 )
    buf.shape = ( w, h,3 )
 
    # canvas.tostring_argb give pixmap in RGB mode. Roll the ALPHA channel to have it in RGB mode
    #buf = np.roll ( buf, 3, axis = 2 )
    return buf

def fig2img ( fig ):
    # from http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGB format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.frombuffer( "RGB", ( w ,h ), buf.tostring( ) ).transpose(Image.FLIP_TOP_BOTTOM)

def visualize_points(text_layer, pts):
    '''
    text_layer is of PIL image type
    pts is an array
    '''
    text_layer = np.array(text_layer).astype(np.uint8).copy()
    
    for pt in pts:
        pt = [int(a) for a in pt]
        cv2.circle(text_layer,(pt[0],pt[1]), 5, (255,0,0),10)
    plt.imshow(text_layer)
    plt.show()

# generate text layer
def text_on_canvas(text, myf, ro, color = (0.5,0.5,0.5), margin = 1):
    axis_lim = 1
    
    fig = plt.figure(figsize = (5,5), dpi=100)
    plt.axis([0, axis_lim, 0, axis_lim])
    
    # place the top left corner at (axis_lim/20,axis_lim/2) to avoid clip during rotation
    aa = plt.text(axis_lim/20.,axis_lim/2., text, color = color, ha='left', va = 'top', fontproperties = myf, rotation = ro, wrap=False)
    plt.axis('off')
    text_layer = fig2img(fig) # convert to image
    plt.close()
    
    we = aa.get_window_extent()
    min_x, min_y, max_x, max_y = we.xmin, 500 - we.ymax, we.xmax, 500 - we.ymin
    box = (min_x-margin, min_y-margin, max_x+margin, max_y+margin)
    
    # return coordinates to further calculate the bbox of rotated text
    return text_layer, min_x, min_y, max_x, max_y 



def geneText(text, font_family, font_size, font_color, rot_angle, style):
    # if font size too big, then put.text automatically adjust the position, which makes computed position errorous.
    myf = font_manager.FontProperties(fname=font_family, size=font_size)

    if style < 8: # rotated text
        fcolor = tuple(a/255. for a in font_color) # convert from [0,255] to [0,1]
        # no rotation, just to get the minimum bbox
        htext_layer, min_x, min_y, max_x, max_y = text_on_canvas(text, myf, ro = 0, color = fcolor)
    
        #print min_x,min_y,max_x,  max_y
        M = cv2.getRotationMatrix2D((min_x,min_y),rot_angle,1)
        # pts is 4x3 matrix
        pts = np.array([[min_x, min_y, 1],[max_x, min_y, 1],[max_x, max_y, 1],[min_x, max_y,1]]) # clockwise
        affine_pts = np.dot(M, pts.T).T

        if (affine_pts<= 0).any()  or (affine_pts>= 500).any() :
            return 0, 0 # exceed boundary. skip
        else:
            text_layer = htext_layer.rotate(rot_angle,center=(min_x, min_y),fillcolor='white')

            #visualize_points(htext_layer, pts)
            #visualize_points(text_layer, affine_pts)
            return text_layer, affine_pts
    
    else:
        raise NotImplementedError

    
# add to map
def zk_addToMap(img_bg,text_layer, p):
    text_bi = np.array(text_layer)[:,:,0].copy()
    text_bina = text_bi.copy()
    
    # binarizatoin. if text region, set to 1, if not text region, set to 0. 
    # since no text region has color intensity greater than 128. thus we use 255 as the threshold
    #text_bi[text_bina != 255] = 255
    #text_bi[text_bina == 255] = 0  
    text_bi[text_bina <= 130] = 255
    text_bi[text_bina > 130] = 0  
    text_bi = Image.fromarray(text_bi)
    text_bi.filter(ImageFilter.GaussianBlur(4))
    
    img_bg.paste(text_layer,(p[0],p[1]),mask=text_bi)
    return img_bg,text_bina
# process
word_set = list(word_set)

fonts = glob.glob(fonts_path + '/*.ttf')
map4_set = glob.glob(map4_path + '/*.jpg')

words_size = len(word_set)
cnt = 0 # process on #cnt images
text_num_thresh = 5


for i,patch_path in enumerate(map4_set):
    print ('processing', patch_path)
    img_bg_pil = Image.open(patch_path).convert('RGBA')
    W, H = img_bg_pil.size
    img_bg_cv = cv2.imread(patch_path)
    with open(save_path +os.path.basename(patch_path)[:-4]+'.txt', 'w') as f:
        1
    info_f = open(save_path +os.path.basename(patch_path)[:-4]+'.txt', 'w')
    

    #for i in range(text_num):
    text_num = 0
    text_region = []
    while(text_num < text_num_thresh):
        # get input text string
        text = word_set[random.randint(1, words_size-1)].strip()
        text = re.sub('[^0-9A-Za-z]+', '', text) # remove symbols
        while(len(text) < 1): # text length is zero. no character except for spaces
            text = word_set[random.randint(1, words_size-1)].strip()
            text = re.sub('[^0-9A-Za-z]+', '', text) # remove symbols
            
        # font specification
        font_face = fonts[random.randint(0, len(fonts)-1)]
        font_size = random.randint(10, 80)
        ro = random.randint(-90, 90)
        fcolor = random.randint(0, 128)* np.ones((3))
        
        # some variations to the original input text
        if np.random.randint(0,2): # 50% chance to capitalize the text
            text = text.upper()
           
        
        # 50% chance to insert blank space
        if np.random.randint(0,2): 
            insert_type = np.random.randint(0,3)
            if insert_type == 0: # 1/3 of the chance to insert ONE blank space between chars
                text = " ".join(text)
            if insert_type == 1: #1/3 of the chance to insert TWO blank space bween chars
                text = "  ".join(text)
            if insert_type == 2: #1/3 of the chance to insert FIVE blank space bween chars
                text = "     ".join(text)
                
        #print text
        text_layer, af_pts = geneText(text, font_face, font_size, font_color = fcolor, rot_angle = ro, style = 1)

        # text region 

        if text_layer != 0:
            text_w, text_h = text_layer.size
            
            
            while True:
                breakflag = True
                upper_left_pos = (random.randint(0, W - text_w/2), random.randint(0, H - text_h / 2)) # upper-left pos on bg img
                # these are the text region positions on original text layer
                left = int(np.min(af_pts[:,0]))
                up = int(np.min(af_pts[:,1]))
                right = int(np.max(af_pts[:,0]))
                bottom = int(np.max(af_pts[:,1]))

                x1, y1 = af_pts[0]
                x2, y2 = af_pts[1]
                x3, y3 = af_pts[2]
                x4, y4 = af_pts[3]
                x1, x2 = int(x1 - left + upper_left_pos[0]), int(x2 - left + upper_left_pos[0])
                x3, x4 = int(x3 - left + upper_left_pos[0]), int(x4 - left  + upper_left_pos[0])
                y1, y2 = int(y1 - up + upper_left_pos[1]), int(y2 - up + upper_left_pos[1])
                y3, y4 = int(y3 - up + upper_left_pos[1]), int(y4 - up + upper_left_pos[1])
                bbox = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

                if len(text_region) == 0:
                    im = np.zeros([W,H], dtype = "uint8")
                    text_region.append(bbox)
                    mask = cv2.fillConvexPoly(im, np.array(bbox), 10)
                    break
                else:
                    mask_copy = mask
                    im1 = np.zeros([W,H], dtype = "uint8")
                    mask1 = cv2.fillConvexPoly(im1,np.array(bbox),10)
                    masked_and = mask_copy + mask1
                    and_area =np.sum(np.float32(np.greater(masked_and,10))) #use and_are to check if masked_and has overlap area
                    if and_area > 1.0 :
                        continue
                    elif x1>H or x2>H or x3>H or x4>H or y1>W or y2>W or y3>W or y4>W:# not exceed the boundary
                        continue
                    else:
                        text_region.append(bbox)
                        mask = mask + mask1
                        break
            text_layer = text_layer.crop(( left, up, right, bottom )) # crop out the text region
            img_bg_pil,_ = zk_addToMap(img_bg_pil, text_layer, upper_left_pos ) # place on the bg image


            # calculate the tight bbox position on the bg image ( infer from original text layer)
            [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] = text_region[-1]
            
            #draw =ImageDraw.Draw(img_bg_pil)
            #draw.line(((x2,y2),(x3,y3)), fill=128)
            c_x, c_y = 0.5 * (x1 + x3), 0.5 * (y1 + y3)

            #info_f.write('%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n' % ( x1, y1, x2, y2, x3, y3, x4, y4, c_x, c_y))
            info_f.write('%d,%d,%d,%d,%d,%d,%d,%d\n' % (x1, y1, x2, y2, x3, y3, x4, y4))

            text_num += 1


        img_bg_pil.convert('RGB').save(save_path +os.path.basename(patch_path))
        
        
    info_f.close()
        
    cnt += 1
    if cnt == 5:
        break
    
print ('done processing')     
