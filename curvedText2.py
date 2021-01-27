from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import text as mtext
from matplotlib import font_manager as fm
import matplotlib
import numpy as np
import math
from PIL import Image,ImageDraw,ImageFont,ImageFilter
import cv2
import glob
import os
import numpy
import random
import re

def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = numpy.fromstring(fig.canvas.tostring_argb(), dtype=numpy.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = numpy.roll(buf, 3, axis=2)
    return buf

def fig2img ( fig ):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )


def zk_addToMap(img_bg, text_layer, p):
    text_bi = np.array(text_layer)[:, :, 0].copy()
    text_bina = text_bi.copy()

    # binarizatoin. if text region, set to 1, if not text region, set to 0.
    # since no text region has color intensity greater than 128. thus we use 255 as the threshold
    # text_bi[text_bina != 255] = 255
    # text_bi[text_bina == 255] = 0

    #text_bi[text_bina <= 130] = 255
    #text_bi[text_bina > 130] = 0

    text_bi[text_bina <= 210] = 255
    text_bi[text_bina > 210] = 0

    text_bi = Image.fromarray(text_bi)
    text_bi.filter(ImageFilter.GaussianBlur(4))

    img_bg.paste(text_layer, (p[0], p[1]), mask=text_bi)
    return img_bg, text_bina

class CurvedText(mtext.Text):
    """
    A text object that follows an arbitrary curve.
    """
    def __init__(self, x, y, text, axes, **kwargs):
        super(CurvedText, self).__init__(x[0],y[0],' ', **kwargs)

        #增加一个记录每个字符center的数组
        self.xList=[]
        self.yList=[]

        axes.add_artist(self)

        ##saving the curve:
        self.__x = x
        self.__y = y
        self.__zorder = self.get_zorder()

        ##creating the text objects
        self.__Characters = []
        for c in text:
            if c == ' ':
                ##make this an invisible 'a':
                t = mtext.Text(0,0,'a')
                t.set_alpha(0.0)
            else:
                t = mtext.Text(0,0,c, **kwargs)

            #resetting unnecessary arguments
            t.set_ha('center')
            t.set_rotation(0)
            t.set_zorder(self.__zorder +1)

            self.__Characters.append((c,t))
            axes.add_artist(t)


    ##overloading some member functions, to assure correct functionality
    ##on update
    def set_zorder(self, zorder):
        super(CurvedText, self).set_zorder(zorder)
        self.__zorder = self.get_zorder()
        for c,t in self.__Characters:
            t.set_zorder(self.__zorder+1)

    def draw(self, renderer, *args, **kwargs):
        """
        Overload of the Text.draw() function. Do not do
        do any drawing, but update the positions and rotation
        angles of self.__Characters.
        """
        self.update_positions(renderer)

    def update_positions(self,renderer):
        """
        Update positions and rotations of the individual text elements.
        """

        #preparations

        ##determining the aspect ratio:
        ##from https://stackoverflow.com/a/42014041/2454357

        ##data limits
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        ## Axis size on figure
        figW, figH = self.axes.get_figure().get_size_inches()
        ## Ratio of display units
        _, _, w, h = self.axes.get_position().bounds
        ##final aspect ratio
        aspect = ((figW * w)/(figH * h))*(ylim[1]-ylim[0])/(xlim[1]-xlim[0])

        #points of the curve in figure coordinates:
        x_fig,y_fig = (
            np.array(l) for l in zip(*self.axes.transData.transform([
            (i,j) for i,j in zip(self.__x,self.__y)
            ]))
        )

        #point distances in figure coordinates
        x_fig_dist = (x_fig[1:]-x_fig[:-1])
        y_fig_dist = (y_fig[1:]-y_fig[:-1])
        r_fig_dist = np.sqrt(x_fig_dist**2+y_fig_dist**2)

        #arc length in figure coordinates
        l_fig = np.insert(np.cumsum(r_fig_dist),0,0)

        #angles in figure coordinates
        rads = np.arctan2((y_fig[1:] - y_fig[:-1]),(x_fig[1:] - x_fig[:-1]))
        degs = np.rad2deg(rads)


        rel_pos = 10
        for c,t in self.__Characters:
            #finding the width of c:
            t.set_rotation(0)
            t.set_va('center')
            bbox1  = t.get_window_extent(renderer=renderer)
            w = bbox1.width
            h = bbox1.height

            #ignore all letters that don't fit:
            if rel_pos+w/2 > l_fig[-1]:
                t.set_alpha(0.0)
                rel_pos += w
                continue

            elif c != ' ':
                t.set_alpha(1.0)

            #finding the two data points between which the horizontal
            #center point of the character will be situated
            #left and right indices:
            il = np.where(rel_pos+w/2 >= l_fig)[0][-1]
            ir = np.where(rel_pos+w/2 <= l_fig)[0][0]

            #if we exactly hit a data point:
            if ir == il:
                ir += 1

            #how much of the letter width was needed to find il:
            used = l_fig[il]-rel_pos
            rel_pos = l_fig[il]

            #relative distance between il and ir where the center
            #of the character will be
            fraction = (w/2-used)/r_fig_dist[il]

            ##setting the character position in data coordinates:
            ##interpolate between the two points:

            #每个字符的坐标
            x = self.__x[il]+fraction*(self.__x[ir]-self.__x[il])
            y = self.__y[il]+fraction*(self.__y[ir]-self.__y[il])

            #getting the offset when setting correct vertical alignment
            #in data coordinates
            t.set_va(self.get_va())
            bbox2  = t.get_window_extent(renderer=renderer)

            bbox1d = self.axes.transData.inverted().transform(bbox1)
            bbox2d = self.axes.transData.inverted().transform(bbox2)
            dr = np.array(bbox2d[0]-bbox1d[0])

            #the rotation/stretch matrix
            rad = rads[il]
            rot_mat = np.array([
                [math.cos(rad), math.sin(rad)*aspect],
                [-math.sin(rad)/aspect, math.cos(rad)]
            ])

            ##computing the offset vector of the rotated character
            drp = np.dot(dr,rot_mat)

            #setting final position and rotation:
            t.set_position(np.array([x,y])+drp)
            t.set_rotation(degs[il])

            #记录字符中心点位置
            self.xList.append(x+drp[0])
            self.yList.append(y+drp[1])

            t.set_va('center')
            t.set_ha('center')

            #updating rel_pos to right edge of character
            rel_pos += w-used




def arbitraryTextGene(curType,fontPath,fontSize,rAngle,left_up,content,pic,test_overlap_img,txtfile,fontcolor):

    fig = plt.figure(figsize=(4,4), dpi=100)

    length = 400

    ax=fig.add_axes([0, 0, 1, 1])

    #去除边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    N = 100

    curves=[
            [ -np.cos(np.linspace(0,2*np.pi,N)),
              np.sin(np.linspace(0,2*np.pi,N))],
            [
                np.linspace(-1, 1, N),
                np.linspace(-1, 1, N),
            ],
            [
                np.linspace(-1, 1, N),
                np.sin(np.linspace(0, 2 * np.pi, N)),
            ]
        ]

    #cindex=0
    cindex = curType

    curve=curves[cindex]

    #text='you even can annotate parametric curves'
    text = content

    # plotting the curve
    ax.plot(*curve, color='b',alpha=0.0)

    #ax.scatter([-1,1],[-1,1],color='r')

    # adjusting plot limits
    stretch = 0.2
    xlim = ax.get_xlim()
    w = xlim[1] - xlim[0]
    ax.set_xlim([xlim[0] - stretch * w, xlim[1] + stretch * w])
    ylim = ax.get_ylim()
    h = ylim[1] - ylim[0]
    ax.set_ylim([ylim[0] - stretch * h, ylim[1] + stretch * h])

    fontsize=fontSize

    #fontProperty
    #fp= fontPath

    fp=fm.FontProperties(fname=fontPath)

    #fontcolor
    fcolor = tuple(a / 255. for a in fontcolor)

    # adding the text
    ctext = CurvedText(
        x=curve[0],
        y=curve[1],
        text=text,  # 'this this is a very, very long text',
        va='center',
        axes=ax,  ##calls ax.add_artist in __init__
        fontsize=fontsize,
        fontproperties=fp,
        color=fcolor
    )

    #plt.show()

    plt.ion()


    #print(ctext.xList,ctext.yList)

    #paste text to image
    text_layer = fig2img(fig)

    #
    plt.pause(0.01)
    plt.close()

    #text_layer.show()

    xList=[]

    yList=[]


    for i in range(0, len(ctext.xList)):
        xList.append(ctext.xList[i] * 0.66)
        yList.append(ctext.yList[i] * 0.66)


    for i in range(0, len(ctext.xList)):
        xList[i] = xList[i] + 1
        yList[i] = yList[i] + 1


    #print(xList)

    #print(yList)

    for i in range(0,len(ctext.yList)):
        yList[i]=2-yList[i]


    for i in range(0, len(ctext.xList)):
        xList[i] = int((length/2) * xList[i])
        yList[i] = int((length/2) * yList[i])

    #transform to cv2
    text_img = cv2.cvtColor(numpy.asarray(text_layer), cv2.COLOR_RGB2BGR)

    (h, w) = text_img.shape[:2]  # 10
    center = (w // 2, h // 2)  # 11

    angle=rAngle

    M = cv2.getRotationMatrix2D(center, angle, 1.0)  # 12
    rotated = cv2.warpAffine(text_img, M, (w, h),borderValue=(255,255,255))  # 13
    #cv2.imshow("Rotated by 45 Degrees", rotated)  # 14

    #旋转centerline的坐标
    pointList=[]

    for i in range(0, len(ctext.xList)):
        pointList.append([xList[i],yList[i]])

    for i in range(0, len(pointList)):
        P=pointList[i]
        pointList[i] = np.dot(M,np.array([[P[0]],[P[1]],[1]]))

    # draw center points 注意center points里的点有两倍

    # pointList里的点有两倍，截取一半
    pointList=pointList[0:int(len(pointList)/2)]

    #print("len pointList:",len(pointList))
    #for i in range(0, len(pointList)):
        #cv2.circle(rotated, (pointList[i][0],pointList[i][1]), 2, (0,0,255), -1)


    #cv2.imshow('PIL',rotated)

    #cv2.waitKey()

    finalText = Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))

    #patch_path='E:/Spatial Computing & Informatics Laboratory/CutTextArea/dataset/synth_osm_od_z14/out_8032_5299_14.jpg'

    #img_bg_pil = Image.open(patch_path).convert('RGBA')



    #add text area
    #area_img = np.zeros((length, length, 3), np.uint8)

    #bound_img = np.zeros((length, length, 3), np.uint8)

    '''
    for i in range(0,length):
        for j in range(0,length):
            #area_img[i][j]=[255,255,255]
            bound_img[i][j] = [255, 255, 255]

    for i in range(0, len(pointList)):
        cv2.circle(area_img, (pointList[i][0],pointList[i][1]), int(fontsize*0.85), (0,0,255), -1)

    #cv2.imshow('',area_img)

    #cv2.waitKey()

    img_gray = cv2.cvtColor(area_img, cv2.COLOR_BGR2GRAY)

    #cv2.imshow('',img_gray)

    #cv2.waitKey()

    contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    

    #从圆上抽取点

    
    for i in range(0,len(contours[0])):
        if i%10==0:
            poly.append([contours[0][i][0][0],contours[0][i][0][1]])

    poly = np.array([poly],dtype = np.int32)
    print(poly)
    
    

    #cv2.drawContours(bound_img, contours, -1, (0, 0, 0), 1)
    
    '''

    poly = []

    #find the poly using centerline point

    #pointList需要做坐标偏移
    for i in range(0,len(pointList)):
        pointList[i][0]=pointList[i][0]+left_up[0]
        pointList[i][1] = pointList[i][1] + left_up[1]

    shift=fontsize*0.8

    for i in range(0,len(pointList)):
        if i<len(pointList)-1:
            vy=pointList[i+1][1]-pointList[i][1]
            vx=pointList[i+1][0]-pointList[i][0]
            sxy=math.sqrt(vy*vy+vx*vx)
            x1=pointList[i][0]-shift*(vy/sxy)
            y1=pointList[i][1]+shift*(vx/sxy)
            poly.append([x1,y1])

        else:
            vy = pointList[i][1] - pointList[i-1][1]
            vx = pointList[i][0] - pointList[i-1][0]
            sxy = math.sqrt(vy * vy + vx * vx)
            x1 = pointList[i][0] - shift * (vy / sxy)
            y1 = pointList[i][1] + shift * (vx / sxy)
            poly.append([x1, y1])


    for j in range(0,len(pointList)):
        i=len(pointList)-1-j
        #print("i:",i)
        if i<len(pointList)-1:
            vy=pointList[i+1][1]-pointList[i][1]
            vx=pointList[i+1][0]-pointList[i][0]
            sxy=math.sqrt(vy*vy+vx*vx)
            x2=pointList[i][0]+shift*(vy/sxy)
            y2=pointList[i][1]-shift*(vx/sxy)
            poly.append([x2, y2])
        else:
            vy = pointList[i][1] - pointList[i-1][1]
            vx = pointList[i][0] - pointList[i-1][0]
            sxy = math.sqrt(vy * vy + vx * vx)
            x2 = pointList[i][0] + shift * (vy / sxy)
            y2 = pointList[i][1] - shift * (vx / sxy)
            poly.append([x2, y2])


    #draw points
    #for i in range(10,20):
        #print(poly[i][0], poly[i][1])
        #cv2.circle(bound_img, (poly[i][0], poly[i][1]), 2, (0, 0, 255), -1)

    #print(poly)



    #用poly判断文本区域和别的文本区域是否有重叠

    poly = np.array([poly], dtype=np.int32)

    pointList = np.array([pointList], dtype=np.int32)

    #test_overlap_img_copy=np.zeros([512,512], dtype = "uint8")

    test_overlap_img_copy=test_overlap_img.copy()

    #print(test_overlap_img.shape)

    overlap_img_2=np.zeros([test_overlap_img.shape[0],test_overlap_img.shape[1]], dtype = "uint8")

    #for i in range(0,512):
        #for j in range(0,512):
            #test_overlap_img_copy[i][j]=test_overlap_img[i][j]

    cv2.fillPoly(overlap_img_2, poly, 10)

    test_overlap_img_copy=test_overlap_img_copy+overlap_img_2

    and_area=np.sum(np.float32(np.greater(test_overlap_img_copy,10)))

    #如果有重合部分
    if and_area >1.0:
        print('有重合文本！无法粘贴！')
        return test_overlap_img





    #把poly的值放入txt文件
    content=''

    for i in range(0,len(poly[0])):
        if i!=len(poly[0])-1:
            content=content+str(poly[0][i][0].item())+','+str(poly[0][i][1].item())+';'
        else:
            content = content + str(poly[0][i][0].item()) + ',' + str(poly[0][i][1].item())

    #把center points放入txt文件

    content=content+'/'

    for i in range(0,len(pointList[0])):
        if i!=len(pointList[0])-1:
            content=content+str(pointList[0][i][0].item())+','+str(pointList[0][i][1].item())+';'
        else:
            content = content + str(pointList[0][i][0].item()) + ',' + str(pointList[0][i][1].item())

    #把local height记录
    content=content+'/'+str(shift)

    #print(content)

    txtfile.writelines(content+'\n')


    #从字符串解析
    polyStr=content.split('/')[0]
    pointListStr = content.split('/')[1]
    localheight=content.split('/')[2]

    polyStr = polyStr.split(';')
    pointListStr = pointListStr.split(';')

    poly2=[]
    pointList2=[]

    for i in range(0,len(polyStr)):
        x=float(polyStr[i].split(',')[0])
        y=float(polyStr[i].split(',')[1])
        poly2.append([x,y])

    for i in range(0,len(pointListStr)):
        x1=float(pointListStr[i].split(',')[0])
        y1=float(pointListStr[i].split(',')[1])
        pointList2.append([x1,y1])

    #print(poly2)
    #print(pointList2)
    #print('localheight:'+localheight)

    poly2 = np.array([poly2], dtype=np.int32)

    pointList2 = np.array([pointList2], dtype=np.int32)






    #获取背景图路径
    img_bg_pil = pic

    # 文字粘贴上图片
    # resultImg,textbina=zk_addToMap(img_bg_pil,finalText,(0,0))
    resultImg, textbina = zk_addToMap(img_bg_pil, finalText, left_up)

    # 把poly和centerline画在图上


    cv2_result_img = cv2.cvtColor(numpy.asarray(resultImg), cv2.COLOR_RGB2BGR)

    cv2.polylines(cv2_result_img, poly2, True,(255,0,0),1)

    #cv2.fillPoly(cv2_result_img, poly2, (255, 0, 0))

    cv2.polylines(cv2_result_img, pointList2,False, (0,0,255),1)

    cv2.imshow('resultImg',cv2_result_img)

    cv2.waitKey()
    


    #print(poly)

    #draw the lines
    #for i in range(0,len(poly)-1):
        #cv2.line(bound_img,(poly[i][0],poly[i][1]),(poly[i+1][0],poly[i+1][1]),(255,0,0))

    #cv2.polylines(bound_img, poly, 1,255)

    #cv2.fillPoly(bound_img,poly,(255,0,0))

    #cv2.imshow('bound_img',bound_img)

    #cv2.waitKey()

    #bound_img = Image.fromarray(cv2.cvtColor(bound_img, cv2.COLOR_BGR2RGB))

    #粘贴边框
    #resultImg, textbina1 = zk_addToMap(resultImg, bound_img, (0, 0))
    #resultImg, textbina1 = zk_addToMap(resultImg, bound_img, left_up)

    #resultImg.show()

    return test_overlap_img_copy

if __name__ == '__main__':

    #patch_path = 'E:/Spatial Computing & Informatics Laboratory/CutTextArea/dataset/synth_osm_od_z14/out_8032_5299_14.jpg'
    #patch_path = 'E:/Spatial Computing & Informatics Laboratory/CutTextArea/dataset/od_z14_512/concat2_8055_5337.jpg'

    # load the words
    word_set = set()
    geoname_f = open("GB.txt", "r",encoding='utf-8')
    for line in geoname_f:
        cols = re.split(r'\t+', line.rstrip('\t'))
        words = cols[1].split(' ')
        for w in words:
            word_set.add(w.strip('()'))
    geoname_f.close()
    set_len = len(word_set)
    word_set.remove('')
    print(len(word_set), ' words in total')
    print('eg:', list(word_set)[0:10])

    word_list=list(word_set)

    #fonts
    fonts_path='fonts'
    fonts = glob.glob(fonts_path + '/*.ttf')
    print(len(fonts),' fonts in total')

    #background image path
    bg_images = glob.glob('E:/Spatial Computing & Informatics Laboratory/CutTextArea/dataset/concat_out/*.jpg')

    count=0

    for patch_path in bg_images[0:10]:

        base_name = os.path.basename(patch_path)

        #print('base_name:',base_name)

        txt_name = '../od_z16_768_test/' + base_name[0:len(base_name) - 4] + '.txt'

        f = open(txt_name, 'w+')

        bg_img = Image.open(patch_path).convert('RGB')

        test_overlap_img=np.zeros([768,768], dtype = "uint8")

        #test_overlap_img=arbitraryTextGene(curType=0,fontPath='fonts/DAYROM__.ttf',fontSize=20,rAngle=-25,left_up=(10,10),content=' Bovard ',pic=bg_img,test_overlap_img=test_overlap_img,txtfile=f)
        #test_overlap_img = arbitraryTextGene(curType=0,fontPath='fonts/DAYROM__.ttf', fontSize=20, rAngle=-45, left_up=(10, 10),content=' Bovard ', pic=bg_img, test_overlap_img=test_overlap_img)
        #test_overlap_img = arbitraryTextGene(curType=1, fontPath='fonts/Essays1743.ttf', fontSize=25, rAngle=-90,left_up=(10, 10),content=' Leavy ', pic=bg_img, test_overlap_img=test_overlap_img)
        #test_overlap_img = arbitraryTextGene(curType=0,fontPath='fonts/TheanoOldStyle-Regular.ttf', fontSize=30, rAngle=-90, left_up=(20, 20),content=' Dornsife ', pic=bg_img, test_overlap_img=test_overlap_img)

        # 一个图片加10次文字，50%直，50%弯曲
        for i in range(0,10):
            # font-size range:18-40
            fs=random.randint(18,40)

            # 旋转角度-180~180；-145~35
            ra=random.randint(-145,35)

            #左上角坐标
            #left=random.randint(0,111)
            left = random.randint(0, 367)

            #up=random.randint(0,111)
            up = random.randint(0, 367)

            #curveType
            cur=random.randint(0,2)

            #color: 0~128
            fcolor = random.randint(0,128) * np.ones((3))
            #fcolor=(40,40,40)

            #content
            content=' ' + word_list[random.randint(0, len(word_list) - 1)] + ' '
            #content=' Bovard '

            if cur==2:
                test_overlap_img = arbitraryTextGene(curType=1,
                                                     fontPath=fonts[random.randint(0, len(fonts) - 1)], fontSize=fs,
                                                     rAngle=ra, left_up=(left, up), content=content, pic=bg_img, test_overlap_img=test_overlap_img,
                                                     txtfile=f,fontcolor=fcolor)
            else:
                test_overlap_img = arbitraryTextGene(curType=0,
                                                     fontPath=fonts[random.randint(0, len(fonts) - 1)], fontSize=fs,
                                                     rAngle=ra, left_up=(left, up), content=content, pic=bg_img, test_overlap_img=test_overlap_img,
                                                     txtfile=f,fontcolor=fcolor)


        f.close()

        bg_img.save('../od_z16_768_test/'+base_name)

        #Image.open(bg_img)

        count=count+1

        print('count:',count)



