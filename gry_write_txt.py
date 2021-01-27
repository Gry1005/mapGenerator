from matplotlib import font_manager
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
def fig2data(fig):
    # from http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
    """
    @brief Convert a Matplotlib figure to a 3D numpy array with RGB channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGB values
    """
    # draw the renderer
    fig.canvas.draw()
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (w, h, 3)
    # canvas.tostring_argb give pixmap in RGB mode. Roll the ALPHA channel to have it in RGB mode
    #buf = np.roll ( buf, 3, axis = 2 )
    return buf
def fig2img(fig):
    # from http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGB format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, d = buf.shape
    return Image.frombuffer( "RGB", ( w ,h ), buf.tostring( ) )
# generate text layer
def text_on_canvas(text, myf, ro, color=(0.5, 0.5, 0.5), margin=1):
    axis_lim = 1
    fig = plt.figure(figsize=(5, 5), dpi=100)
    plt.axis([0, axis_lim, 0, axis_lim])
    # place the top left corner at (axis_lim/20,axis_lim/2) to avoid clip during rotation

    #aa = plt.text(axis_lim / 20., axis_lim / 2., text, color=color, ha='left', va='top', fontproperties=myf,rotation=ro, wrap=False)
    aa = plt.text(axis_lim / 20., axis_lim / 2., text, alpha=color[0], ha='left', va='top', fontproperties=myf,rotation=ro, wrap=False)
    plt.axis('off')
    text_layer = fig2img(fig)  # convert to image
    plt.close()

    we = aa.get_window_extent()
    min_x, min_y, max_x, max_y = we.xmin, 500 - we.ymax, we.xmax, 500 - we.ymin
    box = (min_x - margin, min_y - margin, max_x + margin, max_y + margin)
    # return coordinates to further calculate the bbox of rotated text
    return text_layer, min_x, min_y, max_x, max_y

myf = font_manager.FontProperties(fname="fonts/Faith-Collapsing.ttf", size=50)
text_layer, min_x, min_y, max_x, max_y = text_on_canvas('SARA', myf, ro = 0, color=(0.4941, 0.4941, 0.4941))
min_x, min_y, max_x, max_y = int(min_x), int(min_y), int(max_x), int(max_y)
plt.imshow(np.array(text_layer)[min_y:max_y, min_x:max_x, :])
plt.show()