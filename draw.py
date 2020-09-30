# %%
import glob
import matplotlib.pyplot as plt
from PIL import Image
from datasets import get_candidate_info_list


def get_pic(pic_id):
    """ 根据图片id获取图片
        Args:
            pic_id(int): 图片id
        Returns:
            PIL.image: 对应的图片
    """
    pic_path = glob.glob('data\\train\\domain*\\{}.jpg'.format(pic_id))
    return Image.open(pic_path[0])


def bbox_to_rect(xmin, ymin, xmax, ymax, color):
    """ 生成matplotlib的矩形框
        Args:
            xmin, ymin, xmax, ymax(int): 矩形的左上、右下坐标
            color(string): 矩形框的颜色
        Returns:
            plt.Rectangle: matplotlib矩形框
    """
    return plt.Rectangle(xy=(xmin, ymin), width=xmax-xmin, height=ymax-ymin,
                         fill=False, edgecolor=color, linewidth=2)


def draw_pic(pic_id):
    """ 绘制带有矩形标注框的图片，违禁品为红色边框，非违禁为绿色边框
        Args:
            pic_id(string): 图片id
    """
    candidate_info_list = get_candidate_info_list()
    candidates = [
        candidate for candidate in candidate_info_list if candidate.pic_id == pic_id]
    fig = plt.imshow(get_pic(pic_id))
    for candidate in candidates:
        if candidate.object_name == 'normal':
            fig.axes.add_patch(bbox_to_rect(
                candidate.xmin, candidate.ymin, candidate.xmax, candidate.ymax, 'green'))
            fig.axes.text(candidate.xmin, candidate.ymin,
                          candidate.object_name)
        else:
            fig.axes.add_patch(bbox_to_rect(
                candidate.xmin, candidate.ymin, candidate.xmax, candidate.ymax, 'red'))
            fig.axes.text(candidate.xmin, candidate.ymin,
                          candidate.object_name)
    plt.show()

# %%
