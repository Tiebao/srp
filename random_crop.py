# %%
from datasets import XrayDateset, get_candidate_info_list
from lxml import etree
from lxml.builder import E
from PIL import Image
import random
import glob

# %%


def write_xml(xml_path, xmin, ymin, xmax, ymax):
    """ 将标注框的信息写入.xml
        Args:
            xml_path(string): .xml文件的路径
            xmin(int): 标注框的左上横坐标
            ymin(int): 标注框的左上纵坐标
            xmax(int): 标注框的右下横坐标
            ymax(int): 标注框的右下纵坐标
    """
    with open(xml_path, 'rb') as f:
        xml = etree.XML(f.read(), parser=etree.XMLParser(
            remove_blank_text=True))
        xml.insert(-1,
                   E.object(
                       E.name('normal'),
                       E.pose('Unspecified'),
                       E.truncated('0'),
                       E.difficult('0'),
                       E.bndbox(
                           E.xmin(str(xmin)),
                           E.ymin(str(ymin)),
                           E.xmax(str(xmax)),
                           E.ymax(str(ymax))))
                   )
        etree.ElementTree(xml).write(xml_path, pretty_print=True)


def is_overlap(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
    """ 检验两个矩形框是否有重叠，根据两个矩形中心点的距离
    是否小于两个矩形边长之和的一半来判断
        Args:
            两个矩形框的左上和右下坐标
        Returns:
            flag(bool): 为真则两个矩形框有重叠
    """
    center_x1, center_y1 = (xmax1 + xmin1) / 2, (ymax1 + ymin1) / 2
    center_x2, center_y2 = (xmax2 + xmin2) / 2, (ymax2 + ymin2) / 2
    distance_center_x = abs(center_x1 - center_x2)
    distance_center_y = abs(center_y1 - center_y2)
    len_x1, len_y1 = xmax1 - xmin1, ymax1 - ymin1
    len_x2, len_y2 = xmax2 - xmin2, ymax2 - ymin2
    if (distance_center_x < (len_x1 + len_x2) / 2) and (distance_center_y < (len_y1 + len_y2) / 2):
        return True
    else:
        return False


# %%
candidate_info_list = get_candidate_info_list()
pics_path = glob.glob('data\\train\\domain*\\*.jpg')
for pic_path in pics_path:
    pic_info = [item for item in candidate_info_list if item.pic_id ==
                pic_path.split('\\')[3].split('.')[0]]
    height = pic_info[0].height
    width = pic_info[0].width

    for i in range(3):
        # 每张图片尝试随机生成三个矩形框，如果和原来的标注框有重叠，则丢弃
        height_crop = random.randint(round(height/4), round(height/2))
        # width_crop = height_crop
        width_crop = random.randint(round(width/4), round(width/2))

        top = random.randint(round(height/6), height -
                             round(height/6) - height_crop)
        left = random.randint(round(width/6), width -
                              round(width/6) - width_crop)
        bottom = top + height_crop
        right = left + width_crop

        overlapping_flag = sum([is_overlap(left, top, right, bottom, candidate.xmin,
                                           candidate.ymin, candidate.xmax, candidate.ymax)
                                for candidate in pic_info])

        if not overlapping_flag:
            path_parts = pic_path.split('\\')
            xml_path = '\\'.join(
                (path_parts[0], path_parts[1], path_parts[2], 'XML', '{}.xml'.format(pic_info[0].pic_id)))
            write_xml(xml_path, left, top, right, bottom)


# %%
