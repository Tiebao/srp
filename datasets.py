# %%
import glob
from collections import Counter, namedtuple
import torch
import copy
from lxml import etree
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from functools import lru_cache
from torchvision import transforms

# 定义标注框的结构
candidate_info = namedtuple(
    'candidate_info', 'pic_id, candidate_id, object_name, width, height, xmin, ymin, xmax, ymax')


@lru_cache
def get_candidate_info_list():
    """ 提取所有.xml中的标注框
        Returns:
            candidate_info_list(List): 存放所有标注框信息的列表
    """
    xml_list = glob.glob('data\\train\\domain*\\XML\\*.xml')

    candidate_info_list = []
    candidate_id = 0
    for xml_path in xml_list:
        with open(xml_path, 'rb') as f:
            xml = etree.XML(f.read())
            pic_id = xml_path.split('\\')[4].split('.')[0]
            object_names = xml.xpath('//name/text()')
            width = xml.xpath('//width/text()')
            height = xml.xpath('//height/text()')
            xmin = xml.xpath('//xmin/text()')
            ymin = xml.xpath('//ymin/text()')
            xmax = xml.xpath('//xmax/text()')
            ymax = xml.xpath('//ymax/text()')

            for i, object_name in enumerate(object_names):
                candidate_info_list.append(candidate_info(
                    pic_id, candidate_id, object_names[i],
                    round(float(width[0])), round(float(height[0])),
                    round(float(xmin[i])), round(float(ymin[i])),
                    round(float(xmax[i])), round(float(ymax[i]))))
                candidate_id += 1

    return candidate_info_list


def get_pic_slice(candidate_info):
    """ 根据标注框的信息，分割图片并返回
        Args:
            candidate_info(List): 存放所有标注框信息的列表
        Returns:
            PIL.Image: 分割后的图片
    """
    pic_path = glob.glob(
        'data\\train\\domain*\\{}.jpg'.format(candidate_info.pic_id))[0]
    pic_img = Image.open(pic_path)
    pic_slice = pic_img.crop((candidate_info.xmin, candidate_info.ymin,
                              candidate_info.xmax, candidate_info.ymax))
    return pic_slice


class XrayDateset(Dataset):
    """ 数据集的类
        Args:
            is_val(bool): 为真则返回验证集，否则返回训练集。默认为假
            val_stride(int): 验证集的步长，即每隔多长取一个验证集，
                获取训练集和验证集时需保证该值相同。默认为0
            pic_id(string, 可选): 只获取对应ID的图片，默认为None
    """

    def __init__(self, is_val=0, val_stride=0, pic_id=None):
        self.candidate_info_list = copy.copy(get_candidate_info_list())

        if pic_id:
            self.candidate_info_list = [
                candidate for candidate in self.candidate_info_list if candidate.pic_id == pic_id]

        if is_val:
            assert val_stride > 0, val_stride
            self.candidate_info_list = self.candidate_info_list[::val_stride]
        elif val_stride > 0:
            del self.candidate_info_list[::val_stride]
            assert self.candidate_info_list

    def __len__(self):
        """ 获取数据集的长度
            Returns:
                len(int): 数据集的长度
        """
        return len(self.candidate_info_list)

    def __getitem__(self, index):
        """ 获取数据集的每一项
            Args:
                index(int): 下标
            Returns:
                pic_slice(PIL.Image): 图片的分割
                candidate_info(namedtumple): 该分割对应的标注框
        """
        candidate_info = self.candidate_info_list[index]
        pic_slice = get_pic_slice(candidate_info)

        return (pic_slice, candidate_info)


# %%
