"""
    Author:damonzheng
    Function:dataloader_gdal
    Edition:1.0
    Date:2022.5.21
"""
import os
import torch
import cv2
import numpy as np
import random
from torch.nn import functional as F
import gdal

class SegmentationDatasetLoader():
    def __init__(self,
                 image_path,
                 label_path,
                 num_classes=9, multi_scale=True, flip=True,
                 ignore_label=255, base_size=512, crop_size=(256, 256), downsample_rate=1,
                 scale_factor=16, center_crop_test=False, mean=None, std=None, label_mapping=None):

        # if not mean:
        #     self.mean = [0.354099, 0.342272, 0.304951]
        # if not std:
        #     self.std = [0.161363, 0.144342, 0.123388]

        # self.normal = False
        self.shift_value = 10
        self.brightness = False

        self.num_classes = num_classes
        # self.class_weights = torch.FloatTensor([1., 1., 1.,
        #                                         1., 1., 1.,
        #                                         1., 1., 1.,
        #                                         1.]).cuda()
        self.image_rpath = image_path
        self.label_rpath = label_path

        self.multi_scale = multi_scale
        self.flip = flip
        self.crop_size = crop_size
        self.ignore_label = ignore_label
        self.base_size = base_size
        self.scale_factor = scale_factor
        # self.downsample_rate = downsample_rate
        self.center_crop_test = center_crop_test

        self.files = self.read_files()
        # self.label_mapping = {0: [0, 0, 0], 1: [255, 255, 255], 255: [255, 189, 0]}
        self.label_mapping = label_mapping
    
    def read_files(self):
        files = []
        name_list = os.listdir(self.image_rpath)
        name_list.sort()
        for name in name_list:
            char_name = name.split('.')
            image_path = os.path.join(self.image_rpath, char_name[0] + '.tif')  # name
            label_path = os.path.join(self.label_rpath, char_name[0] + '.png')
            files.append({'img': image_path, 'label': label_path, 'name': name})

        return files
    
    def convert_label(self, label):
        """
            转成单通道
        """
        temp = label.copy()
        label_mask = np.zeros((label.shape[0], label.shape[1]))
        for i, (k, v) in enumerate(self.label_mapping.items()):
            label_mask[(((temp[:, :, 0] == v[0]) & (temp[:, :, 1] == v[1])) & (temp[:, :, 2] == v[2]))] = int(k)

        return label_mask

    def read_hyper(self, img_path):
        """
        读取图像信息
    """
        img = gdal.Open(img_path)
        height = img.RasterYSize        # 获取图像的行数
        width = img.RasterXSize         # 获取图像的列数
        band_num = img.RasterCount      # 获取图像波段数

        return img, height, width, band_num

    def hyper2numpy(self, dataset, h, w, band_num):
        """
        把gdal读出来的hyper的dataset格式转成矩阵形式
    """
        all_band_data = np.zeros((h, w, band_num))
        for i in range(0, band_num):
            all_band_data[:,:,i] = dataset.GetRasterBand(i + 1).ReadAsArray(0, 0, w, h)
        return all_band_data
    
    def image_resize(self, image, long_size, label=None):
        h, w = image.shape[:-1]
        if h > w:
            new_h = long_size
            new_w = np.int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int(h * long_size / w + 0.5)

        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        if label is not None:
            label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            return image

        return image, label

    def pad_image(self, image, h, w, size, pad_value):
        pad_image = image.copy()  # shadow copy
        pad_h = max(size[0] - h, 0)  # 判断是否需要填充  [h, w, c]
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:  # 右下方填充
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=pad_value)  # 边框

        return pad_image

    def rand_crop(self, image, label):
        """
            要截取的大小如果大于图像大小, 就用0或者ignore_label填充image和label; 否则不填充
            然后再在该图像上, 随机选择一个左上角, 截取crop_size大小的图像返回
            当然, 这个左上角的选取保证不会截出边界
        """
        h, w = image.shape[:-1]  # [h, w, c]
        image = self.pad_image(image, h, w, self.crop_size, (0.0, 0.0, 0.0))  # pad, 填充零值
        label = self.pad_image(label, h, w, self.crop_size, (self.ignore_label,))  # pad
        new_h, new_w = label.shape  # [h, w], mask
        x = random.randint(0, new_w - self.crop_size[1])  # w, col
        y = random.randint(0, new_h - self.crop_size[0])  # h, row
        image = image[y:y + self.crop_size[0], x:x + self.crop_size[1]]
        label = label[y:y + self.crop_size[0], x:x + self.crop_size[1]]
        return image, label

    def multi_scale_aug(self, image, label=None, rand_scale=1., rand_crop=True):
        long_size = np.int(self.base_size * rand_scale + 0.5)           # 把base_size放缩到新的尺度

        if label is not None:
            image, label = self.image_resize(image, long_size, label)   # 不管原始图像多大，都是放缩到base_size的rand_scale倍的大小
            if rand_crop:
                image, label = self.rand_crop(image, label)
            return image, label
        else:
            image = self.image_resize(image, long_size)
            return image

    def input_transform(self, image):
        image = image.astype(np.float32)
        # if self.normal:
        #     image /= 255.
        #     image -= self.mean
        #     image /= self.std
        # else:
        #     image = image / 127.5 - 1  # [-1, 1]
        image = image / 127.5 - 1  # [-1, 1]
        return image

    def label_transform(self, label):
        return np.array(label).astype('int32')  # int32

    def gen_sample(self, image, label, multi_scale=True, is_flip=True):
        """
            对打开的图像跟标签进行处理并输出
        """
        if multi_scale:
            # 采用多尺度
            rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0  # [0.5, 2] scale    0.5到2随机取一个做放缩
            image, label = self.multi_scale_aug(image, label, rand_scale=rand_scale)

        else:
            # 不采用多尺度
            image, label = self.rand_crop(image, label)


        image = self.input_transform(image)
        label = self.label_transform(label)

        image = image.transpose((2, 0, 1))  # [0, 1, 2] -> [2, 0, 1], 下面对Width处理（下面要flip的，也就是水平翻转的）

        if is_flip:                 # 如果选择filp的话，是随机flip的
            flip = np.random.choice(2) * 2 - 1  # [0, 1] * 2 - 1 = [-1, 1]
            image = image[:, :, ::flip]  # channel [-1, 1] 正向（不变），反向（倒序）
            label = label[:, ::flip]

        # if self.downsample_rate != 1:  # 下采样
        #     image = cv2.resize(image, None, fx=self.downsample_rate, fy=self.downsample_rate,
        #                        interpolation=cv2.INTER_LINEAR)
        #     label = cv2.resize(label, None, fx=self.downsample_rate, fy=self.downsample_rate,
        #                        interpolation=cv2.INTER_NEAREST)

        return image, label

    def __getitem__(self, index):
        item = self.files[index]
        name = item['name']

        # 读取多波段图像, 假设尺寸为[256, 256, 8]
        (img, h, w, band_num) = self.read_hyper(item['img'])
        image = self.hyper2numpy(img, h, w, band_num)
        image = image.astype(np.float32)
        size = image.shape  # [h, w, c]

        # 读取标签图像, 假设尺寸为[256, 256, 3]
        label = cv2.imread(item['label'], cv2.IMREAD_COLOR)  # [H, W, BGR]
        label = label.astype(np.int32)[:, :, ::-1]  # BGR -> RGB
        label = self.convert_label(label)  # 转成单通道[h, w]

        # 入参image是多通道numpy, label是转成单通道后的, 但是还没有onehot. 可选参数:多尺度, 翻转
        image, label = self.gen_sample(image, label, self.multi_scale, self.flip)

        # 因为要做三维卷积, 所以要把image原来的通道当作空间维度, 并扩一个通道维度
        # image = image.transpose((1, 2, 0))
        image = np.expand_dims(image, axis=0).repeat(3, axis=0)

        return image.copy(), label.copy(), np.array(size), name

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    image_path = '/emwuser/znr/data/hyper_sar/img_train_256'
    label_path = '/emwuser/znr/data/hyper_sar/lab_train_256'
    LABEL_MAPPING = {0: [0, 0, 255], 1: [139, 0, 0], 2: [83, 134, 139], 
                    3:[255, 0, 0], 4:[0, 255, 0], 5:[205, 173, 0], 
                    6:[139, 105, 20], 7:[178, 34, 34], 8:[0, 139, 139], 
                    255:[255, 255, 255]}

    aaa = SegmentationDatasetLoader(image_path, label_path, label_mapping = LABEL_MAPPING)
    a, b, c, d = aaa.__getitem__(index=12)
    pass