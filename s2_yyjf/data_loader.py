import os
import torch
import cv2
import numpy as np
import random
from torch.nn import functional as F


class SegmentationDatasetLoader():
    def __init__(self,
                 image_path,
                 label_path,
                 num_classes=10, multi_scale=True, flip=True,
                 ignore_label=255, base_size=1024, crop_size=(512, 512), downsample_rate=1,
                 scale_factor=16, center_crop_test=False, mean=None, std=None, label_mapping=None):

        if not mean:
            self.mean = [0.354099, 0.342272, 0.304951]
        if not std:
            self.std = [0.161363, 0.144342, 0.123388]

        self.normal = False
        self.shift_value = 10
        self.brightness = False

        # if not mean:
        #     self.mean = [39.391, 38.184, 44.618]
        # if not std:
        #     self.std = [49.646, 30.556, 51.206]

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
        self.downsample_rate = downsample_rate
        self.center_crop_test = center_crop_test

        self.files = self.read_files()
        # self.label_mapping = {0: [0, 0, 0], 1: [255, 255, 255], 255: [255, 189, 0]}
        self.label_mapping = label_mapping

    def center_crop(self, image, label):
        h, w = image.shape[:-1]
        x = int(round((w - self.crop_size[1]) / 2.))  # w
        y = int(round((h - self.crop_size[0]) / 2.))  # h
        image = image[y:y + self.crop_size[0], x:x + self.crop_size[1]]
        label = label[y:y + self.crop_size[0], x:x + self.crop_size[1]]

        return image, label

    def read_files(self):
        files = []
        name_list = os.listdir(self.image_rpath)
        name_list.sort()
        for name in name_list:
            char_name = name.split('.')
            image_path = os.path.join(self.image_rpath, char_name[0] + '.tif')  # name
            # #######################################################################################
            # label_name_split = char_name[0].split('_')
            # label_name = label_name_split[0] + '_' + label_name_split[1] + '_label_' + label_name_split[2] + '_' + label_name_split[3]
            # label_path = os.path.join(self.label_rpath, label_name + '.png')
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
        long_size = np.int(self.base_size * rand_scale + 0.5)       # 把base_size放缩到新的尺度

        if label is not None:
            image, label = self.image_resize(image, long_size, label)   # 不管原始图像多大，都是放缩到base_size的rand_scale倍的大小
            if rand_crop:
                image, label = self.rand_crop(image, label)             # 在放缩后的基础上随机取一个左上角的点，然后取crop_size来训练，超出边界就是0
            return image, label
        else:
            image = self.image_resize(image, long_size)
            return image

    def random_brightness(self, img):
        if random.random() < 0.5:
            return img
        img = img.astype(np.float32)
        shift = random.randint(-self.shift_value, self.shift_value)
        img[:, :, :] += shift
        img = np.around(img)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def input_transform(self, image):
        image = image.astype(np.float32)
        if self.normal:
            image /= 255.
            image -= self.mean
            image /= self.std
        else:
            image = image / 127.5 - 1  # [-1, 1]

        return image

    def label_transform(self, label):
        return np.array(label).astype('int32')  # int32

    def gen_sample(self, image, label, multi_scale=True, is_flip=True, center_crop_test=False):
        if multi_scale:
            rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0  # [0.5, 2] scale    0.5到2随机取一个做放缩
            image, label = self.multi_scale_aug(image, label, rand_scale=rand_scale)
        else:
            image, label = self.rand_crop(image, label)

        if center_crop_test:
            image, label = self.image_resize(image, self.base_size, label)
            image, label = self.center_crop(image, label)

        if self.brightness:
            image = self.random_brightness(image)
        image = self.input_transform(image)
        label = self.label_transform(label)

        image = image.transpose((2, 0, 1))  # [0, 1, 2] -> [2, 0, 1], 下面对Width处理（下面要flip的，也就是水平翻转的）

        if is_flip:                 # 如果选择filp的话，是随机flip的
            flip = np.random.choice(2) * 2 - 1  # [0, 1] * 2 - 1 = [-1, 1]
            image = image[:, :, ::flip]  # channel [-1, 1] 正向（不变），反向（倒序）
            label = label[:, ::flip]

        if self.downsample_rate != 1:  # 下采样
            image = cv2.resize(image, None, fx=self.downsample_rate, fy=self.downsample_rate,
                               interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, None, fx=self.downsample_rate, fy=self.downsample_rate,
                               interpolation=cv2.INTER_NEAREST)

        return image, label

    def __getitem__(self, index):
        item = self.files[index]
        name = item['name']
        image = cv2.imread(item['img'], cv2.IMREAD_COLOR)  # color [H, W, 3]
        # image = cv2.imread(item["img"], cv2.IMREAD_GRAYSCALE)  # [H, W]
        image = image.astype(np.float32)[:, :, ::-1]  # BGR -> RGB
        size = image.shape  # [h, w, c]

        # label = cv2.imread(item['label'], cv2.IMREAD_GRAYSCALE)  # [H, W]
        label = cv2.imread(item['label'], cv2.IMREAD_COLOR)  # [H, W, BGR]
        label = label.astype(np.int32)[:, :, ::-1]  # BGR -> RGB
        label = self.convert_label(label)  # [h, w]

        image, label = self.gen_sample(image, label, self.multi_scale, self.flip, self.center_crop_test)

        return image.copy(), label.copy(), np.array(size), name

    def __len__(self):
        return len(self.files)

    def inference(self, model, image, flip=False):
        size = image.size()
        pred = model(image)
        pred = F.interpolate(input=pred, size=(size[-2], size[-1]), mode='bilinear', align_corners=True)
        if flip:
            flip_img = image.numpy()[:, :, :, ::-1]
            flip_output = model(torch.from_numpy(flip_img.copy()))
            flip_output = F.interpolate(input=flip_output, size=(size[-2], size[-1]),
                                        mode='bilinear', align_corners=True)
            flip_pred = flip_output.cpu().numpy().copy()
            flip_pred = torch.from_numpy(flip_pred[:, :, :, ::-1].copy()).cuda()
            pred += flip_pred
            pred = pred * 0.5
        return pred.exp()


