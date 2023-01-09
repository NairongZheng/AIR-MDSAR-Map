"""
    author:damonzheng
    function:test_fusion_models(多模型测试一个波段的图)
    edition:1.0
    date:2020.01.15
"""

import argparse
import cv2
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import math
import time

import hrnet

from default import _C as config
from default import update_config

import torch
from torch.nn import functional as F
import torch.backends.cudnn as cudnn


os.environ['CUDA_VISIBLE_DEVICES'] = "3"

def get_cmap(n_labels, label_dic):
    labels = np.ndarray((n_labels, 3), dtype='uint8')
    for i , (k, v) in enumerate(label_dic.items()):
        labels[i] = v
    cmap = np.zeros([768], dtype='uint8')
    index = 0
    for i in range(0, n_labels):
        for j in range(0, 3):
            cmap[index] = labels[i][j]
            index += 1
    return cmap

class TestDataLoader():
    """
        test data loader
    """
    def __init__(self, image_path):
        self.image_path = image_path
        self.crop_size = (256, 256)
        self.num_classes = 9
        self.samples, self.n_row, self.n_col, self.Height, self.Width = self.make_dataset(self.image_path)

    def make_dataset(self, img_path, height=256, width=256, stride=256):
        img = Image.open(img_path)
        img = np.array(img, dtype='float32')

        Height = img.shape[0]
        Width = img.shape[1]
        # ch = img.shape[2]

        if (Height % height == 0) and (Width % width == 0):
            print('nice image size for slice')
        else:
            pad_h = height - (Height % height)
            pad_w = width - (Width % width)
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')

        Height2 = img.shape[0]
        Width2 = img.shape[1]
        if (Height2 % height == 0) and (Width2 % width == 0):
            print('nice padding image size for slice')

        n_row = math.floor((Height2 - height) / stride) + 1
        n_col = math.floor((Width2 - width) / stride) + 1

        samples = np.zeros((n_row * n_col, height, width, 3), dtype=np.uint8)

        K = 0
        for m in range(n_row):
            row_start = m * stride
            row_end = m * stride + height
            for n in range(n_col):
                col_start = n * stride
                col_end = n * stride + width
                img_mn = img[row_start:row_end, col_start:col_end]
                samples[K, :, :, :] = img_mn
                K += 1

        return samples.copy(), n_row, n_col, Height, Width
    
    def input_transform(self, image):
        image = image.astype(np.float32)
        image = image / 127.5 -1
        return image

    def gen_sample(self, image):
        image = self.input_transform(image)
        image = image.transpose((2, 0, 1))
        return image
    
    def __getitem__(self, index):
        sample = self.samples[index]                # [h, w, c], RGB
        sample = self.gen_sample(sample)

        return sample.copy(), self.n_row, self.n_col, self.Height, self.Width

    def __len__(self):
        return len(self.samples)
    
    def image_resize(self, image, new_size):
        """
            resize
        """
        h, w = image.shape[:-1]
        if h > w:
            new_h = new_size
            new_w = int(w * new_size / h + 0.5)
        else:
            new_w = new_size
            new_h = int(h * new_size / w + 0.5)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)   # 这里没有写反噢
        return image
    
    def multi_scale_aug(self, image, base_size, rand_scale=1.):
        """
            多尺度的操作
        """
        new_size = int(base_size * rand_scale + 0.5)
        image = self.image_resize(image, new_size)
        return image
    
    def pad_image(self, image, h, w, size, pad_value):
        """
            填充图片
        """
        pad_image = image.copy()  # shadow copy
        pad_h = max(size[0] - h, 0)  # 判断是否需要填充  [h, w, c]
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:  # 右下方填充
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=pad_value)  # 边框

        return pad_image
    
    def inference_flip(self, model, image, gpu, flip=False):
        """
            原图先预测一个结果
            flip之后再预测一个结果再flip回来
            二者相加，再除以2得到最终的预测结果
        """
        size = image.size()         # (1, 3, 256, 256)
        pred = model(image)
        pred = F.interpolate(input=pred, size=(size[-2], size[-1]), mode='bilinear', align_corners=True)
        if flip:
            flip_img = image.cpu().numpy()[:, :, :, ::-1]           # 水平翻转
            flip_output = model(torch.from_numpy(flip_img.copy()).cuda(gpu))
            flip_output = F.interpolate(input=flip_output, size=(size[-2], size[-1]), mode='bilinear',
                                        align_corners=True)
            flip_pred = flip_output.cpu().numpy().copy()
            flip_pred = torch.from_numpy(flip_pred[:, :, :, ::-1].copy()).cuda(gpu)
            pred += flip_pred
            pred = pred * 0.5
        return pred.exp()############################################??????
    
    def multi_scale_inference_resize(self, model, image, gpu, scales=None, flip=False, padding=False):
        """
            多尺度推理
            把输入图像resize到原图的[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]然后再进行推理
            每个尺度推理的结果进行相加
        """
        if scales is None:
            scales = [1]
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        image = image.cpu().numpy()[0].transpose((1, 2, 0)).copy()  # [H, W, C]
        final_pred = torch.zeros([1, self.num_classes, ori_height, ori_width]).cuda(gpu)

        base_size = ori_height
        padvalue = 0.
        if base_size == 2048:
            scales = [0.5, 0.75, 1.0]

        for scale in scales:
            new_img = self.multi_scale_aug(image=image, base_size=base_size, rand_scale=scale)  # resize
            height, width = new_img.shape[:-1]
            if padding:
                new_img = self.pad_image(new_img, height, width, self.crop_size, pad_value=padvalue)
            new_img = new_img.transpose((2, 0, 1))  # [C, H, W]
            new_img = np.expand_dims(new_img, axis=0)  # [B, C, H, W]
            new_img = torch.from_numpy(new_img).cuda(gpu)
            preds = self.inference_flip(model, new_img, gpu, flip)  # direct inference
            preds = preds[:, :, 0:height, 0:width]
            preds = F.interpolate(preds, (ori_height, ori_width), mode='bilinear', align_corners=True)
            final_pred += preds             # 把每个尺度的预测结果都加起来
        return final_pred

def create_filename(input_dir):
    img_filename = []
    names = []
    path_list = os.listdir(input_dir)
    path_list.sort()
    for filename in path_list:
        char_name = filename.split('.')[0]
        names.append(char_name)
        file_path = os.path.join(input_dir, filename)
        img_filename.append(file_path)

    return img_filename, names

def save_pred(preds, name, args):
    print(preds.shape)
    water = preds == 0
    baresoil = preds == 1
    road = preds == 2
    industry = preds == 3
    vegetation = preds == 4
    residential = preds == 5
    plantingarea = preds == 6
    other = preds == 7
    farms = preds == 8

    h = preds.shape[0]
    w = preds.shape[1]
    del preds
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = water * 0 + baresoil * 139 + road * 83 + industry * 255 + vegetation * 0 + residential * 205 + plantingarea * 139 + other * 178 + farms * 0
    rgb[:, :, 1] = water * 0 + baresoil * 0 + road * 134 + industry * 0 + vegetation * 255 + residential * 173 + plantingarea * 105 + other * 34 + farms * 139
    rgb[:, :, 2] = water * 255 + baresoil * 0 + road * 139 + industry * 0 + vegetation * 0 + residential * 0 + plantingarea * 20 + other * 34 + farms * 139
    # rgb = rgb[:Height, :Width, :]
    save_img = Image.fromarray(np.uint8(rgb))
    save_img.save(os.path.join(args.save_path, 'pred_' + name + '.png'))

def parse_args():
    """
        配置参数
    """
    parser = argparse.ArgumentParser(description='test segmentation network')
    parser.add_argument('--cfg', help='the path of config file', default='/emwuser/znr/code/yyjf/experimentals_yaml/w_18/test_w18.yaml')
    parser.add_argument('--data', help='the path of testing data', default='/emwuser/znr/data/yyjf')
    parser.add_argument('--batch_size', help='mini-batch size', default=1)
    parser.add_argument('--classes', help='the number of classes', default=9)
    parser.add_argument('--weights_path', help='the path of test model', default='/emwuser/znr/code/yyjf/outputs/w_18')
    parser.add_argument('--fusion_band', help='the fusion band', default=['C', 'Ka', 'S', 'L', 'P', 'sxz'])
    parser.add_argument('--save_path', help='the path of save file', default='/emwuser/znr/data/yyjf/model_fusion')
    parser.add_argument('--gpu', help='gpu id to use', default=0, type=int)
    parser.add_argument('--multi_scaled_test', help='if use multi_scaled_test', default=False)
    args = parser.parse_args()
    update_config(config, args)
    return args

def test_model(test_dataset, testloader, model, args):
    """
        测试模型
    """
    model.eval()
    n_pred = []
    with torch.no_grad():
        # for i, batch in enumerate(tqdm(testloader)):
        for i, batch in enumerate(testloader):
            image, n_row, n_col, Height, Width = batch
            img_shape = image.shape
            image = image.cuda(args.gpu)
            if args.multi_scaled_test:
                scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
                pred = test_dataset.multi_scale_inference_resize(model, image, args.gpu, scales, flip=True)
            else:
                pred = test_dataset.inference_flip(model, image, args.gpu, flip=True)
            pred = F.interpolate(pred, (img_shape[-2], img_shape[-1]), mode='bilinear', align_corners=True)
            pred = pred.cpu().numpy()
            pred = pred.transpose((0, 2, 3, 1))     # [b, h, w, c]      (1, 256, 256, classes)
            # pred = np.argmax(pred, axis=-1)           # (1, 256, 256)
            n_pred.append(pred)
        row = int(n_row.numpy()[0])
        col = int(n_col.numpy()[0])
        stride = 256
        height = (row - 1) * stride + img_shape[-2]
        width = (col - 1) * stride + img_shape[-1]
        pred_np = np.zeros((height, width, args.classes))
        # print(pred_np.shape)
        for i in range(row):
            row_start = i * stride
            row_end = row_start + img_shape[-2]
            for j in range(col):
                col_start = j * stride
                col_end = col_start + img_shape[-1]
                num = i * col + j  # 第几张图片
                l_0 = num // args.batch_size
                l_1 = num % args.batch_size
                lab = n_pred[l_0][l_1]
                pred_np[row_start:row_end, col_start:col_end, :] = lab
        del n_pred
        del test_dataset
        del testloader
        # if not os.path.exists(args.save_path):
        #     os.mkdir(args.save_path)
        # save_pred(pred_np, int(Height.numpy()[0]), int(Width.numpy()[0]), name, args)
        pred_np = pred_np[:Height, :Width, :]
        return pred_np


def main():
    """
        主函数
    """
    args = parse_args()
    main_worker(args.gpu, args)
    
def main_worker(gpu, args):
    """
        main_worker
    """
    args.gpu = gpu
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True
    begin_time = time.time()

    test_img_num = len(os.listdir(os.path.join(args.data, 'C/test/img')))

    # 对每张图进行测试
    for i in range(test_img_num):
        print('testing images {}'.format(i + 1))
        final_result_one = np.zeros((2048, 2048, args.classes), np.float64)

        # 每张图都要融合各波段模型在各波段数据测试的结果
        for bands in tqdm(args.fusion_band):
            model = eval(config.MODEL.NAME + '.get_seg_model')(config)
            if bands == 'C':
                model_abs_path = os.path.join(args.weights_path, 'C/yyjf_C/hrnet_w18_train_512_1e-2_C', 'hrnet_w18_epoch196_fwiou_0.94995_OA_0.9735.pth')
            elif bands == 'Ka':
                model_abs_path = os.path.join(args.weights_path, 'Ka/yyjf_Ka/hrnet_w18_train_512_1e-2_Ka', 'hrnet_w18_epoch199_fwiou_0.95115_OA_0.9737.pth')
            elif bands == 'L':
                model_abs_path = os.path.join(args.weights_path, 'L/yyjf_L/hrnet_w18_train_512_1e-2_L', 'hrnet_w18_epoch188_fwiou_0.91006_OA_0.9509.pth')
            elif bands == 'P':
                model_abs_path = os.path.join(args.weights_path, 'P/yyjf_P/hrnet_w18_train_512_1e-2_P', 'hrnet_w18_epoch176_fwiou_0.91068_OA_0.9511.pth')
            elif bands == 'S':
                model_abs_path = os.path.join(args.weights_path, 'S/yyjf_S/hrnet_w18_train_512_1e-2_S', 'hrnet_w18_epoch188_fwiou_0.93236_OA_0.9637.pth')
            elif bands == 'sxz':
                model_abs_path = os.path.join(args.weights_path, 'sxz/yyjf_sxz/hrnet_w18_train_512_1e-2_sxz', 'hrnet_w18_epoch200_tr_fwiou_0.92919_tr_OA_0.9608_val_fwiou_0.28461_val_OA_0.3655.pth')

            pretrained_dict = torch.load(model_abs_path)        # 加载训练好的模型权重
            model_dict = model.state_dict()                 # 原始初始化的模型
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}      # 只取模型定义中有的部分
            model_dict.update(pretrained_dict)                  # 更新整个模型的参数
            model.load_state_dict(model_dict)               # 把更新好的参数加载到模型中
            model.cuda(args.gpu)
            print('load model {} done!'.format(bands))

            test_img_name = sorted(os.listdir(os.path.join(args.data, '{}/test/img'.format(bands))))[i]
            print('test image name is:{}'.format(test_img_name))
            test_img_path = os.path.join(args.data, '{}/test/img'.format(bands), test_img_name)
            test_dataset = TestDataLoader(test_img_path)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0)
            one_band = test_model(test_dataset, test_loader, model, args)
            one_band = cv2.resize(one_band, (2048, 2048))
            final_result_one += one_band
            del(one_band)
        final_result_one = np.argmax(final_result_one, axis=-1)
        save_pred(final_result_one, '{}'.format(i + 1), args)

    end_time = time.time()
    time_used = end_time - begin_time
    print('Inference time : {}s'.format(time_used))

if __name__ == '__main__':
    main()