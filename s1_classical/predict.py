from weights import Deeplabv3
from tensorflow.python.keras.optimizers import *
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import multi_gpu_model
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

PREDICT_PIC_PATH = '/emwuser/zry/znr/data/GF3_single_dataset/test/img_test'
SAVE_PATH = '/emwuser/zry/znr/data/GF3_single_dataset/test/znr'

learning_rate = 1e-3
img_rows = 256
img_cols = 256
img_channels = 3
n_labels = 10
batch_size = 16
label_dic = [{"name":"water", "rgb":[0,0,255]},             # 0.8068
            {"name":"bareoil", "rgb":[139,0,0]},            # 0.0067
            {"name":"road", "rgb":[83,134,139]},            # 0.1944
            {"name":"industry", "rgb":[255,0,0]},           # 0.1646
            {"name":"resident", "rgb":[205,173,0]},         # 0.6182
            {"name":"vegetation", "rgb":[0,255,0]},         # 0.5689
            {"name":"woodland", "rgb":[0,139,0]},           # 0.4024
            {"name":"paddyland", "rgb":[0,139,139]},        # 0
            {"name":"plantingland", "rgb":[139,105,20]},    # 0.0498
            {"name":"humanbuilt", "rgb":[189,183,107]}]     # 0.2591
            # {"name":"other", "rgb":[178,34,34]}]            # miou:0.3071; fwiou:0.5319; acc:0.6845

def get_cmap():
    labels = np.ndarray((n_labels, 3), dtype='uint8')
    for i in range(0, n_labels):
        labels[i] = label_dic[i]['rgb']
    cmap = np.zeros([768], dtype='uint8')
    index = 0
    for i in range(0, n_labels):
        for j in range(0, 3):
            cmap[index] = labels[i][j]
            index += 1
    print('cmap define finished')
    return cmap

def main():
    cmap = get_cmap()
    # 加载模型
    # with open('model.json','r') as file:
    #     model_json_r = file.read()

    model = Deeplabv3(input_shape=(img_rows, img_cols, img_channels), classes=n_labels)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights('/emwuser/zry/znr/python_code/GF3_single/znr/models/deeplab_loss_07-12.5467--0.7366.hdf5')

    pre_pic = os.listdir(PREDICT_PIC_PATH)
    for pic in tqdm(pre_pic, total=len(pre_pic)):
        img = load_img(os.path.join(PREDICT_PIC_PATH, pic))
        pre_data = np.ndarray((1,256,256,3))
        pre_x = img_to_array(img)
        pre_data[0] = pre_x
        pre_data = pre_data / 127.5 - 1
        res_x = model.predict(pre_data)
        
        result = np.argmax(res_x[0], axis=2)

        imagename, _ = os.path.splitext(pic)
        new_name = imagename + '.png'

        znr = Image.fromarray(np.uint8(result))
        znr.putpalette(cmap)
        znr.save(os.path.join(SAVE_PATH, new_name))
        # cv2.imwrite(os.path.join(SAVE_PATH, new_name), result)
        # img_cmap = Image.open(os.path.join(SAVE_PATH, new_name))
        # img_cmap.putpalette(cmap)
        # img_cmap.save(os.path.join(SAVE_PATH, new_name))
    print("predict finished.")

if __name__ == '__main__':
    main()