"""
Author:DamonZheng
Function:train
Edition:...
Date:2021.5.16
"""
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from random import shuffle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, LearningRateScheduler
from weights import Deeplabv3

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
gpu_num = 3

IMG_TRAIN = '/emwuser/zry/znr/data/GF3_single_dataset/img_train'
LAB_TRAIN = '/emwuser/zry/znr/data/GF3_single_dataset/lab_train'
IMG_VAL = '/emwuser/zry/znr/data/GF3_single_dataset/img_val'
LAB_VAL = '/emwuser/zry/znr/data/GF3_single_dataset/lab_val'

learning_rate = 1e-3
img_rows = 256
img_cols = 256
img_channels = 3
n_labels = 10
batch_size = 16
epochs = 15
img_ext = '.png'
lab_ext = '.png'
# class_weight = {0:1, 1:7.11, 2:8.35, 3:5.09, 4:1.46, 5:1.46, 6:6.19, 7:1, 8:4.26, 9:5.52, 10:0}
# class_weight = [1, 7.11, 8.35, 6.09, 1.46, 1.46, 6.19, 1, 4.26, 5.52, 0]
# class_weight = [1, 7.11, 8.35, 5.09, 1.46, 1.46, 6.19, 4.26, 5.52]

def one_hot_lab(labels):
    lab_nd = np.zeros([img_cols, img_rows, n_labels])
    for i in range(0, n_labels):
        lab_nd[:, :, i] = np.array(labels == i, dtype='uint8')
    return lab_nd

def get_filename(img_path, lab_path):
    file_names = []
    img_filename = []
    lab_filename = []
    for (_, _, imgs) in os.walk(img_path):
        for img in imgs:
            (name, _) = os.path.splitext(img)
            file_names.append(name)
    shuffle(file_names)
    for i in range(0, len(file_names)):
        img_filename.append(os.path.join(img_path, file_names[i] + img_ext))
        lab_filename.append(os.path.join(lab_path, file_names[i] + lab_ext))

    return img_filename, lab_filename

def data_generator(img_filename, lab_filename, batch_size):
    j = 0
    while True:
        img_data = np.ndarray((batch_size, img_cols, img_rows, img_channels))
        lab_data = np.ndarray((batch_size, img_cols, img_rows, n_labels))
        if j > len(img_filename) - batch_size:
            j = 0
        for i in range(j, j + batch_size):
            img = load_img(img_filename[i])
            img = img_to_array(img)
            # img = Image.open(img_filename[i])
            # img = np.asarray(img)
            
            # lab = load_img(lab_filename[i])
            # lab = img_to_array(lab)
            lab = Image.open(lab_filename[i])
            lab = np.asarray(lab)

            img_data[i - j] = img
            lab_data[i - j] = one_hot_lab(lab)

        img_data = img_data / 127.5 - 1.0
        j = j + batch_size
        yield img_data, lab_data

class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self, model, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint,self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint,self).set_model(self.single_model)

# class ParallelModelCheckpoint(ModelCheckpoint):
#     def __init__(self, model, filepath, monitor='val_loss', verbose=0,
#                  save_best_only=False, save_weights_only=False,
#                  mode='auto', save_freq='epoch'):
#         self.single_model = model
#         super(ParallelModelCheckpoint,self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq)

#     def set_model(self, model):
#         super(ParallelModelCheckpoint,self).set_model(self.single_model)

    
def main():
    train_img_filename, train_lab_filename = get_filename(IMG_TRAIN, LAB_TRAIN)
    val_img_filename, val_lab_filename = get_filename(IMG_VAL, LAB_VAL)

    pre_model = Deeplabv3(input_shape=(img_rows, img_cols, img_channels), classes=n_labels)
    # pre_model = Deeplabv3(input_shape=(None, None, img_channels), classes=n_labels)
    pre_model.load_weights('/emwuser/zry/znr/python_code/GF3_single/pre_models/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5', by_name=True)
    model = multi_gpu_model(pre_model, gpus=gpu_num)

    def schedule(epoch):
        return learning_rate * pow(0.92, epoch)

    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto', cooldown=0)
    # lr_reduce = LearningRateScheduler(schedule=schedule)

    model_earlystop = EarlyStopping(patience=3, monitor='loss')
    model_tensorboard = TensorBoard(log_dir='/emwuser/zry/znr/python_code/GF3_single/znr/logs', update_freq='batch')
    model_checkpoint = ParallelModelCheckpoint(pre_model, filepath='/emwuser/zry/znr/python_code/GF3_single/znr/models/deeplab_loss_{epoch:02d}-{val_loss:.4f}--{val_acc:.4f}.hdf5', 
                                        monitor='val_loss',  save_weights_only=True, verbose=1, save_best_only=False)
    # model_checkpoint = ParallelModelCheckpoint(pre_model, filepath='/emwuser/zry/znr/python_code/seg_znr_try/models_att/deeplab_loss_{epoch:02d}.hdf5', 
    #                                     monitor='val_loss',  save_weights_only=True, verbose=0, save_best_only=False)

    # model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'], experimental_run_tf_function = False)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    # history = model.fit(data_generator(train_img_filename, train_lab_filename, batch_size), steps_per_epoch=len(train_img_filename) // batch_size, epochs=epochs, verbose=1, shuffle=True,
    #             validation_data=data_generator(val_img_filename, val_lab_filename, batch_size), validation_steps=len(val_img_filename) // batch_size,
    #             callbacks=[model_earlystop, model_tensorboard, model_checkpoint, lr_reduce], class_weight=class_weight)
    history = model.fit(data_generator(train_img_filename, train_lab_filename, batch_size), steps_per_epoch=len(train_img_filename) // batch_size, epochs=epochs, verbose=1, shuffle=True,
                validation_data=data_generator(val_img_filename, val_lab_filename, batch_size), validation_steps=len(val_img_filename) // batch_size,
                callbacks=[model_earlystop, model_tensorboard, model_checkpoint, lr_reduce])

    pre_model.save('pre_model.h5')
    model.save('model.h5')

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epochs), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), history.history["val_loss"], label="val_loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig('loss.png')

    plt.figure()
    plt.plot(np.arange(0, epochs), history.history["acc"], label="train_acc")
    plt.plot(np.arange(0, epochs), history.history["val_acc"], label="val_acc")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.savefig('acc.png')

if __name__ == '__main__':
    main()