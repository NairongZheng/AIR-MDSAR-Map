"""
    Function:unet_weights
    Date:2021.5.7
"""
# import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ZeroPadding2D, BatchNormalization, MaxPooling2D, Concatenate
from tensorflow.keras.layers import Conv2D, Activation, UpSampling2D, AveragePooling2D, Add, Lambda, Multiply
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, GlobalMaxPooling2D
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import tensorflow as tf
# from .vgg16 import get_vgg_encoder
# from .mobilenet import get_mobilenet_encoder
# from .basic_models import vanilla_encoder
# from .resnet50 import get_resnet50_encoder

pre_path = '/emwuser/zry/znr/python_code/GF3_single/pre_models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

pretrained_url = "https://github.com/fchollet/deep-learning-models/" \
                 "releases/download/v0.1/" \
                 "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
EPS = 1e-9

def att(x, reduction = 32):
    
    def coord_act(x):
        tmpx = tf.nn.relu6(x+3) / 6
        x = x * tmpx
        return x

    x_shape = x.get_shape().as_list()
    [b, h, w, c] = x_shape

    # # 这是tf1的写法
    # x_h = slim.avg_pool2d(x, kernel_size = [1, w], stride = 1)
    # x_w = slim.avg_pool2d(x, kernel_size = [h, 1], stride = 1)
    # x_w = tf.transpose(x_w, [0, 2, 1, 3])
    # y = tf.concat([x_h, x_w], axis=1)

    # 这是tf2的写法
    x_h_a = AveragePooling2D(pool_size=(1, w))(x)
    x_h_m = MaxPooling2D(pool_size=(1, w))(x)
    x_h = Add()([x_h_a, x_h_m])

    x_w_a = AveragePooling2D(pool_size=(h, 1))(x)
    x_w_m = MaxPooling2D(pool_size=(h, 1))(x)
    x_w = Add()([x_w_a, x_w_m])

    se_1 = x_w
    # x_w = tf.transpose(x_w, [0, 2, 1, 3])
    x_w = Lambda(tf.transpose, arguments={'perm':[0, 2, 1, 3]})(x_w)
    y = Concatenate(axis=1)([x_h, x_w])

    
    mip = max(8, c // reduction)
    # # 这是tf1的写法
    # y = slim.conv2d(y, mip, (1, 1), stride=1, padding='VALID', normalizer_fn = slim.batch_norm, activation_fn=coord_act,scope='ca_conv1')

    # 这是tf2的写法
    y = Conv2D(mip, (1, 1), strides=1, padding='same', use_bias=False)(y)
    y = BatchNormalization(epsilon=EPS)(y)
    y = Activation(coord_act)(y)
    # y = Activation('relu')(y)

    # x_h, x_w = tf.split(y, num_or_size_splits=2, axis=1)
    x_h, x_w = Lambda(tf.split, arguments={'num_or_size_splits':2, 'axis':1})(y)
    # x_w = tf.transpose(x_w, [0, 2, 1, 3])
    x_w = Lambda(tf.transpose, arguments={'perm':[0, 2, 1, 3]})(x_w)

    # # # 这是tf1的写法
    # a_h = slim.conv2d(x_h, c, (1, 1), stride=1, padding='VALID', normalizer_fn = None, activation_fn=tf.nn.sigmoid,scope='ca_conv2')
    # a_w = slim.conv2d(x_w, c, (1, 1), stride=1, padding='VALID', normalizer_fn = None, activation_fn=tf.nn.sigmoid,scope='ca_conv3')

    # 这是tf2的写法
    a_h = Conv2D(c, (1,1), strides=1, padding='same', use_bias=False)(x_h)
    a_h = Activation('sigmoid')(a_h)
    a_w = Conv2D(c, (1,1), strides=1, padding='same', use_bias=False)(x_w)
    a_w = Activation('sigmoid')(a_w)

    # out_ca = x * a_h * a_w
    out_ca = Multiply()([x, a_h, a_w])

    # 以下是se
    squeeze_gap = GlobalAveragePooling2D()(x)
    squeeze_gmp = GlobalMaxPooling2D()(x)
    squeeze = Add()([squeeze_gap, squeeze_gmp])
    excitation = Dense(mip, activation='relu')(squeeze)
    excitation = Dense(c, activation='sigmoid')(excitation)
    excitation = Lambda(lambda x1: K.expand_dims(x1, 1))(excitation)
    excitation = Lambda(lambda x2: K.expand_dims(x2, 1))(excitation)

    se_1 = Conv2D(c, (1,1), strides=1, padding='same', use_bias=False)(se_1)
    se_1 = BatchNormalization(epsilon=EPS)(se_1)
    se_1 = Activation('relu')(se_1)
    se_1_a = AveragePooling2D(pool_size=(1, w))(se_1)
    se_1_m = MaxPooling2D(pool_size=(1, w))(se_1)
    se_1 = Add()([se_1_a, se_1_m])
    se_1 = Conv2D(c, (1,1), strides=1, padding='same', use_bias=False)(se_1)
    se_1 = Activation('sigmoid')(se_1)

    excitation = Add()([se_1, excitation])
    # excitation = tf.reshape(excitation, [-1, 1, 1, c])
    scale = Multiply()([x, excitation])
    # out_se = tf.keras.layers.add([x, scale])

    # out = Add()([out_ca, scale])
    out = Multiply()([x, a_h, a_w, excitation])

    return out

def vanilla_encoder(input_height=256,  input_width=256):

    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2

    img_input = Input(shape=(input_height, input_width, 3))

    x = img_input
    levels = []

    x = (ZeroPadding2D((pad, pad)))(x)
    x = (Conv2D(filter_size, (kernel, kernel), padding='valid'))(x)
    x = (BatchNormalization())(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D((pool_size, pool_size)))(x)
    levels.append(x)

    x = (ZeroPadding2D((pad, pad)))(x)
    x = (Conv2D(128, (kernel, kernel),
         padding='valid'))(x)
    x = (BatchNormalization())(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D((pool_size, pool_size)))(x)
    levels.append(x)

    for _ in range(3):
        x = (ZeroPadding2D((pad, pad)))(x)
        x = (Conv2D(256, (kernel, kernel), padding='valid'))(x)
        x = (BatchNormalization())(x)
        x = (Activation('relu'))(x)
        x = (MaxPooling2D((pool_size, pool_size)))(x)
        levels.append(x)

    return img_input, levels


def get_vgg_encoder(input_height=256,  input_width=256, pretrained='imagenet'): # 用的是这个

    assert input_height % 32 == 0
    assert input_width % 32 == 0

    img_input = Input(shape=(input_height, input_width, 3))

    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               name='block1_conv1')(img_input)
    x = att(x) ###############################################################
    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    f1 = x
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
               name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
               name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv1')(x)
    x = att(x) ###############################################################
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    f3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    f4 = x

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv1')(x)
    x = att(x) ###############################################################
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    f5 = x

    # if pretrained == 'imagenet':
    #     VGG_Weights_path = keras.utils.get_file(
    #         pretrained_url.split("/")[-1], pretrained_url)
    #     Model(img_input, x).load_weights(VGG_Weights_path)
    Model(img_input, x).load_weights(pre_path, by_name=True) #############################################################
    return img_input, [f1, f2, f3, f4, f5]


def get_segmentation_model(input, output):

    img_input = input
    o = output

    o_shape = Model(img_input, o).output_shape
    i_shape = Model(img_input, o).input_shape

    output_height = o_shape[1]
    output_width = o_shape[2]
    input_height = i_shape[1]
    input_width = i_shape[2]
    n_classes = o_shape[3]
    # o = (Reshape((output_height*output_width, -1)))(o)

    o = (Activation('softmax'))(o)
    model = Model(img_input, o)
    model.output_width = output_width
    model.output_height = output_height
    model.n_classes = n_classes
    model.input_height = input_height
    model.input_width = input_width

    return model


def segnet_decoder(f, n_classes, n_up=4):

    assert n_up >= 2

    o = f
    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(512, (3, 3), padding='valid'))(o)
    o = (BatchNormalization())(o)
    o = att(o) ###############################################################

    o = (UpSampling2D((2, 2)))(o)
    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(256, (3, 3), padding='valid'))(o)
    o = (BatchNormalization())(o)
    o = att(o) ###############################################################

    for _ in range(n_up-2):
        o = (UpSampling2D((2, 2)))(o)
        o = (ZeroPadding2D((1, 1)))(o)
        o = (Conv2D(128, (3, 3), padding='valid'))(o)
        o = (BatchNormalization())(o)

    o = att(o) ###############################################################
    o = (UpSampling2D((2, 2)))(o)
    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(64, (3, 3), padding='valid'))(o)
    o = (BatchNormalization())(o)

    o = Conv2D(n_classes, (3, 3), padding='same')(o)

    return o


def _segnet(n_classes, encoder,  input_height=256, input_width=256,
            encoder_level=3):

    img_input, levels = encoder(
        input_height=input_height,  input_width=input_width)

    feat = levels[encoder_level]
    o = segnet_decoder(feat, n_classes, n_up=4)
    model = get_segmentation_model(img_input, o)

    return model


def segnet(n_classes, input_height=256, input_width=256, encoder_level=3):

    model = _segnet(n_classes, vanilla_encoder,  input_height=input_height,
                    input_width=input_width, encoder_level=encoder_level)
    model.model_name = "segnet"
    # model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def vgg_segnet(n_classes, input_height=256, input_width=256, encoder_level=3):

    model = _segnet(n_classes, get_vgg_encoder,  input_height=input_height,
                    input_width=input_width, encoder_level=encoder_level)
    model.model_name = "vgg_segnet"
    # model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# def resnet50_segnet(n_classes, input_height=256, input_width=256,
#                     encoder_level=3):

#     model = _segnet(n_classes, get_resnet50_encoder, input_height=input_height,
#                     input_width=input_width, encoder_level=encoder_level)
#     model.model_name = "resnet50_segnet"
#     return model


# def mobilenet_segnet(n_classes, input_height=256, input_width=256,
#                      encoder_level=3):

#     model = _segnet(n_classes, get_mobilenet_encoder,
#                     input_height=input_height,
#                     input_width=input_width, encoder_level=encoder_level)
#     model.model_name = "mobilenet_segnet"
#     return model


if __name__ == '__main__':
    model = vgg_segnet(8)
    # model = segnet(8)
    # model = mobilenet_segnet(8)
    # from keras.utils import plot_model
    # plot_model( m , show_shapes=True , to_file='model.png')
    model.summary()