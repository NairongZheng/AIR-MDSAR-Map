from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import os
from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.GPUS = (4, )
_C.WORKERS = 4
_C.PRINT_FREQ = 20

_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.RANK = 0

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'hrnet'
_C.MODEL.PRETRAINED = ''
_C.MODEL.EXTRA = CN(new_allowed=True)

_C.LOSS = CN()
_C.LOSS.USE_OHEM = False
_C.LOSS.OHEMTHRES = 0.9
_C.LOSS.OHEMKEEP = 100000
_C.LOSS.CLASS_BALANCE = True

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.DATASET = 'yyjf'
_C.DATASET.NUM_CLASSES = 9
_C.DATASET.NUM_BAND = 8             # 这个要改的
_C.DATASET.TRAIN_IMAGE_ROOT = ''
_C.DATASET.TRAIN_LABEL_ROOT = ''
_C.DATASET.TEST_IMAGE_ROOT = ''
_C.DATASET.TEST_LABEL_ROOT = ''

# training
_C.TRAIN = CN()
_C.TRAIN.MODEL = ''     # 新加的用来选择模型
_C.TRAIN.IMAGE_SIZE = [512, 512]  # width * height, CROP_SIZE
_C.TRAIN.FLIP = True
_C.TRAIN.MULTI_SCALE = True
_C.TRAIN.SCALE_FACTOR = 16
_C.TRAIN.IGNORE_LABEL = 255
_C.TRAIN.BASE_SIZE = 1500
_C.TRAIN.DOWNSAMPLERATE = 1
_C.TRAIN.CENTER_CROP_TEST = True

_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.01
_C.TRAIN.OPTIMIZER = 'sgd'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 484
_C.TRAIN.RESUME = False

_C.TRAIN.BATCH_SIZE_PER_GPU = 4
_C.TRAIN.SHUFFLE = True

_C.TRAIN.EXTRA = CN(new_allowed=True)#################

# TEST
_C.TEST = CN()
_C.TEST.MODEL_FILE = ''
_C.TEST.IMAGE_SIZE = [512, 512]
_C.TEST.BASE_SIZE = 1500
_C.TEST.BATCH_SIZE_PER_GPU = 4
_C.TEST.FLIP_TEST = False
_C.TEST.MULTI_SCALE = False


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    # cfg.merge_from_list(args.opts)
    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

