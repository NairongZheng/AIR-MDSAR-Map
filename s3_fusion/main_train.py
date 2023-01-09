"""
    author:damonzheng
    function:3dresunet
    edition:1.0
    date:2022.5.21
"""

import argparse
import os
import yaml
import logging
from pathlib import Path
import numpy as np
import time
import timeit

from default import _C as config
from default import update_config

import ResUNet
from hrnet_3d import get_seg_model
from hrnet import hrnet_2d_get_seg_model

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
# from loss import SegmentationLosses
from criterion import CrossEntropy, OhemCrossEntropy
from data_loader_gdal import SegmentationDatasetLoader
from data_loader_2d import SegmentationDatasetLoader_2d
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from train_val import train, validate

import weights_init

os.environ['CUDA_VISIBLE_DEVICES'] = "1,2"

def parse_args():
    """
       initialize parameter
    """
    parser = argparse.ArgumentParser(description='train segmentation network (3dresunet/3ddhrnet)')
    parser.add_argument('--cfg', help='the path of config file', default='/emwuser/znr/code/hyper_sar/experimentals_yaml/3dhrnet_w18_train_512_1e-2_MS_xiongan.yaml')

    # DDP options
    parser.add_argument('--nodes', help='the number of nodes for distributed training', default=1, type=int)        # 要使用的节点数
    parser.add_argument('--gpu', help='gpu id to use', default=0, type=int)                                         # 使用的gpu号
    parser.add_argument('--node_rank', help='ranking within the nodes', default=0, type=int)                        # 当前节点在所有节点中的排名
    parser.add_argument('--local_rank', help='local_rank', default=0, type=int)
    parser.add_argument('--dist_backend', help='distributed backend', default='nccl', type=str)
    parser.add_argument('--dist_url', help='url used to set up distributed training', default='tcp://127.0.0.1:8054', type=str)
    parser.add_argument('--multiprocessing_distributed', action='store_true', default=False, 
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    # training options
    parser.add_argument('--batch_size', help='mini-batch size', default=8)
    parser.add_argument('--nesterov', help='whether use nesterov', default=False, action='store_true')

    args = parser.parse_args()
    update_config(config, args)
    return args

# def create_logger(args):
#     """
#         log
#     """
#     # 创建日志
#     time_str = time.strftime('%Y-%m-%d-%H-%M')
#     head = '%(asctime)-15s %(message)s'
#     output_path = Path(args.outputs_path) / time_str
#     output_path.mkdir(parents=True, exist_ok=True)
#     log_filename = os.path.join(str(output_path), '{}.log'.format(time_str))
#     logging.basicConfig(filename=log_filename, format=head, level=logging.INFO)
#     # logger = logging.getLogger()
#     # logger.setLevel(logging.INFO)
#     console = logging.StreamHandler()       # 输出到控制台
#     logging.getLogger('').addHandler(console)

#     logging.info('=> the path of config file is:{}'.format(args.config_file))
    
#     tensorboard_path = output_path  / 'tensorboard'
#     models_path = output_path  / 'models'
#     tensorboard_path.mkdir(parents=True, exist_ok=True)
#     models_path.mkdir(parents=True, exist_ok=True)

#     args.logs_path = log_filename
#     args.tensorboard_path = str(tensorboard_path)
#     args.models_path = str(models_path)
#     logging.info('=> the path of output logs is:{}'.format(args.logs_path))
#     logging.info('=> the path of output tensorboard is:{}'.format(args.tensorboard_path))
#     logging.info('=> the path of output models is:{}'.format(args.models_path))
#     # return logger

def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)          # 模型跟tensorboard的保存路径
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]         # 配置文件的名称

    final_output_dir = root_output_dir / dataset / cfg_name     # 最终模型跟log文件的保存路径

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / (cfg_name + '_' + time_str)     # tensorboard的保存路径
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)

def main():
    # 加载配置文件的参数
    args = parse_args()

    # # 创建日志
    # create_logger(args)

    ngpus_per_node = torch.cuda.device_count()
    args.gpus = ngpus_per_node
    # logging.info('=> there are {} gpus can be used'.format(ngpus_per_node))
    
    # judge if using distributed training or not
    args.distributed = args.nodes > 1 or args.multiprocessing_distributed

    if args.distributed:
        args.world_size = ngpus_per_node * args.nodes       # 总进程数=每个节点的GPU数量*节点数
        # logging.info('=> use distributed training, world_size:{}'.format(args.world_size))
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, config))
    else:
        # logging.info('=> do not use distributed training')
        main_worker(args.gpu, ngpus_per_node, args, config)

def main_worker(gpu, ngpus_per_node, args, config):
    """
        main_worker
    """
    args.gpu = gpu

    # 创建日志
    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, 'train')
    logging.info('=> this is gpu-{}'.format(gpu))
    if gpu == 0:
        logging.info('=> args are:{}'.format(args))

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    if args.distributed:
        if args.dist_url == "env://":
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.node_rank * args.gpus + gpu      # 进程编号=机器编号*每台机器可用GPU数量+当前GPU编号(这里的gpu其实就是local_rank)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    # define model
    # znr
    if config.TRAIN.MODEL == 'hrnet_3d':
        print('using hrnet')
        model = get_seg_model(config)
        # model.apply(weights_init.init_model)
    elif config.TRAIN.MODEL == 'resunet_3d':
        model = ResUNet.ResUNet(out_channel=config.DATASET.NUM_CLASSES)
        model.apply(weights_init.init_model)
    elif config.TRAIN.MODEL == 'hrnet_2d':
        print('using hrnet_2d')
        model = hrnet_2d_get_seg_model(config)
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        torch.cuda.set_device(gpu)
        model.cuda(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    else:
        model = torch.nn.DataParallel(model).cuda()
    
    # define optimozer and criterion
    class_weights = None
    optimizer = torch.optim.SGD([{'params': filter(lambda p: p.requires_grad, model.parameters()),
                                      'lr': config.TRAIN.LR}],
                                    lr=config.TRAIN.LR,
                                    momentum=config.TRAIN.MOMENTUM,
                                    weight_decay=config.TRAIN.WD,
                                    nesterov=config.TRAIN.NESTEROV,
                                    )
    if config.LOSS.USE_OHEM:
        criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                     thres=config.LOSS.OHEMTHRES,
                                     min_kept=config.LOSS.OHEMKEEP,
                                     weight=class_weights).cuda(args.gpu)
    else:
        criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                 weight=class_weights).cuda(args.gpu)

    # loading dataset
    cudnn.benchmark = True
    
    train_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])  # [Height, Width]
    if config.TRAIN.MODEL == 'hrnet_2d':
        train_dataset = SegmentationDatasetLoader_2d(
            image_path=config.DATASET.TRAIN_IMAGE_ROOT,
            label_path=config.DATASET.TRAIN_LABEL_ROOT,
            num_classes=config.DATASET.NUM_CLASSES,
            multi_scale=config.TRAIN.MULTI_SCALE,
            flip=config.TRAIN.FLIP,
            ignore_label=config.TRAIN.IGNORE_LABEL,
            base_size=config.TRAIN.BASE_SIZE,
            crop_size=train_size,
            downsample_rate=1,
            scale_factor=16,
            center_crop_test=False, 
            label_mapping=config.TRAIN.EXTRA.LABEL_MAPPING)
    else:
        train_dataset = SegmentationDatasetLoader(
            image_path=config.DATASET.TRAIN_IMAGE_ROOT,
            label_path=config.DATASET.TRAIN_LABEL_ROOT,
            num_classes=config.DATASET.NUM_CLASSES,
            multi_scale=config.TRAIN.MULTI_SCALE,
            flip=config.TRAIN.FLIP,
            ignore_label=config.TRAIN.IGNORE_LABEL,
            base_size=config.TRAIN.BASE_SIZE,
            crop_size=train_size,
            downsample_rate=1,
            scale_factor=16,
            center_crop_test=False, 
            label_mapping=config.TRAIN.EXTRA.LABEL_MAPPING)

    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=config.TRAIN.SHUFFLE and train_sampler is None,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler)
    if config.TRAIN.MODEL == 'hrnet_2d':
        test_dataset = SegmentationDatasetLoader_2d(
            image_path=config.DATASET.TEST_IMAGE_ROOT,
            label_path=config.DATASET.TEST_LABEL_ROOT,
            num_classes=config.DATASET.NUM_CLASSES,
            flip=False,
            ignore_label=config.TRAIN.IGNORE_LABEL, 
            label_mapping=config.TRAIN.EXTRA.LABEL_MAPPING)
    else:
        test_dataset = SegmentationDatasetLoader(
            image_path=config.DATASET.TEST_IMAGE_ROOT,
            label_path=config.DATASET.TEST_LABEL_ROOT,
            num_classes=config.DATASET.NUM_CLASSES,
            flip=False,
            ignore_label=config.TRAIN.IGNORE_LABEL, 
            label_mapping=config.TRAIN.EXTRA.LABEL_MAPPING)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    # training
    best_train_fwiou = 0
    best_val_fwiou = 0
    epoch_iters = np.int32(train_dataset.__len__() / args.batch_size / 2)       # 最后这个除数就是几块卡
    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters

    for epoch in range(0, end_epoch):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_loss, train_MIoU, train_MOA, train_IoU_array, train_fw_iou, train_rank = train(config, epoch, end_epoch,
                                                                       epoch_iters, config.TRAIN.LR,
                                                                       num_iters, train_loader, criterion, optimizer,
                                                                       model, writer_dict, args)
        # train(config, epoch, end_epoch, epoch_iters, config.TRAIN.LR,
        #         num_iters, train_loader, criterion, optimizer, model, writer_dict, args)

        val_IoU_array, val_MIoU, val_MOA, val_fw_iou, rank = validate(config, test_loader, model, writer_dict, args.gpu)

        if rank == 0:
            if train_fw_iou > best_train_fwiou:
                best_train_fwiou = train_fw_iou
                if train_fw_iou > 0.7:
                    save_name = config.TRAIN.MODEL + '_epoch' + str(epoch + 1) + '_tr_fwiou_' + str(train_fw_iou)[:7] + '_tr_OA_' + str(train_MOA)[:6]\
                        + '_val_fwiou_' + str(val_fw_iou)[:7] + '_val_OA_' + str(val_MOA)[:6] + '.pth'
                    torch.save(model.module.state_dict(), os.path.join(final_output_dir, save_name))
            if val_fw_iou > best_val_fwiou:
                best_val_fwiou = val_fw_iou
                torch.save(model.module.state_dict(), os.path.join(final_output_dir, 'best_val_fwiou_epoch{}.pth'.format(epoch + 1)))
            train_msg = 'train_loss: {:.3f}, train_OA: {:.6f}, train_miou: {:.6f}, train_fwiou: {:.6f}, train_best_fwiou: {:.6f}'.format(train_loss, train_MOA, train_MIoU, train_fw_iou, best_train_fwiou)
            logging.info(train_msg)
            logging.info(train_IoU_array)

            val_msg = 'val_OA: {:.6f}, val_miou: {:.6f}, val_fwiou: {:.6f}, val_best_fwiou: {:.6f}'.format(val_MOA, val_MIoU, val_fw_iou, best_val_fwiou)
            logging.info(val_msg)
            logging.info(val_IoU_array)

            if epoch == end_epoch - 1:
                end = timeit.default_timer()
                logging.info('Hours: %d' % np.int32((end - start) / 3600))
                logging.info('Done')

if __name__ == '__main__':
    main()