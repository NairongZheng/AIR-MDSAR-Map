import logging
import os
import time

import numpy as np
# import numpy.ma as ma
from tqdm import tqdm

import torch
# import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F

from utils import AverageMeter
from utils import get_confusion_matrix
from utils import adjust_learning_rate

from utils import get_world_size, get_rank


def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that
    process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        dist.reduce(reduced_inp, dst=0)
    return reduced_inp


def train(config, epoch, num_epoch, epoch_iters, base_lr, num_iters,
          trainloader, criterion, optimizer, model, writer_dict, args):  # writer_dict
    # Training
    model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    m_oa = AverageMeter()
    m_iou = AverageMeter()
    fw_iou = AverageMeter()

    tic = time.time()
    cur_iters = epoch * epoch_iters

    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    rank = get_rank()
    world_size = get_world_size()

    confusion_matrix_print = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))

    for i_iter, batch in enumerate(trainloader):
        images, labels, _, _ = batch
        images = images.cuda(non_blocking=True)
        labels = labels.long().cuda(non_blocking=True)

        size = labels.size()
        # images = images.to(device)
        # labels = labels.long().to(device)

        preds = model(images)
        preds = F.interpolate(input=preds, size=(size[-2], size[-1]), mode='bilinear', align_corners=True)
        loss = criterion(preds, labels)

        # reduced_loss = reduce_tensor(loss)

        # measure acc and record loss
        confusion_matrix = get_confusion_matrix(
            labels,
            preds,
            size,
            config.DATASET.NUM_CLASSES,
            ignore=config.TRAIN.IGNORE_LABEL)

        confusion_matrix_print += confusion_matrix

        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)

        oa = np.sum(tp) / (np.sum(confusion_matrix) + 1e-7)
        m_oa.update(oa, images.size(0))

        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        m_iou.update(mean_IoU, images.size(0))

        freq = np.sum(confusion_matrix, axis=1) / (np.sum(confusion_matrix) + 1e-7)
        iu = np.diag(confusion_matrix) / (
                np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                np.diag(confusion_matrix) + 1e-7)
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        fw_iou.update(FWIoU, images.size(0))

        ave_loss.update(loss.item(), images.size(0))

        # compute gradient and do SGD step
        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        # ave_loss.update(reduced_loss.item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter + cur_iters)

        if (i_iter + 1) % config.PRINT_FREQ == 0 and rank == 0:

            print_loss = ave_loss.average() / world_size
            # confusion_matrix = torch.from_numpy(confusion_matrix).to(device)
            # reduced_confusion_matrix = reduce_tensor(confusion_matrix)
            # confusion_matrix = reduced_confusion_matrix.cpu().numpy()

            # mean_IoU = 0
            # oa = 0
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f},lr: {:.6f}, Loss: {:.6f}, Mean_IoU: {:.4f}, OA: {:.4f}, FWIoU: {:.6f}'\
                .format(epoch + 1, num_epoch, i_iter + 1, epoch_iters, batch_time.average(), lr,
                        print_loss, m_iou.average(), m_oa.average(), fw_iou.average())

            logging.info(msg)
            writer.add_scalar('train_loss',  print_loss, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

    pos = confusion_matrix_print.sum(1)
    res = confusion_matrix_print.sum(0)
    tp = np.diag(confusion_matrix_print)
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))

    print_loss = ave_loss.average() / world_size

    return print_loss, m_iou.average(), m_oa.average(), IoU_array, fw_iou.average(), rank

def validate(config, testloader, model, writer_dict, device):  # writer_dict
    rank = get_rank()
    world_size = get_world_size()
    model.eval()
    miou = AverageMeter()
    fwiou = AverageMeter()
    moa = AverageMeter()
    confusion_matrix_print = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))

    with torch.no_grad():
        for _, batch in enumerate(testloader):
            image, label, _, _ = batch
            size = label.size()
            image = image.cuda(non_blocking=True)
            label = label.long().cuda()

            # losses, pred = model(image, label)
            pred = model(image)
            pred = F.upsample(input=pred, size=(size[-2], size[-1]), mode='bilinear', align_corners=True)

            confusion_matrix = get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

            confusion_matrix_print += confusion_matrix

            pos = confusion_matrix.sum(1)
            res = confusion_matrix.sum(0)
            tp = np.diag(confusion_matrix)

            oa = np.sum(tp) / (np.sum(confusion_matrix) + 1e-7)
            moa.update(oa, image.size(0))

            IoU_array = (tp / np.maximum(1.0, pos + res - tp))
            mean_IoU = IoU_array.mean()
            miou.update(mean_IoU, image.size(0))

            freq = np.sum(confusion_matrix, axis=1) / (np.sum(confusion_matrix) + 1e-7)
            iu = np.diag(confusion_matrix) / (
                    np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                    np.diag(confusion_matrix) + 1e-7)
            FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
            fwiou.update(FWIoU, image.size(0))


    pos = confusion_matrix_print.sum(1)
    res = confusion_matrix_print.sum(0)
    tp = np.diag(confusion_matrix_print)

    # iou_and_miou
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    # OA = np.sum(tp) / np.sum(confusion_matrix)
    # mean_IoU = IoU_array.mean()
    # # print_loss = ave_loss.average() / world_size

    # freq = pos / (np.sum(confusion_matrix) + 1e-7)
    # iu = tp / (pos + res - tp + 1e-7)
    # FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()

    if rank == 0:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        # writer.add_scalar('valid_loss', print_loss, global_steps)
        writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1
    # return print_loss, IoU_array, mean_IoU, OA, FWIoU, rank
    return IoU_array, miou.average(), moa.average(), fwiou.average(), rank