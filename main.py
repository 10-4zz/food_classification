import os
import random
import numpy as np
import time
import datetime

import logging

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.amp import autocast, GradScaler

from config import get_config
from data import build_loader
from loggers import LoggerController, summery_model, throughput
from lr_scheduler import build_scheduler
from loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from models import build_model
from optimizer import build_optimizer
from utils import get_options, ModelEma, AverageMeter, accuracy, get_grad_norm, save_checkpoint_ema_new


def get_opts_config():
    parser = get_options()
    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.enabled = True
    cudnn.benchmark = True


def train_one_epoch(config, model, model_ema, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, logger,
                    total_epochs, mesa=1.0):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()

    scaler = GradScaler()

    for idx, (samples, targets) in enumerate(data_loader):

        optimizer.zero_grad()
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if config.AMP:
            with autocast():
                if mesa > 0:
                    with torch.inference_mode():
                        ema_output = model_ema.ema(samples).detach()
                    ema_output = torch.clone(ema_output)
                    ema_output = ema_output.softmax(dim=-1).detach()
                    outputs = model(samples)
                    loss = criterion(outputs, targets) + criterion(outputs, ema_output) * mesa
                else:
                    outputs = model(samples)
                    loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            if config.TRAIN.CLIP_GRAD:
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = get_grad_norm(model.parameters())
                scaler.step(optimizer)
                scaler.update()
        else:
            if mesa > 0:
                with torch.inference_mode():
                    ema_output = model_ema.ema(samples).detach()
                ema_output = torch.clone(ema_output)
                ema_output = ema_output.softmax(dim=-1).detach()
                outputs = model(samples)
                loss = criterion(outputs, targets) + criterion(outputs, ema_output) * mesa
            else:
                outputs = model(samples)
                loss = criterion(outputs, targets)
            loss.backward()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(model.parameters())
            optimizer.step()

        lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch + 1}/{total_epochs}][{idx + 1}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch + 1} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model, logger):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # acc1 = reduce_tensor(acc1)
        # acc5 = reduce_tensor(acc5)
        # loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{(idx + 1)}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


def main():
    args, config = get_opts_config()

    seed = config.SEED
    setup_seed(seed=seed)

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE / 512.0

    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.LOCAL_RANK = 0
    config.freeze()

    # adjust ema decay according to total batch size, may not be optimal
    args.model_ema_decay = args.model_ema_decay ** (config.DATA.BATCH_SIZE / 4096.0)

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger_controller = LoggerController(log_file_name="log.txt" ,output_dir=config.OUTPUT)
    logger = logger_controller.create_logger(
        logger_name='food_classification',
        log_level=logging.DEBUG,
        format_type="default"
    )

    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")

    logger.info(config.dump())

    _, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config=config, logger=logger)

    logger.info(f"Creating model:{config.MODEL.NAME}")
    model = build_model(config=config)
    model.cuda()

    logger.info(str(model))
    summery_model(
        model=model,
        logger=logger,
        image_size=config.DATA.IMAGE_SIZE,
    )

    optimizer = build_optimizer(config, model)

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    total_epochs = config.TRAIN.EPOCHS + config.TRAIN.COOLDOWN_EPOCHS
    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = nn.CrossEntropyLoss()

    max_accuracy = 0.0
    max_accuracy_e = 0.0

    model_ema = None
    if args.model_ema:
        if not config.EVAL_MODE:
            logger.info(f'Model EMA decay {args.model_ema_decay}')
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume=config.MODEL.RESUME)
        acc1_e, acc5_e, loss_e = validate(config, data_loader_val, model_ema.ema, logger)
        torch.cuda.empty_cache()
        logger.info(f"Accuracy of the ema network on the {len(dataset_val)} test images: {acc1_e:.1f}%")
        if config.EVAL_MODE:
            return

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, total_epochs):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, model_ema, criterion, data_loader_train, optimizer, epoch, mixup_fn,
                        lr_scheduler, logger, total_epochs,
                        mesa=config.AUG.MESA if epoch >= int(0.25 * total_epochs) else -1.0)
        acc1, acc5, loss = validate(config, data_loader_val, model, logger)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if model_ema is not None and not args.model_ema_force_cpu:
            acc1_e, acc5_e, loss_e = validate(config, data_loader_val, model_ema.ema, logger)
            logger.info(f"Accuracy of the ema network on the {len(dataset_val)} test images: {acc1_e:.1f}%")
        else:
            acc1_e, acc5_e, loss_e = 0, 0, 0

        if (epoch + 1) % config.SAVE_FREQ == 0 or (epoch + 1) == (total_epochs):
            save_checkpoint_ema_new(config, epoch + 1, model, model_ema, max(max_accuracy, acc1),
                                    max(max_accuracy_e, acc1_e), optimizer, lr_scheduler, logger)

        if ((epoch + 1) % config.SAVE_FREQ == 0 or (epoch + 1) == total_epochs) and acc1 >= max_accuracy:
            save_checkpoint_ema_new(config, epoch + 1, model, model_ema, max(max_accuracy, acc1),
                                    max(max_accuracy_e, acc1_e), optimizer, lr_scheduler, logger, name='max_acc')
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')
        with open(os.path.join(args.output, 'max_acc.log'), 'w', encoding='utf-8') as f:
            f.write(f'Max accuracy: {max_accuracy:.2f}%')

        if model_ema is not None and not args.model_ema_force_cpu:
            if ((epoch + 1) % config.SAVE_FREQ == 0 or (epoch + 1) == total_epochs) and acc1_e >= max_accuracy_e:
                save_checkpoint_ema_new(config, epoch + 1, model, model_ema, max(max_accuracy, acc1),
                                        max(max_accuracy_e, acc1_e), optimizer, lr_scheduler, logger,
                                        name='max_ema_acc')
            max_accuracy_e = max(max_accuracy_e, acc1_e)
            logger.info(f'Max ema accuracy: {max_accuracy_e:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    main()

