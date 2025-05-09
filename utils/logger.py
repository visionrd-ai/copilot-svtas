'''
Author: Thyssen Wen
Date: 2022-03-16 20:52:46
LastEditors: Thyssen Wen
LastEditTime: 2022-04-28 14:15:08
Description: logger config function ref: https://github.com/PaddlePaddle/PaddleVideo
FilePath: /ETESVS/utils/logger.py
'''
import logging
import os
import sys
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter

Color = {
    'RED': '\033[31m',
    'HEADER': '\033[35m',  # deep purple
    'PURPLE': '\033[95m',  # purple
    'OKBLUE': '\033[94m',
    'OKGREEN': '\033[92m',
    'WARNING': '\033[93m',
    'FAIL': '\033[91m',
    'ENDC': '\033[0m'
}


def coloring(message, color="OKGREEN"):
    assert color in Color.keys()
    if os.environ.get('COLORING', True):
        return Color[color] + str(message) + Color["ENDC"]
    else:
        return message


logger_initialized = {}


def setup_logger(output=None, name="SVTAS", level="INFO", tensorboard=False):
    """
    Initialize the SVTAS logger and set its verbosity level to "INFO".
    """
    def time_zone(sec, fmt):
        real_time = datetime.datetime.now()
        return real_time.timetuple()
    logging.Formatter.converter = time_zone

    logger = logging.getLogger(name)
    if level == "INFO":
        logger.setLevel(logging.INFO)
    elif level=="DEBUG":
        logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if level == "DEBUG":
        plain_formatter = logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
            datefmt="%m/%d %H:%M:%S")
    else:
        plain_formatter = logging.Formatter(
            "[%(asctime)s] %(message)s",
            datefmt="%m/%d %H:%M:%S")
    local_rank = int(os.environ['LOCAL_RANK'])
    if local_rank < 0:
        local_rank = 0
        
    if local_rank == 0:
        # stdout logging: master only
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        if tensorboard is True:
            idx = str(0)
            path = os.path.join(output, "tensorboard", idx)
            isExists = os.path.exists(path)
            while isExists:
                idx = str(int(idx) + 1)
                path = os.path.join(output, "tensorboard", idx)
                isExists = os.path.exists(path)
            writer = SummaryWriter(path, comment=name)

    # file logging: all workers
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, name + ".log")
        if local_rank > 0:
            filename = filename + ".rank{}".format(local_rank) + ".log"

        # PathManager.mkdirs(os.path.dirname(filename))
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # fh = logging.StreamHandler(_cached_log_stream(filename)
        fh = logging.FileHandler(filename, mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)
    logger_initialized[name] = dict(logging=name)
    if tensorboard is True and local_rank <= 0:
        logger_initialized[name]['tensorboard'] = writer
    return logger


def get_logger(name, output=None, tensorboard=False):
    logger = logging.getLogger(name)
    if name in list(logger_initialized.keys()):
        if tensorboard is True:
            return logger_initialized[name]['tensorboard']
        return logger

    return setup_logger(name=name, output=name)


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self, name='', fmt='f', need_avg=True):
        self.name = name
        self.fmt = fmt
        self.need_avg = need_avg
        self.reset()

    def reset(self):
        """ reset """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ update """
        if isinstance(val, torch.Tensor):
            val = val.cpu().detach().numpy()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def total(self):
        return '{self.name}_sum: {self.sum:{self.fmt}}'.format(self=self)

    @property
    def total_minute(self):
        return '{self.name}_sum: {s:{self.fmt}} min'.format(s=self.sum / 60,
                                                            self=self)

    @property
    def get_mean(self):
        return self.avg if self.need_avg else self.val

    @property
    def mean(self):
        return '{self.name}_avg: {self.avg:{self.fmt}}'.format(
            self=self) if self.need_avg else ''

    @property
    def value(self):
        return '{self.name}: {self.val:{self.fmt}}'.format(self=self)


def log_batch(metric_list, batch_id, epoch_id, total_epoch, mode, ips, logger):
    batch_cost = str(metric_list['batch_time'].value) + ' sec,'
    reader_cost = str(metric_list['reader_time'].value) + ' sec,'

    metric_values = []
    for m in metric_list:
        if not (m == 'batch_time' or m == 'reader_time'):
            metric_values.append(metric_list[m].value)
    metric_str = ' '.join([str(v) for v in metric_values])
    if mode in ["train", "validation"]:
        epoch_str = "epoch:[{:>3d}/{:<3d}]".format(epoch_id, total_epoch)
    step_str = "{:s} step:{:<4d}".format(mode, batch_id)

    if mode in ["train", "validation"]:
        logger.info("{:s} {:s} {:s} {:s} {:s} {}".format(
            coloring(epoch_str, "HEADER"),
            coloring(step_str, "PURPLE"), coloring(metric_str, 'OKGREEN'),
            coloring(batch_cost, "OKGREEN"), coloring(reader_cost, 'OKGREEN'), ips))
    elif mode in ["test"]:
        logger.info("{:s} {:s} {:s} {:s} {}".format(
            coloring(step_str, "PURPLE"), coloring(metric_str, 'OKGREEN'),
            coloring(batch_cost, "OKGREEN"), coloring(reader_cost, 'OKGREEN'), ips))

def log_epoch(metric_list, epoch, mode, ips, logger):
    batch_cost = 'avg_' + str(metric_list['batch_time'].value) + ' sec,'
    reader_cost = 'avg_' + str(metric_list['reader_time'].value) + ' sec,'
    batch_sum = str(metric_list['batch_time'].total) + ' sec,'

    metric_values = []
    for m in metric_list:
        if not (m == 'batch_time' or m == 'reader_time'):
            metric_values.append(metric_list[m].mean)
    metric_str = ' '.join([str(v) for v in metric_values])

    end_epoch_str = "END epoch:{:<3d}".format(epoch)

    logger.info("{:s} {:s} {:s} {:s} {:s} {:s} {}".format(
        coloring(end_epoch_str, "RED"), coloring(mode, "PURPLE"),
        coloring(metric_str, "OKGREEN"), coloring(batch_cost, "OKGREEN"),
        coloring(reader_cost, "OKGREEN"), coloring(batch_sum, "OKGREEN"), ips))

def tenorboard_log_epoch(metric_list, epoch, mode, writer):
    if isinstance(writer, SummaryWriter):
        for m in metric_list:
            if not (m == 'batch_time' or m == 'reader_time'):
                writer.add_scalar(mode + "/" + m, metric_list[m].get_mean, epoch)