import os
import sys
import shutil
import logging
import glob

import torch as t


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.makedirs(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    t.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


class RecordTrainResult(object):
    def __init__(self, checkpoint_path):

        create_exp_dir(os.path.join(checkpoint_path, "code_bkps"), scripts_to_save=glob.glob('*.py'))
        # 第一步，创建一个logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)  # Log等级总开关
        # 第二步，创建一个handler，用于写入日志文件
        # ymdhm = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        log_path = os.path.join(checkpoint_path, "result")
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_file = os.path.join(log_path, 'log.txt')

        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
        # 第三步，定义handler的输出格式
        log_format = '%(asctime)s %(message)s'
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        # 第四步，将logger添加到handler里面
        logger.addHandler(fh)
        self.logger = logger

        self.best_valid_acc = 0
        self.is_max_valid_acc = False

    def add_log(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def save_model_state_dict(self, valid_acc, epoch, state_dict, optimizer_state_dict, save_path):
        """
        state_dict = model.state_dict(),
        optimizer_state_dict = optimizer.state_dict()
        """
        if valid_acc > self.best_valid_acc:
            self.best_valid_acc = valid_acc
            self.is_max_valid_acc = True
        else:
            self.is_max_valid_acc = False

        model_state_dict = {
            'epoch': epoch,
            'is_max_valid_acc': self.is_max_valid_acc,
            'valid_acc': valid_acc,
            'state_dict': state_dict,
            'optimizer_state_dict': optimizer_state_dict
            }

        save_checkpoint(model_state_dict, self.is_max_valid_acc, save_path)

