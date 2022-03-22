# coding:utf8
import warnings
import torch as t
import os


class DefaultConfig(object):
    env = 'default'  # visdom 环境
    vis_port =8097 # visdom 端口

    model = 'LeNet'
    pretrained_weight_path = None  # 加载预训练的模型的路径，为None代表不加载

    checkpoint_path = os.path.join("models/lenet", "checkpoints")  # save path for trained CNN weight
    train_data_root = './data/train/'  # 训练集存放路径
    test_data_root = './data/test1'  # 测试集存放路径
    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    # optimizer configurations
    weighted_loss = False  # Add weight to the loss
    max_epoch = 10
    lr = 2e-3  # initial learning rate
    weight_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    momentum = 0.9
    # weight_decay = 0e-5  # 损失函数
    batch_size = 128  # batch size
    report_freq = 20  # print info every N batch

    use_gpu = True  # user GPU or not
    num_workers = 2  # how many workers for loading data
    device = t.device('cuda') if use_gpu else t.device('cpu')

    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))


opt = DefaultConfig()
