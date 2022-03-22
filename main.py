# coding:utf8
import os
import sys

# TODO: select a dataset
from data.mnists.mnist.augmentated_dataset import AugmentedData_random_choose as ValData
from data.mnists.mnist.training_dataset import DefaultMNIST as TrainData
from torch.utils.data import DataLoader

# TODO: select a model
import torch as t
import models
from torch import nn

# Experiment settings.
from config import opt

# Evaluate model performance.
from torchnet import meter
import numpy as np

# Display the experiment process, and recording it.
from tqdm import tqdm
from utils.visualize import Visualizer

from utils.profile import count_parameters_in_MB
from utils.fileio import RecordTrainResult


@t.no_grad()  # pytorch>=0.5
def test(**kwargs):
    opt._parse(kwargs)

    # configure model
    model = getattr(models, opt.model)().eval()
    if opt.pretrained_weight_path:
        model.load(opt.pretrained_weight_path)
    model.to(opt.device)

    # data
    train_data = ValData()
    test_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    results = []
    for ii, (data, path) in tqdm(enumerate(test_dataloader)):
        input = data.to(opt.device)
        score = model(input)
        probability = t.nn.functional.softmax(score, dim=1)[:, 0].detach().tolist()
        # label = score.max(dim = 1)[1].detach().tolist()

        batch_results = [(path_.item(), probability_) for path_, probability_ in zip(path, probability)]

        results += batch_results
    write_csv(results, opt.result_file)

    return results


def write_csv(results, file_name):
    import csv
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(results)


def train(**kwargs):
    opt._parse(kwargs)  # support modify the configs in terminal, but should be used together with "fire.Fire()"
    vis = Visualizer(opt.env, port=opt.vis_port)

    # step 0: set logging files
    checkpoint_path = opt.checkpoint_path  # save model weight, code backups,
    rtr = RecordTrainResult(checkpoint_path=checkpoint_path,)

    # step 1: initialize dataloader
    train_data = TrainData()
    val_data = ValData()
    train_dataloader = DataLoader(train_data, opt.batch_size,
                                  shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, opt.batch_size,
                                shuffle=False, num_workers=opt.num_workers)

    num_classes = train_data.class_to_idx.__len__()

    # step 2: configure model
    model = getattr(models, opt.model)()
    if opt.pretrained_weight_path:
        model.load(opt.pretrained_weight_path)
    model.to(opt.device)
    rtr.add_log("param size = %f MB", count_parameters_in_MB(model))

    # step 3: criterion and optimizer
    if opt.weighted_loss:
        weight = t.FloatTensor([1 / 2, 0.28 / 2, 0.32])
        criterion = nn.CrossEntropyLoss(weight=weight)
    else:
        criterion = nn.CrossEntropyLoss()

    criterion = criterion.to(opt.device)
    lr = opt.lr
    # optimizer = t.optim.SGD(
    #     params=model.parameters(),
    #     lr=lr,
    #     momentum=opt.momentum,
    #     weight_decay=opt.weight_decay
    # )
    optimizer = t.optim.Adam(model.parameters(), lr=lr)

    scheduler = t.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(opt.max_epoch))

    previous_loss = 1e10

    # step 4: training and validation
    for epoch in range(opt.max_epoch):
        # rtr.add_log('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        rtr.add_log('epoch %d lr %e', epoch, lr)
        # train one epoch for multiple iterations
        train_loss, train_cm, train_acc = train_once(train_dataloader, model, criterion, optimizer, num_classes, vis=vis)
        # validate and visualize
        val_cm, val_acc = val_once(model, val_dataloader, num_classes)

        rtr.add_log('train_acc %f', train_acc)
        rtr.add_log('val_acc %f', val_acc)
        rtr.save_model_state_dict(val_acc, epoch, model.state_dict, optimizer.state_dict, save_path=opt.checkpoint_path)

        vis.plot('train_acc', train_acc)
        vis.plot('val_acc', val_acc)
        vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
            epoch=epoch, loss=train_loss, val_cm=str(val_cm.value()), train_cm=str(train_cm),
            lr=lr))

        # update learning rate, method 1.
        scheduler.step()
        # update learning rate, method 2.
        # if loss_meter.value()[0] > previous_loss:
        #     lr = lr * opt.lr_decay
        #     # 第二种降低学习率的方法:不会有moment等信息的丢失
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
        # previous_loss = loss_meter.value()[0]


def train_once(train_dataloader, model, criterion, optimizer, num_classes, vis=None):
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(num_classes)
    loss_meter.reset()
    confusion_matrix.reset()

    for ii, (data, label) in tqdm(enumerate(train_dataloader)):

        # train model
        input = data.to(opt.device)
        target = label.to(opt.device)

        optimizer.zero_grad()

        score = model(input)
        loss = criterion(score, target)

        loss.backward()
        optimizer.step()

        # meters update and visualize
        loss_meter.add(loss.item())
        # detach 一下更安全保险
        confusion_matrix.add(score.detach(), target.detach())

        if (ii + 1) % opt.report_freq == 0:
            if vis is not None:
                vis.plot('loss', loss_meter.value()[0])
            # 进入debug模式
            if os.path.exists(opt.debug_file):
                import ipdb
                ipdb.set_trace()

    cm_value = confusion_matrix.value()
    accuracy = 100. * np.sum([cm_value[i][i] for i in range(num_classes)]) / (cm_value.sum())

    return loss_meter.value()[0], cm_value, accuracy


@t.no_grad()
def val_once(model, dataloader, num_classes):
    """
    计算模型在验证集上的准确率等信息
    """
    model.eval()
    confusion_matrix = meter.ConfusionMeter(num_classes)
    for ii, (val_input, label) in tqdm(enumerate(dataloader)):
        label, angle= label[:,0], label[:,1]
        val_input = val_input.to(opt.device)
        score = model(val_input)
        confusion_matrix.add(score.detach().squeeze(), label.type(t.LongTensor))

    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 100. * np.sum([cm_value[i][i] for i in range(num_classes)]) / (cm_value.sum())
    return confusion_matrix, accuracy


def help():
    """
    打印帮助的信息： python file.py help
    """

    print("""
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:""".format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)


if __name__ == '__main__':
    # TODO : save current config.py, dataset.py, model.py, main.py to the checkpoints
    import fire

    fire.Fire()
