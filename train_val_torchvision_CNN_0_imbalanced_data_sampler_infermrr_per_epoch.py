import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset

# models
import torchvision.models as models
from lenet import LeNet5, LeNet51
from lenet import LeNet224S1 as LeNet224
# from torchvision.datasets import MNIST
from torchvision import transforms
from dataset.torchsampler import ImbalancedDatasetSampler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
# from peer_models import SingleBranch as TIPNet
from peer_models import SingleBranchBN4 as TIPNet

from infer_mrr import infer_mrr_per_category

parser = argparse.ArgumentParser("mnist")
parser.add_argument('--data', type=str, default='./mnist', help='location of the data corpus')
parser.add_argument('--set', type=str, default='mnist', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')  # raw: 256
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')  # raw: 0
# parser.add_argument('--epochs', type=int, default=180, help='num of training epochs')
# parser.add_argument('--epochs', type=int, default=60, help='num of training epochs')
parser.add_argument('--epochs', type=int, default=180, help='num of training epochs')
parser.add_argument('--mrr_epochs', type=int, default=30, help='num of initial mrr infer epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=6, help='total number of layers')  # raw: 8
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='layer6epoch50arch', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='PCDARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--net', type=str, default="resnet", help='CNN type')
parser.add_argument('--pretrain', action='store_true', default=False, help='pretrain or not')
parser.add_argument('--weighted_loss', action='store_true', default=False, help='pretrain or not')
args = parser.parse_args()

if args.net == 0:
    Resnet = 1
    Shufflenet = 0
    Squeezenet = 0
if args.net == 1:
    Resnet = 0
    Shufflenet = 1
    Squeezenet = 0

if args.net == 2:
    Resnet = 0
    Shufflenet = 0
    Squeezenet = 1

#TODO 5: CNN input size
W = 224
# W = 32
# W = 112

#TODO 3: Edit path
if args.net == "resnet":
    # args.save = '/home/p/Documents/experiment/RotatedPatternRecognition/PreRotation/' \
    #             'SWARC/checkpoint/stable_data_augmentation_with_angle/' \
    #             'train_0_valid_0_resnet18_pretrained_'+str(args.pretrain)+'_epoch_' + str(args.epochs)\
    #             +"batchsize"+str(args.batch_size)+"_using_imbalanced_dataset_sampler"
    # args.save = '/home/p/Documents/experiment/RotatedPatternRecognition/PreRotation/' \
    #             'SWARC/checkpoint/stable_data_augmentation_with_angle/' \
    #             'train_0_valid_0_resnet34_pretrained_'+str(args.pretrain)+'_epoch_' + str(args.epochs)\
    #             +"batchsize"+str(args.batch_size)+"_using_imbalanced_dataset_sampler"
    # args.save = '/home/p/Documents/experiment/RotatedPatternRecognition/PreRotation/' \
    #             'SWARC/checkpoint/stable_data_augmentation_with_angle/' \
    #             'train_0_valid_0_resnet50_pretrained_'+str(args.pretrain)+'_epoch_' + str(args.epochs)\
    #             +"batchsize"+str(args.batch_size)+"_using_imbalanced_dataset_sampler"
    # args.save = '/home/p/Documents/experiment/RotatedPatternRecognition/PreRotation/' \
    #             'SWARC/checkpoint/stable_data_augmentation_with_angle/' \
    #             'train_0_valid_0_resnet101_pretrained_'+str(args.pretrain)+'_epoch_' + str(args.epochs)\
    #             +"batchsize"+str(args.batch_size)+"_using_imbalanced_dataset_sampler"
    # args.save = '/home/p/Documents/experiment/RotatedPatternRecognition/PreRotation/' \
    #             'SWARC/checkpoint/stable_data_augmentation_with_angle/' \
    #             'infer_mrr_per_epoch_train_0_valid_0_resnet152_pretrained_'+str(args.pretrain)+'_epoch_' + str(args.epochs)\
    #             +"batchsize"+str(args.batch_size)+"_using_imbalanced_dataset_sampler"


    # infer_mrr_per_epoch

    args.save = '/home/p/Documents/experiment/RotatedPatternRecognition/PreRotation/' \
                'SWARC/checkpoint/stable_data_augmentation_with_angle/' \
                'infer_mrr_per_epoch_train_0_valid_0_tv_3_1_resnet18_pretrained_'+str(args.pretrain)+'_epoch_' + str(args.epochs)\
                +"_batchsize"+str(args.batch_size)+"_using_imbalanced_dataset_sampler"
    # args.save = '/home/p/Documents/experiment/RotatedPatternRecognition/PreRotation/' \
    #             'SWARC/checkpoint/stable_data_augmentation_with_angle/' \
    #             'infer_mrr_per_epoch_train_0_valid_0_tv_1_1_resnet34_pretrained_'+str(args.pretrain)+'_epoch_' + str(args.epochs)\
    #             +"_batchsize"+str(args.batch_size)+"_using_imbalanced_dataset_sampler"
    # args.save = '/home/p/Documents/experiment/RotatedPatternRecognition/PreRotation/' \
    #             'SWARC/checkpoint/stable_data_augmentation_with_angle/' \
    #             'infer_mrr_per_epoch_train_0_valid_0_tv_1_1_resnet152_pretrained_'+str(args.pretrain)+'_epoch_' + str(args.epochs)\
    #             +"_batchsize"+str(args.batch_size)+"_using_imbalanced_dataset_sampler"

if args.net == "shufflenet":
    args.save = '/home/p/Documents/experiment/RotatedPatternRecognition/' \
                'PreRotation/SWARC/checkpoint/stable_data_augmentation_with_angle/' \
                'infer_mrr_per_epoch_train_0_valid_0_shufflenet_pretrained_'+str(args.pretrain)+'_epoch_' + str(args.epochs)\
                +"_batchsize_"+str(args.batch_size)+"_using_imbalanced_dataset_sampler"

    # args.save = '/home/p/Documents/experiment/RotatedPatternRecognition/' \
    #             'PreRotation/SWARC/checkpoint/stable_data_augmentation_with_angle/' \
    #             'infer_mrr_per_epoch_train_0_valid_0_shufflenet_x2.0_pretrained_'+str(args.pretrain)+'_epoch_' + str(args.epochs)\
    #             +"_batchsize_"+str(args.batch_size)+"_using_imbalanced_dataset_sampler"


if args.net == "lenet":
    args.save = '/home/p/Documents/experiment/RotatedPatternRecognition/PreRotation/' \
                'SWARC/checkpoint/stable_data_augmentation_with_angle/' \
                'infer_mrr_per_epoch_train_0_valid_0_tv_3_1_lenet_pretrained_'+str(args.pretrain)+'_epoch_' + str(args.epochs)\
                +"_batchsize"+str(args.batch_size)+"_using_imbalanced_dataset_sampler"

    args.save = '/home/p/Documents/experiment/RotatedPatternRecognition/PreRotation/' \
                'SWARC/checkpoint/stable_data_augmentation_with_angle/' \
                'infer_mrr_per_epoch_train_0_valid_0_tv_3_1_lenetS1_pretrained_'+str(args.pretrain)+'_epoch_' + str(args.epochs)\
                +"_batchsize"+str(args.batch_size)+"_using_imbalanced_dataset_sampler"

    args.save = '/home/p/Documents/experiment/RotatedPatternRecognition/PreRotation/' \
                'SWARC/checkpoint/stable_data_augmentation_with_angle/' \
                'infer_mrr_per_epoch_train_0_valid_0_tv_4_1_lenet5_pretrained_'+str(args.pretrain)+'_epoch_' + str(args.epochs)\
                +"_batchsize"+str(args.batch_size)+"_using_imbalanced_dataset_sampler"

    args.save = '/home/p/Documents/experiment/RotatedPatternRecognition/PreRotation/' \
                'SWARC/checkpoint/stable_data_augmentation_with_angle/' \
                'infer_mrr_per_epoch_train_0_valid_0_tv_4_1_bgd_ship_224_lenet5_pretrained_'+str(args.pretrain)+'_epoch_' + str(args.epochs)\
                +"_batchsize"+str(args.batch_size)+"_using_imbalanced_dataset_sampler"


if args.net == "squeezenet":
    # args.save = '/home/p/Documents/experiment/RotatedPatternRecognition/' \
    #             'PreRotation/SWARC/checkpoint/stable_data_augmentation_with_angle/' \
    #             'infer_mrr_per_epoch_train_0_valid_0_squeezenet_pretrained_'+str(args.pretrain)+'_epoch_' + str(args.epochs)\
    #             +"_batchsize_"+str(args.batch_size)+"_using_imbalanced_dataset_sampler"
    args.save = '/home/p/Documents/experiment/RotatedPatternRecognition/PreRotation/' \
                'SWARC/checkpoint/stable_data_augmentation_with_angle/' \
                'infer_mrr_per_epoch_train_0_valid_0_tv_3_1_squeezenet_small_pretrained_'+str(args.pretrain)+'_epoch_' + str(args.epochs)\
                +"_batchsize"+str(args.batch_size)+"_"+str(args.learning_rate)+"_using_imbalanced_dataset_sampler"

if args.net == "mobilenet":
    args.save = '/home/p/Documents/experiment/RotatedPatternRecognition/' \
                'PreRotation/SWARC/checkpoint/stable_data_augmentation_with_angle/' \
                'infer_mrr_per_epoch_train_0_valid_0_mobilenetv3_small_pretrained_'+str(args.pretrain)+'_epoch_' + str(args.epochs)\
                +"_batchsize_"+str(args.batch_size)+"_using_imbalanced_dataset_sampler_test"

if args.net == "mnasnet":
    args.save = '/home/p/Documents/experiment/RotatedPatternRecognition/' \
                'PreRotation/SWARC/checkpoint/stable_data_augmentation_with_angle/' \
                'infer_mrr_per_epoch_train_0_valid_0_mnasnet_small_pretrained_'+str(args.pretrain)+'_epoch_' + str(args.epochs)\
                +"_batchsize_"+str(args.batch_size)+"_using_imbalanced_dataset_sampler_test"

args.save += "_inputsize_"+str(W)+str(args.learning_rate)

# TODO 4： USING_WEIGHTED_LOSS
if args.weighted_loss:
    args.save += "_using_weighted_loss"

# utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
utils.create_exp_dir(args.save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# TODO 1: Augment the Dataset
from dataset.augmentated_dataset import AugmentedData, AugmentedData_random_choose
from dataset.transform_dataset import get_mean_std
import torchvision
import os
# Load train data set
# DCD_path = "/home/p/Documents/experiment/ /PreRotation/SWARC/dataset/SCTD++target_normal"
# train:val = 1:1
DCD_path = "/home/p/Documents/experiment/RotatedPatternRecognition/PreRotation/SWARC/dataset/SCTD++target_normal/train_val/train_val_4_1/"
# DCD_path = "/home/p/Documents/experiment/RotatedPatternRecognition/PreRotation/SWARC/dataset/SCTD++target_normal/train_val/train_val_4_1_bgd_ship/"
dset_train = os.path.join(DCD_path, "train")
data = torchvision.datasets.ImageFolder(dset_train,
                                        transform=transforms.Compose([
                                            transforms.Resize((W, W)),
                                            transforms.ToTensor(),
                                        ]))

mean, std = get_mean_std(data)
train_data = torchvision.datasets.ImageFolder(dset_train,
                                              transform=transforms.Compose([
                                                  # transforms.RandomRotation((-180, 180)),
                                                  transforms.Resize((W, W)),
                                                  transforms.ToTensor(),
                                                  # transforms.RandomHorizontalFlip(p=0.5),
                                                  # transforms.RandomVerticalFlip(p=0.5),
                                                  transforms.Normalize(mean, std)
                                              ])
                                              )
# Load Normal Test Dataset
# DCD_path = "/home/p/Documents/experiment/RotatedPatternRecognition/PreRotation/SWARC/dataset/SCTD++target_normal"
# train:val = 1:1
DCD_path = "/home/p/Documents/experiment/RotatedPatternRecognition/PreRotation/SWARC/dataset/SCTD++target_normal/train_val/train_val_4_1/"
# DCD_path = "/home/p/Documents/experiment/RotatedPatternRecognition/PreRotation/SWARC/dataset/SCTD++target_normal/train_val/train_val_4_1_bgd_ship/"
dset_valid = os.path.join(DCD_path, "val")
valid_data = torchvision.datasets.ImageFolder(dset_valid,
                                              transform=transforms.Compose([
                                                  # transforms.RandomRotation((-180, 180)),
                                                  transforms.Resize((W, W)),
                                                  transforms.ToTensor(),
                                                  # transforms.RandomHorizontalFlip(p=0.5),
                                                  # transforms.RandomVerticalFlip(p=0.5),
                                                  transforms.Normalize(mean, std)
                                              ])
                                              )
# Load Augmented Test dataset
# pt_path = "/home/p/Documents/experiment/RotatedPatternRecognition/PreRotation/SWARC/dataset/rotated_SCTD/rot360/val/"
# train:val = 1:1
pt_path = "/home/p/Documents/experiment/RotatedPatternRecognition/PreRotation/SWARC/dataset/rotated_SCTD_normal/rot360_4_1/val"
pt_path = "/home/p/Documents/experiment/RotatedPatternRecognition/PreRotation/SWARC/dataset/rotated_SCTD_normal/rot360_4_1_bgd_ship_224/val"
valid_data_path = os.path.join(pt_path, "tensor")
valid_target_path = os.path.join(pt_path, "target")
# valid_data_augmented = AugmentedData(data_path=valid_data_path, target_path=valid_target_path, img_num=33)
valid_data_augmented = AugmentedData_random_choose(data_path=valid_data_path, target_path=valid_target_path, sample_interval=60,
                                         img_num=42+1)

# capsule with dataloader
train_queue = DataLoader(
    train_data, batch_size=args.batch_size, sampler=ImbalancedDatasetSampler(train_data), pin_memory=True, num_workers=2)
# train_queue = DataLoader(
#     train_data, batch_size=args.batch_size, shuffle=True, sampler=ImbalancedDatasetSampler(train_data), pin_memory=True, num_workers=2)

valid_queue = DataLoader(
    valid_data, batch_size=2, shuffle=False, pin_memory=True, num_workers=2)

valid_queue_augmented = DataLoader(
    valid_data_augmented, batch_size=16, shuffle=False, pin_memory=True, num_workers=2)

# num_train = len(train_data)
# indices = list(range(num_train))
# split = int(np.floor(0.3 * num_train))
# train_queue = DataLoader(
#     train_data, batch_size=args.batch_size,  # shuffle=True,
#     sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
#     pin_memory=True, num_workers=2)
#
# # valid_queue = train_queue
# valid_queue = DataLoader(
#     valid_data, batch_size=2,  # shuffle=False,
#     sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
#     pin_memory=True, num_workers=2)

# number of classes of targets in the dataset
num_classes = 2

def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    # TODO 6: CUDA_VISIBLE_DEVICES
    # torch.cuda.device_count()
    # torch.cuda.get_device_name(1)
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    torch.cuda.set_device(0)
    # device = torch.device("cuda:1")
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    # TODO 2: Define the model
    if args.net == "resnet":
        model = models.resnet18(pretrained=args.pretrain)
        # model = models.resnet34(pretrained=args.pretrain)
        # model = models.resnet50(pretrained=args.pretrain)
        # model = models.resnet101(pretrained=args.pretrain)
        # model = models.resnet152(pretrained=args.pretrain)

        num_fc_ftr = model.fc.in_features
        model.fc = torch.nn.Linear(num_fc_ftr, num_classes)

    if args.net == "shufflenet":
        # model = models.shufflenet_v2_x0_5(pretrained=False, num_classes=2)
        model = models.shufflenet_v2_x0_5(pretrained=args.pretrain)
        # model = models.shufflenet_v2_x2_0(pretrained=args.pretrain)
        output_channels = model._stage_out_channels[-1]
        model.fc = nn.Linear(output_channels, num_classes)

    if args.net == "squeezenet":
        model = models.squeezenet1_1(pretrained=args.pretrain)
        final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
        model.classifier = nn.Sequential(
            # nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )


    if args.net == "mobilenet":
        model = models.mobilenet_v3_small(pretrained=args.pretrain)
        model.classifier = nn.Sequential(
            nn.Linear(model.lastconv_output_channels, model.last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(model.last_channel, num_classes),
        )

    if args.net == "mnasnet":
        model = models.mnasnet0_5(pretrained=args.pretrain)
        model.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True),
                                        # nn.Linear(1280, num_classes))
                                        nn.Linear(256, num_classes))

    if args.net == "alexnet":
        model = models.alexnet(pretrained=args.pretrain)

    if args.net == "lenet":
        model = LeNet224()
        # model = LeNet5(classes=num_classes)


    model = model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    if args.weighted_loss:
        weight = torch.FloatTensor([1 / 2, 0.28 / 2, 0.32])
        criterion = nn.CrossEntropyLoss(weight=weight)
    else:
        criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    best_valid_acc = 0
    is_best = False
    for epoch in range(args.epochs):

        # model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        test = True
        if test:
            train_acc, train_obj = train(train_queue, model, criterion, optimizer)
            logging.info('train_acc %f', train_acc)

            valid_acc, valid_obj = infer(valid_queue, model, criterion)
            logging.info('valid_acc %f', valid_acc)
        # mrr_per_epoch = False
        mrr_per_epoch = True
        if mrr_per_epoch & (epoch>args.mrr_epochs):
            # valid_acc_mrr, valid_ang_mrr = infer_mrr(model, valid_queue)
            valid_acc_mrr, valid_ang_mrr, _ = infer_mrr_per_category(model, valid_queue_augmented, img_size=(W, W), classes=num_classes)
            logging.info('valid_acc_mrr')
            logging.info(valid_acc_mrr)
            logging.info("valid_acc_mrr.mean() %f", valid_acc_mrr.mean())
            logging.info('valid_ang_mrr')
            logging.info(valid_ang_mrr)
            logging.info("valid_ang_mrr.mean() %f", valid_ang_mrr.mean())
        if test:
            # utils.save(model, os.path.join(args.save, 'weights.pt'))
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                is_best = True
            else:
                is_best = False
            state_set = {
                'epoch': epoch,
                'best_valid_acc': best_valid_acc,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }
            utils.save_checkpoint(state_set, is_best, args.save)
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])


def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = input.cuda()
        target = target.cuda(non_blocking=True)

        optimizer.zero_grad()
        logits = model(input)
        # print(logits.shape, target.shape)
        # print(target)
        # print(logits.shape)
        # print(target.shape)
        loss = criterion(logits, target)
        # print(type(loss))
        # print(loss)
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 2))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)
            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 2))
            n = input.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer_augmented(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target, ang = target[:, 0].int().long(), target[:, 1]
            # print(input.shape)
            # print(target)
            # print(ang)

            target = target.cuda(non_blocking=True)

            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 2))
            n = input.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
