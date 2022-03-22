# 图像旋转插值函数
import cv2
import numpy as np
import torch
import random
import os
from fileio import mkdir
import pickle


def rot_img_tensor(img, img_center, ang, N, init_range, shape=(28, 28), pad=0, **kwargs):
    """
    输入img是个二维tensor， shape=torch.Size(Height,Width,Channel)
    rotation_matrix为计算好的旋转矩阵
    # 输入angle是角度，作为rotation_matrix的索引
    shape为旋转插值后的输出
    pad为边界值 # pad = -0.42421296
    """
    # print(img.shape)
    img = img.numpy()
    grayimg = True if img.shape[2] == 1 else False
    img_rotated_arr = np.empty((N, shape[2], shape[0], shape[1]), dtype=np.float32)
    # random_init_ang = int(random.random() * self.init_range)  # random int
    random_init_ang = (random.random() - 0.5) * init_range  # random float
    if len(kwargs) > 0:
        for i in range(0, N, 1):
            random_rotation_matrix = cv2.getRotationMatrix2D(img_center, ang[i] + random_init_ang, 1)
            img_rotated_arr[i, :, :, :] = cv2.warpAffine(img, random_rotation_matrix, (shape[0], shape[1]),
                                                         flags=kwargs["flags"],
                                                         borderValue=pad).transpose((2, 0, 1))
    else:
        for i in range(0, N, 1):
            random_rotation_matrix = cv2.getRotationMatrix2D(img_center, ang[i] + random_init_ang, 1)
            if grayimg:
                img_rotated_arr[i, 0, :, :] = cv2.warpAffine(img[:, :, 0], random_rotation_matrix, (shape[0], shape[1]),
                                                             borderValue=pad)
            else:
                img_rotated_arr[i, :, :, :] = cv2.warpAffine(img, random_rotation_matrix, (shape[0], shape[1]),
                                                             borderValue=pad).transpose((2, 0, 1))
    return img_rotated_arr, ang + random_init_ang


def rotation_augment(data, rot_range=360, rot_times=360, init_range=0):
    """
    Args:
        data: a torch.utils.data.Dataset object
        rot_range: use to define rotation range (0, rot_range) for N times rotation, default is 360
        rot_times(int): rotation times, default is 360
        init_range(int): use to define the rotation range(-init_range/2,init_range/2), default is 0

    Returns:
        tuple: save (rotated tensors, (targets, angles)) as binary files.
    """
    N = rot_times
    delta_angle = rot_range / N  # 1 degree
    # 计算旋转矩阵
    img_channel, img_height, img_width = data[0][0].shape  # torch.Size([3, 128, 128])
    img_center = (img_width // 2, img_height // 2)
    # N = 0.1
    # ang = list(range(0, 360, N))
    ang = np.linspace(0, rot_range - delta_angle, N)
    data_len = len(data)

    tensor_buffer = torch.empty((data_len * N, img_channel, img_height, img_width))
    target_buffer = torch.empty((data_len * N, 2))
    for index, datum in enumerate(data):
        img, target = datum[0].permute((1, 2, 0)), int(datum[1])
        # to return a PIL Image
        print(type(img))
        print(img.shape)
        # img = Image.fromarray(img.numpy(), mode='RGB')
        img_rotated_arr, ang = rot_img_tensor(img, img_center, ang, N, init_range,
                                              shape=(img_height, img_width, img_channel), pad=-0)

        tensor_buffer[(index * N):((index + 1) * N), :, :, :] = torch.from_numpy(img_rotated_arr)

        target_buffer[(index * N):((index + 1) * N), 0] = torch.tensor(target)

        target_buffer[(index * N):((index + 1) * N), 1] = torch.from_numpy(ang)

    print("=======Rotation Augmentation Done=======")
    return tensor_buffer, target_buffer


def rotation_augment_save(data, rot_range=360, rot_times=360, init_range=0,
                          save_path="rotated_SCTD/rot360/",
                          save_type="train",
                          data_name="tensor",
                          target_name="target",
                          **kwargs
                          ):
    """
    Args:
        data: a torch.utils.data.Dataset object
        rot_range: use to define rotation range (0, rot_range) for N times rotation, default is 360
        rot_times(int): rotation times, default is 360
        init_range(int): use to define the rotation range(-init_range/2,init_range/2), default is 0
        save_path="rotated_SCTD/rot360/",
        save_type="train",
        data_name="tensor",
        target_name="target"
        path = os.path.join(save_path,save_type,data_name+str(img_index)+".pt")
    Returns:
        tuple: save (rotated tensors, (targets, angles)) as binary files.
    """
    N = rot_times
    delta_angle = rot_range / N  # 0.1
    # 计算旋转矩阵
    img_channel, img_height, img_width = data[0][0].shape
    img_center = (img_width // 2, img_height // 2)
    # N = 0.1
    # ang = list(range(0, 360, N))
    ang = np.linspace(0, rot_range - delta_angle, N)
    data_len = len(data)

    target_buffer = torch.empty((data_len * N, 2))

    data_save_path = os.path.join(save_path, save_type)
    target_save_path = os.path.join(save_path, save_type)
    mkdir(data_save_path)
    mkdir(target_save_path)

    for img_index, datum in enumerate(data):
        img, target = datum[0].permute((1, 2, 0)), int(datum[1])
        # to return a PIL Image
        print(type(img))
        print(img.shape)
        img_rotated_arr, ang = rot_img_tensor(img, img_center, ang, N, init_range,
                                              shape=(img_height, img_width, img_channel), pad=-0, **kwargs)
        target_buffer[(img_index * N):((img_index + 1) * N), 0] = torch.tensor(target)
        target_buffer[(img_index * N):((img_index + 1) * N), 1] = torch.from_numpy(ang)

        torch.save(torch.from_numpy(img_rotated_arr), data_save_path + "_" + str(img_index) + ".pt")

    torch.save(target_buffer, target_save_path + ".pt")
    print("=======Rotation Augmentation and Saving Process Done=======")
    # return tensor_buffer, target_buffer
    return None


def get_mean_std(train_data):
    """
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    """
    print('Compute mean and variance for training data.')
    print(len(train_data))
    # train_loader = torch.utils.data.DataLoader(
    #     train_data, batch_size=1, shuffle=False, num_workers=0,
    #     pin_memory=True)
    num_channel = train_data[0][0].shape[0]  # CxHxW
    mean = torch.zeros(num_channel)
    std = torch.zeros(num_channel)
    # for X, _ in train_loader:
    len_data = len(train_data)
    for i in range(len_data):
        idata = train_data[i][0]
        # for d in range(num_channel):
        # mean[d] += X[:, d, :, :].mean()
        # std[d] += X[:, d, :, :].std()
        mean += idata.mean([1, 2])
        std += idata.std([1, 2])
    mean.div_(len_data)
    std.div_(len_data)
    # return list(mean.numpy() * 100), list(std.numpy())
    return list(mean.numpy()), list(std.numpy())


def save_augmented_dataset(data, target,
                           save_path="rotated_SCTD/rot360/",
                           save_type="train",
                           data_name="tensor.pt",
                           target_name="target.pt") -> None:
    save_path += save_type
    mkdir(save_path)
    torch.save(data, os.path.join(save_path, data_name))
    torch.save(target, os.path.join(save_path, target_name))
    print("=======Augmented dataset has been save as binary file in to \"/{}\"=======".format(save_path))
    return None
