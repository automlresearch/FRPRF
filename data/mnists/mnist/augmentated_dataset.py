import torch
from torch.utils.data import Dataset
import numpy as np


class AugmentedData(Dataset):
    def __init__(self, data_path="./RotatedMnistWithAngle/val", target_path="./RotatedMnistWithAngle/val", img_num=10000
                 , augment_time=360, suffix=".pt") -> None:
        self.img_id = -1
        self.data = None
        self.data_path = data_path
        self.target = torch.load(target_path+suffix)
        if img_num is None:
            print("image num of raw image dataset is not provided!")
        self.img_num = img_num
        self.augment_time = augment_time
        self.augment_img_num = self.img_num * self.augment_time

        self.suffix = suffix

    def __len__(self):
        return self.augment_img_num

    def get_labels(self):
        target = self.target[:, 0].int().long()
        return target

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # print(index)
        img_id, ang_id = divmod(index, self.augment_time)
        # ang_id -= 1
        if self.img_id != img_id:
            self.img_id = img_id
            self.data = torch.load(self.data_path+"_"+str(self.img_id)+self.suffix)

        img = self.data[ang_id]
        target = self.target[index]

        return img, target


class AugmentedData_random_choose(Dataset):

    def __init__(self, data_path="./data/mnists/mnist/RotatedMnistWithAngle/val",
                 target_path="./data/mnists/mnist/RotatedMnistWithAngle/val", img_num=10000
                 , augment_time=360, sample_interval=60, suffix=".pt") -> None:
        self.img_id = -1
        self.data = None
        self.data_path = data_path
        self.target = torch.load(target_path+suffix)
        if img_num is None:
            print("image num of raw image dataset is not provided!")
        self.img_num = img_num
        self.augment_time = augment_time
        augment_img_num = img_num * augment_time  # Overall imgs
        sample_img_num = int(augment_img_num/sample_interval)
        index_random = list(np.sort(np.random.randint(0, augment_img_num, sample_img_num)))  # random numbers
        self.index_random = index_random
        self.index_random_len = sample_img_num
        self.suffix = suffix

    def __len__(self):
        return self.index_random_len

    def get_labels(self):
        target = self.target[:, 0].int().long()
        return list(target)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # print(index)
        index_random = self.index_random[index]
        img_id, ang_id = divmod(index_random, self.augment_time)
        # ang_id -= 1
        # if self.img_id != img_id:
        #     self.img_id = img_id
        #     self.data = torch.load(self.data_path+"_"+str(self.img_id)+self.suffix)
        img_id = img_id
        data = torch.load(self.data_path+"_"+str(img_id)+self.suffix)

        img = data[ang_id]
        target = self.target[index_random]

        return img, target



# train_data = AugmentedData_single_pt_file(data_path=train_data_path, target_path=train_target_path)
class AugmentedData_single_pt_file(Dataset):
    def __init__(self, data_path: str, target_path: str) -> None:
        self.data = torch.load(data_path)
        self.target = torch.load(target_path)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        return img, target
