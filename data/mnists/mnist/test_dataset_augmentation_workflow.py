"""
1. load a torch.utils.data.Dataset object, trainset, validset or testset.
2. transform the loaded images, and save the transformed images binary files.
3. create a new torch.utils.data.Dataset object, which loads images from the saved binary files.
This is an offline strategy to augment your dataset. Firstly, you need to run step1 and step2 to load the original data,
transform it to their multiple transformed versions, and save the transformed versions into local disks.
Then, you can directly use the code in step3 to train, validate, or test your model.
"""
import torch
import torchvision
import torchvision.transforms as transforms
import os

# TODO: 1. load a torch.utils.data.Dataset object
DCD_path = "/home/p/Documents/experiment/RotatedPatternRecognition/PreRotation/SWARC/dataset/SCTD++target_normal"
dset_train = os.path.join(DCD_path, "train")
dset_val = os.path.join(DCD_path, "val")
# mean, std = 0.1307, 0.3081
W = 224
data = torchvision.datasets.ImageFolder(dset_train,
                                              transform=transforms.Compose([
                                                  # transforms.RandomRotation((-180, 180)),
                                                  transforms.Resize((W, W)),
                                                  transforms.ToTensor(),
                                                  # transforms.RandomHorizontalFlip(p=0.5),
                                                  # transforms.RandomVerticalFlip(p=0.5),
                                                  # transforms.Normalize((mean,), (std,))
                                              ])
                                              )
from transform_dataset import get_mean_std
print(data[0][0].shape)
mean, std = get_mean_std(data)
print(mean, std)
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
# valid_data = torchvision.datasets.ImageFolder(dset_val,
#                                               transform=transforms.Compose([
#                                                   transforms.Resize((W, W)),
#                                                   # transforms.RandomRotation((-180, 180)),
#                                                   transforms.ToTensor(),
#                                                   # transforms.RandomHorizontalFlip(p=0.5),
#                                                   # transforms.RandomVerticalFlip(p=0.5),
#                                                   transforms.Normalize((mean,), (std,))])
#                                               )

print(train_data[0][0].shape)

img_num = train_data.__len__()

# TODO: 2. transform the loaded imgs, and save the transformed imgs binary files.
from transform_dataset import rotation_augment, save_augmented_dataset, rotation_augment_save
# Augment the tensor dataset, and return the whole dataset
rot_times = 360
# for rot_times in [36, 72, 108, 144, 180, 216, 252, 288, 324, 360]  # , 720, 1440, 2520, 3600]:
rot_range = 360
rotation_augment_save(train_data, rot_range=360, rot_times=rot_times, init_range=0,
                       save_path="rotated_SCTD/rot{}/".format(rot_times),
                       save_type="train",
                       data_name="tensor",
                       target_name="target")

# TODO: 3. create a new torch.utils.data.Dataset object, which loads imgs from the saved binary files
# Directly use the following code to train, validate, or test your model
from augmentated_dataset import AugmentedData
import os

pt_path = "rotated_SCTD/rot360/train/"
train_data_path = os.path.join(pt_path, "tensor")
train_target_path = os.path.join(pt_path, "target")
train_data = AugmentedData(data_path=train_data_path, target_path=train_target_path)


print("=======You have successfully load augmented dataset=======")

# TODO: 4. check if the images are rightly rotated.

from torchvision import transforms
from torchvision.transforms import functional as F

img_ind = 0  # 0,1,2,3,...,N-3,N-2,N-1
ang_ind = 0  # 0,1,2,3,...,357,358,359
ind = (img_ind-1) + ang_ind
image = train_data[ind][0].cpu().clone()  # clone the tensor
image = image.squeeze(0)  # remove the fake batch dimension
# unloader = transforms.ToPILImage()
# image = unloader(image)
image = F.to_pil_image(image)
# image.show()
import numpy as np
import matplotlib.pyplot as plt
# plt.imshow(np.array(image)[:,:,0],cmap="gray")
plt.imshow(np.array(image))
plt.show()


print("----------------")

def mingqiang_implementation():
    """
    import torch
    import torchvision
    import torchvision.transforms as transforms

    # Augment the tensor dataset, and return the whole dataset
    from transform_dataset import rotation_augment

    import numpy as np
    import math
    import cv2
    from PIL import Image

    import os
    from fileio import mkdir
    import pickle

    def main():
        args = parse_args()
        min_scale = args.min_scale
        work_dir = './dataset/MNIST-Scale_'+str(min_scale)+'-1.0/'
        try:
            os.mkdir(work_dir)
        except:
            None

        os.chdir(work_dir)

        transform = transforms.Compose([
            transforms.ToTensor(),  # 0-1
            transforms.Normalize((0.5,), (0.5,))  # single channel
        ])

        trainset = torchvision.datasets.MNIST(root='/media/n/SanDiskSSD/HardDisk/data', train=True,
                                              download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='/media/n/SanDiskSSD/HardDisk/data', train=False,
                                             download=True, transform=transform)

        np.random.seed(0)
        # train_imgs = trainset.data[0:10000, :, :]  # N W H
        # train_labels = trainset.targets[0:10000]
        # test_imgs = testset.data[0:10000, :, :]  # N W H
        # test_labels = testset.targets[0:10000]

        train_imgs = trainset.data  # N W H
        train_labels = trainset.targets
        test_imgs = testset.data  # N W H
        test_labels = testset.targets

        dict = {}
        dict['train_labels'] = train_labels.numpy()  # (10000,)
        pickle.dump(dict, open('train_labels.pickle', 'wb'))

        dict = {}
        dict['test_labels'] = test_labels.numpy()  # (10000,)
        pickle.dump(dict, open('test_labels.pickle', 'wb'))

        normal_train_imgs = np.zeros(train_imgs.shape)
        scaled_test_imgs = np.zeros(test_imgs.shape)

        width, height = train_imgs.shape[1], train_imgs.shape[2]

        for i, (img) in enumerate(train_imgs):
            img = img.numpy()
            img = transform(img)
            normal_train_imgs[i, :, :] = img

        dict = {}
        dict['normal_train_imgs'] = normal_train_imgs  # (10000,)
        pickle.dump(dict, open('normal_train_imgs.pickle', 'wb'))

        for i, (img) in enumerate(test_imgs):
            img = img.numpy()
            random_num = np.random.uniform(min_scale, 1)
            scaled_width = math.floor(width * random_num)
            scaled_height = scaled_width
            scaled_img = cv2.resize(img, (scaled_width, scaled_height))  # W H C
            img_pil = Image.fromarray(scaled_img)
            single_size = 28
            # img=cv2.resize(scaled_img,(single_size,single_size))
            if scaled_width > single_size or scaled_height > single_size:
                img = img_cropping(img_pil, single_size)
            else:
                img = img_padding(img_pil, single_size)
            img = transform(img)  # use transform, convert 0~255 to -1~1
            scaled_test_imgs[i, :, :] = img.squeeze_(0).numpy()
        dict = {}
        dict['scaled_test_imgs'] = scaled_test_imgs  # (10000,)
        pickle.dump(dict, open('scaled_test_imgs.pickle', 'wb'))
    """
    pass
