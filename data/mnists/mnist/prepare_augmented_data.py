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

from torchvision.datasets import MNIST

# TODO: 1. load a torch.utils.data.Dataset object
def augment_data(data_type="train"):
    # ImageFoler Mode
    read_from_folder = False
    if read_from_folder:
        # DCD_path = "/home/p/Documents/experiment/RotatedPatternRecognition/PreRotation/SWARC/dataset/SCTD++target_normal"
        # DCD_path = "/home/p/Documents/experiment/RotatedPatternRecognition/PreRotation/SWARC/dataset/SCTD++target_normal/train_val/train_val_1_1/"
        DCD_path = "/home/p/Documents/experiment/RotatedPatternRecognition/PreRotation/SWARC/dataset/SCTD++target_normal/train_val/train_val_4_1_bgd/"
        DCD_path = "/home/p/Documents/experiment/RotatedPatternRecognition/PreRotation/SWARC/dataset/SCTD++target_normal/train_val/train_val_4_1_bgd_ship/"

        # ================================================================================== #
        # ------------------ Calculate mean std of training dataset ------------------------
        # ================================================================================== #
        dset = os.path.join(DCD_path, "train")
        # mean, std = 0.1307, 0.3081
        W = 32
        data = torchvision.datasets.ImageFolder(dset,
                                                transform=transforms.Compose([
                                                    transforms.Resize((W, W)),
                                                    transforms.ToTensor(),
                                                ]))

        from transform_dataset import get_mean_std
        print(data[0][0].shape)
        mean, std = get_mean_std(data)
        print(mean, std)

        # ================================================================================== #
        # Load training validation dataset and normalize it using the above-calculated mean and std
        # ================================================================================== #
        dset_data_type = os.path.join(DCD_path, data_type)
        train_data = torchvision.datasets.ImageFolder(dset_data_type,
                                                      transform=transforms.Compose([
                                                          # transforms.RandomRotation((-180, 180)),
                                                          transforms.Resize((W, W)),
                                                          transforms.ToTensor(),
                                                          # transforms.RandomHorizontalFlip(p=0.5),
                                                          # transforms.RandomVerticalFlip(p=0.5),
                                                          transforms.Normalize(mean, std)
                                                      ]))

        print(train_data[0][0].shape)
        print("data.classes: ", train_data.classes)
        print("data.class_to_idx: ", train_data.class_to_idx)

        img_num = train_data.__len__()
    else:
        data = MNIST("./", train=True, download=False,
                           transform=transforms.Compose([
                               # transforms.RandomRotation((-180, 180)),
                               transforms.ToTensor(),
                               # transforms.RandomHorizontalFlip(p=0.5),
                               # transforms.RandomVerticalFlip(p=0.5),
                               # transforms.Normalize((mean,), (std,))
                           ])
                           )
        from transform_dataset import get_mean_std
        print(data[0][0].shape)  # torch.Size([1, 28, 28])
        mean, std = get_mean_std(data)
        print(mean, std)  # [0.13065974] [0.3015038]

        data = MNIST("./", train=True if data_type=="train" else False, download=False,
                     transform=transforms.Compose([
                         # transforms.RandomRotation((-180, 180)),
                         transforms.ToTensor(),
                         # transforms.RandomHorizontalFlip(p=0.5),
                         # transforms.RandomVerticalFlip(p=0.5),
                         transforms.Normalize((mean,), (std,))
                     ])
                     )

    # TODO: 2. transform the loaded imgs, and save the transformed imgs binary files.
    from transform_dataset import rotation_augment, save_augmented_dataset, rotation_augment_save
    # Augment the tensor dataset, and return the whole dataset
    rot_times = 360
    # for rot_times in [36, 72, 108, 144, 180, 216, 252, 288, 324, 360]  # , 720, 1440, 2520, 3600]:
    rot_range = 360
    rotation_augment_save(data, rot_range=360, rot_times=rot_times, init_range=0,
                          # save_path="rotated_SCTD_normal/rot{}_4_1_bgd/".format(rot_times),
                          save_path="./RotatedMnistWithAngle".format(rot_times),
                          # save_path="rotated_SCTD_normal/rot{}_1_1/".format(rot_times),
                          save_type=data_type,
                          data_name="tensor",
                          target_name="target")


# augment_data("train")
augment_data("val")

# Directly use the following code to train, validate, or test your model
if 0:
    from augmentated_dataset import AugmentedData
    import os

    pt_path = "rotated_SCTD/rot360/train/"
    train_data_path = os.path.join(pt_path, "tensor")
    train_target_path = os.path.join(pt_path, "target")
    train_data = AugmentedData(data_path=train_data_path, target_path=train_target_path)


    print("=======You have successfully load augmented dataset=======")