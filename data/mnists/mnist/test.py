
# TODO: 3. create a new torch.utils.data.Dataset object, which loads imgs from the saved binary files
# Directly use the following code to train, validate, or test your model
import numpy as np

from augmentated_dataset import AugmentedData
import os

# pt_path = "rotated_SCTD/rot360/train/"
# train_data_path = os.path.join(pt_path, "tensor.pt")
# train_target_path = os.path.join(pt_path, "target.pt")
# train_data = AugmentedData(data_path=train_data_path, target_path=train_target_path)

from augmentated_dataset import AugmentedData
import os

img_num = 146

pt_path = "rotated_SCTD/rot360/train/"
train_data_path = os.path.join(pt_path, "tensor_")
train_target_path = os.path.join(pt_path, "target")
train_data = AugmentedData(data_path=train_data_path, target_path=train_target_path, img_num=img_num)

print("=======You have successfully augmented you dataset=======")

# TODO: 4. check if the images are rightly rotated.

from torchvision import transforms
from torchvision.transforms import functional as F

img_ind = 80  # 0,1,2,3,...,N-3,N-2,N-1
ang_ind = 45  # 0,1,2,3,...,357,358,359
ind = (img_ind-1) + ang_ind
image = train_data[ind][0].cpu().clone()  # clone the tensor
image = image.squeeze(0)  # remove the fake batch dimension
# unloader = transforms.ToPILImage()
# image = unloader(image)
image = image.permute(1, 2, 0)

import matplotlib.pyplot as plt
mean, std =[40.714695, 28.072208, 15.175995], [0.23218124, 0.19987544, 0.12810878]
plt.imshow(image.numpy())
plt.imshow(image.numpy()*std+mean)

print("----------------")