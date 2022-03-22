from torchvision.datasets import MNIST
from torchvision import transforms
from typing import Callable, Optional

# transformation for training dataset
transformation = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            ])

class DefaultMNIST(MNIST):

    def __init__(
            self,
            root="./data/mnists/mnist/",
            train: bool = True,
            transform=transformation,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super(DefaultMNIST, self).__init__(root=root,
                                           train=train,
                                           transform=transform,
                                           target_transform=target_transform,
                                           download=download)


if __name__ == "__main__":
    dataset = DefaultMNIST()
    print(dataset.__len__(), dataset.class_to_idx)
    print("--End--")
