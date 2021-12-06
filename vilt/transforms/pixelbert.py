from .utils import (
    inception_normalize,
    MinMaxResize,
)
from torchvision import transforms
from .randaug import RandAugment
import albumentations as A
from albumentations.pytorch import ToTensorV2


def pixelbert_transform(size=800):
    longer = int((1333 / 800) * size)
    return transforms.Compose(
        [
            MinMaxResize(shorter=size, longer=longer),
            transforms.ToTensor(),
            inception_normalize,
        ]
    )

def precomputed_transform():
    return A.Compose(
        [
            A.RandomResizedCrop(192, 192, scale=(0.08, 0.5)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(0.8, 0.8, 0.4, 0.2, p=0.8),
            A.ToGray(p=0.2),
            A.GaussianBlur(sigma_limit=[.1, 2.], p=0.2),
            A.Solarize(p=0.2),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2()
        ]
    )

def pixelbert_transform_randaug(size=800):
    longer = int((1333 / 800) * size)
    trs = transforms.Compose(
        [
            MinMaxResize(shorter=size, longer=longer),
            transforms.ToTensor(),
            inception_normalize,
        ]
    )
    trs.transforms.insert(0, RandAugment(2, 9))
    return trs
