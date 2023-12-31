"""
    Utility file consisting of common functions and variables used during training and evaluation
"""

import json
import torch
import torchvision.transforms as transforms
import utils.enums as enums
from sklearn.metrics import precision_score, recall_score, f1_score
import torchvision.transforms.functional as F


# Basic Image Transform to convert to Pytorch tensor for GPU training
image_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Image Transform to apply Color Jitter augmentation (used only during training)
image_transform_jitter = transforms.Compose([transforms.ToPILImage(), transforms.ColorJitter(hue=.2, saturation=.2),
                                             transforms.ToTensor(),  transforms.Normalize([0.5] * 3, [0.5] * 3)])

# Image Transform to apply Random Flip augmentation (used only during training)
image_transform_flip = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.5),
                                           transforms.RandomRotation(10), transforms.ToTensor(),
                                           transforms.Normalize([0.5] * 3, [0.5] * 3)])

# Image Transform to apply Color Jitter and Random Flip augmentation (used only during training)
image_transform_jitter_flip = transforms.Compose([transforms.ToPILImage(), transforms.ColorJitter(hue=.2, saturation=.2),
                                                  transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(10),
                                                  transforms.ToTensor(),  transforms.Normalize([0.5] * 3, [0.5] * 3)])


# Image Transform to apply RandomBrightness augmentation (used only during training)
image_transform_brightness = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# Image Transform to apply RandomShear augmentation (used only during training)
image_transform_shear = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=15),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# Image Transform to apply RandomBrightness and RandomShear augmentations (used only during training)
image_transform_brightness_shear = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ColorJitter(brightness=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=15),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])


# Image Transform to apply RandomTranslation and RandomRotation augmentation (used only during training)
image_transform_translation_rotation = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])


# Dictionary to convert cloth categories to classes for numerical purposes
CLOTH_CATEGORIES = {enums.SAREE: 0,
                    enums.WOMEN_KURTA: 1,
                    enums.LEHENGA: 2,
                    enums.BLOUSE: 3,
                    enums.GOWNS: 4,
                    enums.DUPATTAS: 5,
                    enums.LEGGINGS_AND_SALWARS: 6,
                    enums.PALAZZOS: 7,
                    enums.PETTICOATS: 8,
                    enums.MOJARIS_WOMEN: 9,
                    enums.DHOTI_PANTS: 10,
                    enums.KURTA_MEN: 11,
                    enums.NEHRU_JACKETS: 12,
                    enums.SHERWANIS: 13,
                    enums.MOJARIS_MEN: 14,
                    enums.MEN_PAGDI: 15,
                    enums.WOMEN_ANARKALI_KURTA: 16,
                    enums.WOMEN_A_LINE_KURTA: 17
                    }


def read_json_data(file_name):
    """
        Utility function to read data from json file

        Args:
            file_name (str): Path to json file to be read

        Returns:
            article_list (List[dict]): List of dict that contains metadata for each item
    """
    with open(file_name) as f:
        article_list = [json.loads(line) for line in f]
        return article_list


def get_accuracy(y_pred, y_actual):
    """
        Utility function to compute accuracy for the minibatch

        Args:
            y_pred (Tensor): Predicted class labels
            y_actual (Tensor): Ground Truth class labels
    """
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == y_actual).sum().item()
    return correct


def calculate_metrics(y_true, y_pred):
    """
        Utility function to calculate Precision, Recall, F1-score, and Error Rate

        Args:
            y_true (List): True class labels
            y_pred (List): Predicted class labels

        Returns:
            precision (float): Precision score
            recall (float): Recall score
            f1 (float): F1-score
            error_rate (float): Error rate
    """
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    error_rate = 1.0 - (precision + recall + f1) / 3.0
    return precision, recall, f1, error_rate
