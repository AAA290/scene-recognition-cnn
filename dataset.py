import os
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class Scene15Dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Initialize the Scene-15 dataset.

        Args:
            data_dir (str): Root directory containing per-class sub-folders
                            (e.g., ./data/train or ./data/custom_test).
            transform: Optional torchvision transform pipeline applied to each image.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []  # List of (image_path, integer_label) tuples.

        # Fixed label mapping as required by the project specification (1-indexed).
        self.label_map = {
            "bedroom": 1,     "Coast": 2,       "Forest": 3,
            "Highway": 4,     "industrial": 5,  "Insidecity": 6,
            "kitchen": 7,     "livingroom": 8,  "Mountain": 9,
            "Office": 10,     "OpenCountry": 11, "store": 12,
            "Street": 13,     "Suburb": 14,     "TallBuilding": 15
        }

        # Walk through sub-folders and collect valid image paths with their labels.
        for class_name in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, class_name)

            if os.path.isdir(class_dir) and class_name in self.label_map:
                true_label = self.label_map[class_name]

                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, true_label))

        print(f"Dataset loaded from: {data_dir}  |  Total images: {len(self.samples)}")

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieve a single (image, label) pair by index.

        Images are converted to RGB to handle grayscale inputs that would
        otherwise cause a channel mismatch in the CNN.
        """
        img_path, label = self.samples[idx]

        # Force 3-channel RGB to avoid dimension errors from grayscale images.
        image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ImageNet normalisation statistics (mean and std per channel).
# These values are adopted from the MAE (Meta Platforms, Inc.) open-source implementation.
IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD  = [0.229, 0.224, 0.225]


def build_transforms(is_train, img_size=224):
    """
    Construct the image pre-processing and augmentation pipeline.

    The augmentation strategy is inspired by the MAE (Masked Autoencoders)
    open-source implementation from Meta Platforms, Inc.

    Args:
        is_train (bool): If True, return a training pipeline with random
                         augmentations; otherwise return a deterministic
                         evaluation pipeline.
        img_size (int):  Target spatial resolution. Default is 224.

    Returns:
        torchvision.transforms.Compose: The composed transform pipeline.
    """
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0),
                                         interpolation=PIL.Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ])
    else:
        # Resize slightly larger than the target, then take a centre crop.
        # This preserves the aspect ratio and keeps the main subject centred.
        crop_pct = 224 / 256
        size = int(img_size / crop_pct)
        return transforms.Compose([
            transforms.Resize(size, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ])