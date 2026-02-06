import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageEnhance
import numpy as np


class ContrastAdjustment:
    """
    Randomly adjusts image contrast between 0.1 and 1.5 times the original.
    """
    def __init__(self, contrast_range=(0.1, 1.5)):
        self.contrast_range = contrast_range
    
    def __call__(self, img):
        """
        Args:
            img: PIL Image
        Returns:
            PIL Image with adjusted contrast
        """
        contrast_factor = np.random.uniform(*self.contrast_range)
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(contrast_factor)


def get_train_transforms(img_size=150, use_zoom=True, use_contrast=True, use_translation=True):
    """
    Training preprocessing and augmentations.
    Returns torchvision.transforms.Compose object.
    """
    transform_list = [
        transforms.Resize((img_size, img_size)), #convert to 150x150
    ]
    
    if use_zoom:
        transform_list.append(
            transforms.RandomResizedCrop(
                size=(img_size, img_size),
                scale=(0.85, 1.0),  # 1.0 - 0.85 = 0.15 (15% zoom range)
                ratio=(1.0, 1.0),   # Keep square aspect ratio
                interpolation=transforms.InterpolationMode.BILINEAR
            )
        )
    
    if use_contrast:
        transform_list.append(
            ContrastAdjustment(contrast_range=(1.0, 1.5))
        )
    
    if use_translation:
        transform_list.append(
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),  # horizontal and vertical translation
                fill=0  # fill black
            )
        )
    
    # Converting to tensor which is automatically scaled to [0, 1]
    transform_list.append(transforms.ToTensor())
    
    return transforms.Compose(transform_list)


def get_val_transforms(img_size=150):
    """
    Applying only resizing and scaling.
    for validation data
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

def get_visualization_transforms():
    """
    Returns transforms for visualizing images. Slightly bigger resolution.
    To feed the model during grad-cam visualization.
    """
    return transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
    ])

class ImageDataset(Dataset):
    
    def __init__(self, root_dir, transform=None, class_mode='4-class'):
        """
        root_dir: Root directory (Train, Valid, or Test folder)
        transform: torchvision transforms to apply
        class_mode: '4-class' or '3-class' classification
        """
        self.root_dir = root_dir
        self.transform = transform
        self.class_mode = class_mode
        
        # Define class mappings
        if class_mode == '4-class':
            self.class_names = ['Covid-19', 'Normal', 'Bacterial Pneumonia', 'Viral Pneumonia']
            self.folder_to_class = {
                'Covid-19': 0,
                'Normal': 1,
                'Bacterial Pneumonia': 2,
                'Viral Pneumonia': 3
            }
        elif class_mode == '3-class':
            self.class_names = ['Covid-19', 'Normal', 'Pneumonia']
            self.folder_to_class = {
                'Bacterial Pneumonia': 2,  # Merged to Pneumonia
                'Covid-19': 0,
                'Normal': 1,
                'Viral Pneumonia': 2  # Merged to Pneumonia
            }
        elif class_mode == '2-class':
            # Only Bacterial and Viral (Pneumonia subset)
            self.class_names = ['Bacterial Pneumonia', 'Viral Pneumonia']
            self.folder_to_class = {
                'Bacterial Pneumonia': 0,
                'Viral Pneumonia': 1
            }
        else:
            raise ValueError(f"Invalid class_mode: {class_mode}")
        
        self.images = []
        self.labels = []
        
        # Load all images and labels
        self._load_data()
    
    def _load_data(self):
        # Scan the directory and load all image paths and labels.
        for folder_name in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder_name)
            
            if folder_name not in self.folder_to_class:
                continue
            # Map folder name to class label
            class_label = self.folder_to_class[folder_name]
            
            # Load all images from this folder
            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith(('.jpg')):
                    img_path = os.path.join(folder_path, img_name)
                    self.images.append(img_path)
                    self.labels.append(class_label)
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Returns:
            tuple: (image_tensor, label)
        """
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image if loading fails
            img = Image.new('RGB', (150, 150), (0, 0, 0))
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        return img, label


def create_dataloaders(train_dir, val_dir, test_dir, 
                      batch_size=32, class_mode='4-class', 
                      img_size=150, num_workers=4,
                      use_zoom=True, use_contrast=True, use_translation=True):
    """
    Create DataLoaders for training, validation, and testing.
    
    
    train_dir: Path to training data folder
    val_dir: Path to validation data folder
    test_dir: Path to test data folder
    batch_size: Batch size for data loading
    class_mode: '4-class' or '3-class'
    img_size: Image size (150x150 as per paper)
    num_workers: Number of worker processes for data loading
    use_zoom: Enable/disable zoom augmentation for training (default True)
    use_contrast: Enable/disable contrast augmentation for training (default True)
    use_translation: Enable/disable translation augmentation for training (default True)
    

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Create datasets with appropriate transforms
    train_dataset = ImageDataset(
        root_dir=train_dir,
        transform=get_train_transforms(img_size, use_zoom, use_contrast, use_translation),
        class_mode=class_mode
    )
    
    val_dataset = ImageDataset(
        root_dir=val_dir,
        transform=get_val_transforms(img_size),
        class_mode=class_mode
    )
    
    test_dataset = ImageDataset(
        root_dir=test_dir,
        transform=get_val_transforms(img_size),
        class_mode=class_mode
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, test_loader
    
    
