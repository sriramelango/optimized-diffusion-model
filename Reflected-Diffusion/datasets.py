"""Return training and evaluation/test datasets from config files."""
import json
import os
import os.path
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as vdsets
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from torch.utils.data import Dataset, DataLoader, DistributedSampler, TensorDataset


def identity(x):
    return x

def cycle_loader(dataloader, sampler=None):
    while 1:
        if sampler is not None:
            sampler.set_epoch(np.random.randint(0, 100000))
        for data in dataloader:
            yield data

# fast class to load all images
class ImageFolderFast(vdsets.VisionDataset):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        self.image_paths = os.listdir(root)
        self.transform = transform

    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.image_paths[index])
        with open(image_path, "rb") as f:
            img = Image.open(f)
            x = img.convert("RGB")
        if self.transform is not None:
            x = self.transform(x)
        return x, #needed to make it consistent: index dataset[0][0] for image

    def __len__(self):
        return len(self.image_paths)

# fast class to load all images
class ImageFolderClassFast(vdsets.VisionDataset):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        with open(os.path.join(root, "dataset.json"), "r") as f:
            self.image_paths = json.load(f)["labels"]
        self.transform = transform

    def __getitem__(self, index):
        pair = self.image_paths[index]
        image_path = os.path.join(self.root, pair[0])
        with open(image_path, "rb") as f:
            img = Image.open(f)
            x = img.convert("RGB")
        if self.transform is not None:
            x = self.transform(x) 
        return x, pair[1]

    def __len__(self):
        return len(self.image_paths)

class GTOHaloTrajectoryDataset(Dataset):
    def __init__(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        self.data = torch.tensor(data, dtype=torch.float32)
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        x = self.data[idx]
        return x, 0  # dummy label for compatibility

class GTOHaloImageDataset(Dataset):
    def __init__(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        self.data = data.astype(np.float32)
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        vec = self.data[idx]
        classifier = np.array([vec[0]], dtype=np.float32)  # first value as label
        # Pad to 81 values (9×9) instead of 72 (8×9) to make it square
        padded = np.pad(vec, (0, 81 - len(vec)), 'constant')
        img = padded.reshape(1, 9, 9)  # 9×9 = 81 values
        return torch.tensor(img, dtype=torch.float32), torch.tensor(classifier, dtype=torch.float32)

class GTOHaloOptimalImageDataset(Dataset):
    """
    Optimal 5×5×3 dataset structure for GTO Halo trajectory generation.
    
    This dataset class transforms 67-dimensional spherical trajectory vectors into 
    semantically meaningful 5×5×3 images for diffusion model training.
    
    Dataset Structure (from spherical dataset README):
    - Input: 67D vector [halo_energy, time_vars(3), thrust_vars(60), mass_vars(3)]
    - Output: 5×5×3 image + halo_energy class label
    
    Image Layout (5×5 = 25 positions):
    [0,0] Time Variables   [0,1] Thrust_0    [0,2] Thrust_1    [0,3] Thrust_2    [0,4] Thrust_3
    [1,0] Thrust_4         [1,1] Thrust_5    [1,2] Thrust_6    [1,3] Thrust_7    [1,4] Thrust_8  
    [2,0] Thrust_9         [2,1] Thrust_10   [2,2] Thrust_11   [2,3] Thrust_12   [2,4] Thrust_13
    [3,0] Thrust_14        [3,1] Thrust_15   [3,2] Thrust_16   [3,3] Thrust_17   [3,4] Thrust_18
    [4,0] Thrust_19        [4,1] Fuel Mass   [4,2] Halo Period [4,3] Manifold Len [4,4] PADDING
    
    Channel Distribution:
    - Time Variables [0,0]: Ch0=shooting_time, Ch1=initial_coast, Ch2=final_coast
    - Thrust Segments [0,1]-[4,0]: Ch0=α_i, Ch1=β_i, Ch2=r_i (spherical coords)
    - Mass Variables [4,1]-[4,3]: All channels get same value (redundant encoding)
    - Padding [4,4]: All channels = 0
    
    Advantages:
    - 96% efficiency (24 data + 1 padding out of 25 positions)
    - Semantic spatial structure (thrust sequence flows naturally)
    - Natural channel groupings (related components share channels)
    - Square image compatible with standard CNN architectures
    """
    
    def __init__(self, pkl_path):
        """
        Initialize the dataset.
        
        Args:
            pkl_path (str): Path to the spherical dataset pickle file
                          (training_data_boundary_100000_spherical.pkl)
        """
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        self.data = data.astype(np.float32)
        
        # Validate dataset dimensions
        expected_dim = 67  # As per spherical dataset documentation
        if self.data.shape[1] != expected_dim:
            raise ValueError(f"Expected {expected_dim}D vectors, got {self.data.shape[1]}D")
            
        print(f"Loaded {len(self.data)} samples with {self.data.shape[1]} dimensions each")
        print("Dataset structure: [halo_energy(1), time_vars(3), thrust_vars(60), mass_vars(3)]")
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        """
        Get a single sample and convert to optimal 5×5×3 image format.
        
        Args:
            idx (int): Sample index
            
        Returns:
            tuple: (image_tensor, class_label_tensor)
                - image_tensor: torch.Tensor of shape (3, 5, 5) containing the trajectory
                - class_label_tensor: torch.Tensor of shape (1,) containing halo energy
        """
        vec = self.data[idx]
        
        # ===== EXTRACT COMPONENTS FROM 67D VECTOR =====
        # Based on spherical dataset documentation structure
        halo_energy = np.array([vec[0]], dtype=np.float32)  # Index [0]: Class label
        time_vars = vec[1:4]                                # Indices [1:4]: Time variables (3 values)
        thrust_vars = vec[4:64].reshape(20, 3)              # Indices [4:64]: Thrust variables (20 segments × 3 spherical coords)  
        mass_vars = vec[64:67]                              # Indices [64:67]: Mass variables (3 values)
        
        # Validate extracted components
        assert time_vars.shape == (3,), f"Expected 3 time variables, got {time_vars.shape}"
        assert thrust_vars.shape == (20, 3), f"Expected 20×3 thrust variables, got {thrust_vars.shape}"
        assert mass_vars.shape == (3,), f"Expected 3 mass variables, got {mass_vars.shape}"
        
        # ===== CREATE 5×5×3 IMAGE =====
        # Initialize with zeros (padding will remain as zeros)
        img = np.zeros((3, 5, 5), dtype=np.float32)
        
        # ===== POSITION [0,0]: TIME VARIABLES =====
        # Time variables are naturally related (all temporal), so they share spatial position
        # with natural channel distribution:
        # - Channel 0: shooting_time (time from Earth to shooting point)
        # - Channel 1: initial_coast (coast time before thrust sequence)  
        # - Channel 2: final_coast (coast time after thrust sequence)
        img[:, 0, 0] = time_vars
        
        # ===== POSITIONS [0,1] THROUGH [4,0]: THRUST SEGMENTS =====
        # Thrust segments flow spatially to represent temporal progression
        # Each position contains one thrust segment with spherical coordinates:
        # - Channel 0: α_i (azimuthal angle)
        # - Channel 1: β_i (polar angle)  
        # - Channel 2: r_i (magnitude) - guaranteed ≤ 1.0 due to spherical representation
        
        # Define thrust positions in spatial order (left-to-right, top-to-bottom)
        thrust_positions = [
            # Row 0: Positions 1-4 (thrust segments 0-3)
            (0, 1), (0, 2), (0, 3), (0, 4),
            # Row 1: All positions (thrust segments 4-8)  
            (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
            # Row 2: All positions (thrust segments 9-13)
            (2, 0), (2, 1), (2, 2), (2, 3), (2, 4),
            # Row 3: All positions (thrust segments 14-18)
            (3, 0), (3, 1), (3, 2), (3, 3), (3, 4),
            # Row 4: Position 0 only (thrust segment 19)
            (4, 0)
        ]
        
        # Validate we have correct number of thrust positions
        assert len(thrust_positions) == 20, f"Expected 20 thrust positions, got {len(thrust_positions)}"
        
        # Fill thrust segments into their spatial positions
        for i, (row, col) in enumerate(thrust_positions):
            # thrust_vars[i] = [α_i, β_i, r_i] for segment i
            img[:, row, col] = thrust_vars[i]
            
        # ===== POSITIONS [4,1], [4,2], [4,3]: MASS VARIABLES =====
        # Mass variables are semantically different, so each gets its own spatial position
        # with redundant channel encoding (all 3 channels get the same value):
        
        # Position [4,1]: Fuel mass (physical spacecraft property in kg)
        img[:, 4, 1] = mass_vars[0]  # fuel_mass replicated across all channels
        
        # Position [4,2]: Halo period (orbital dynamics property in time units)  
        img[:, 4, 2] = mass_vars[1]  # halo_period replicated across all channels
        
        # Position [4,3]: Manifold length (trajectory geometry property, dimensionless)
        img[:, 4, 3] = mass_vars[2]  # manifold_length replicated across all channels
        
        # ===== POSITION [4,4]: PADDING =====
        # This position remains as zeros (initialized above)
        # Could be used for future expansion of the dataset
        
        # ===== RETURN TENSORS =====
        # Convert to PyTorch tensors for training
        image_tensor = torch.tensor(img, dtype=torch.float32)
        class_label_tensor = torch.tensor(halo_energy, dtype=torch.float32)
        
        return image_tensor, class_label_tensor

def get_dataset(config, evaluation=False, distributed=True):
    
    dataroot = config.dataroot
    if config.data.dataset == "CIFAR10":
        
        train_transforms = transforms.Compose(
            [
                transforms.Resize(config.data.image_size),
                transforms.RandomHorizontalFlip() if config.data.random_flip else identity,
                transforms.ToTensor(),
            ]
        )
        test_transforms = transforms.Compose(
            [
                transforms.Resize(config.data.image_size),
                transforms.ToTensor(),
            ]
        )

        train_set = vdsets.CIFAR10(dataroot, train=True, transform=train_transforms)
        test_set = vdsets.CIFAR10(dataroot, train=False, transform=test_transforms)
        workers = 4
    elif config.data.dataset == "ImageNet32":
        data_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        train_set = ImageFolderFast(os.path.join(dataroot, "ds_imagenet", "train_32x32"), transform=data_transforms)
        test_set = ImageFolderFast(os.path.join(dataroot, "ds_imagenet", "valid_32x32"), transform=data_transforms)
        workers = 4
    elif config.data.dataset == "ImageNet64C":
        data_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        train_set = ImageFolderClassFast(os.path.join(dataroot, "imagenet-64x64", "train"), transform=data_transforms)
        test_set = ImageFolderClassFast(os.path.join(dataroot, "imagenet-64x64", "valid"), transform=data_transforms)
        workers = 4
    elif config.data.dataset == "GTOHalo":
        train_set = GTOHaloTrajectoryDataset(config.data.pkl_path)
        test_set = GTOHaloTrajectoryDataset(config.data.pkl_path)
        workers = 4
    elif config.data.dataset == "GTOHaloImage":
        train_set = GTOHaloImageDataset(config.data.pkl_path)
        test_set = GTOHaloImageDataset(config.data.pkl_path)
        workers = 4
    elif config.data.dataset == "GTOHaloOptimalImage":
        train_set = GTOHaloOptimalImageDataset(config.data.pkl_path)
        test_set = GTOHaloOptimalImageDataset(config.data.pkl_path)
        workers = 4
    else:
        raise ValueError(f"{config.data.dataset} is not valid")

    if evaluation:
        if distributed and getattr(config, 'ngpus', 1) > 1:
            sampler = DistributedSampler(test_set, shuffle=False)
        else:
            sampler = None
        test_loader = DataLoader(
            test_set,
            batch_size=config.eval.batch_size,
            sampler=sampler,
            num_workers=workers,
            pin_memory=True,
            shuffle=(sampler is None)
        )
        return test_loader
    else:
        if config.training.batch_size % config.ngpus != 0:
            raise ValueError(f"Train Batch Size {config.training.batch_size} is not divisible by {config.ngpus} gpus.")
        if config.eval.batch_size % config.ngpus != 0:
            raise ValueError(f"Eval Batch Size {config.eval.batch_size} is not divisible by {config.ngpus} gpus.")
        if distributed and getattr(config, 'ngpus', 1) > 1:
            train_sampler = DistributedSampler(train_set)
            test_sampler = DistributedSampler(test_set)
        else:
            train_sampler = None
            test_sampler = None
        train_loader = DataLoader(
            train_set,
            batch_size=config.training.batch_size // config.ngpus,
            sampler=train_sampler,
            num_workers=workers,
            pin_memory=True,
            shuffle=(train_sampler is None),
            persistent_workers=True if workers > 0 else False,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=config.eval.batch_size // config.ngpus,
            sampler=test_sampler,
            num_workers=workers,
            pin_memory=True,
            shuffle=(test_sampler is None),
        )
        train_loader, test_loader = cycle_loader(train_loader, train_sampler), cycle_loader(test_loader, test_sampler)
        return train_loader, test_loader
