from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders( pathToData, batch_size=32, shuffle=True, num_workers=2, is_train=True ):

    # Add data augmentation for training set only
    if is_train:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),

            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.RandomAffine( degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),

            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    else:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    dataset = datasets.ImageFolder( root=pathToData, transform=transform )

    loader = DataLoader( dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True )

    return loader, dataset