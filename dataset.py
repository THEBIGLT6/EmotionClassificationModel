from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders( pathToData, batch_size=32, shuffle=True, num_workers=2 ):

    transform = transforms.Compose([ transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5]) ])

    dataset = datasets.ImageFolder( root=pathToData, transform=transform )

    loader = DataLoader( dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True )

    return loader, dataset