from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# standard image size used by pretrained models like ResNet 224x224
IMAGE_SIZE = 224
BATCH_SIZE = 16

# normalisation values used for pretrained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms():
    # training data: add a bit of randomness to help the model generalise
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # random flip so the model doesn’t rely on left/right orientation
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        # match the format expected by pretrained models
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, eval_transform


def get_datasets(data_dir="data"):
    # load images from folders
    # folder names = class labels (person / not_person)
    data_dir = Path(data_dir)
    train_transform, eval_transform = get_transforms()

    train_dataset = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(data_dir / "val", transform=eval_transform)
    test_dataset = datasets.ImageFolder(data_dir / "test", transform=eval_transform)

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(data_dir="data", batch_size=BATCH_SIZE):
    # wrap datasets into batches
    # shuffle for training so model doesn’t see data in same order
    train_dataset, val_dataset, test_dataset = get_datasets(data_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, train_dataset.classes


if __name__ == "__main__":
    # quick check to make sure everything is working
    train_loader, val_loader, test_loader, classes = get_dataloaders()
    print("Classes:", classes)
    images, labels = next(iter(train_loader))
    print("Image batch shape:", images.shape)
    print("Labels:", labels)