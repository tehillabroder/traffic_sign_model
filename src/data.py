from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# GTSRB images vary in size, so we need a fixed size for batching
# Resize to a modest square 48x48 for initial custom CNN
IMAGE_SIZE = 48
BATCH_SIZE = 64

def get_transforms():
    # training data: add a bit of randomness to help the model generalise
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    return train_transform, eval_transform


def get_datasets(data_dir="data"):
    train_transform, eval_transform = get_transforms()

    full_train_dataset = datasets.GTSRB(
        root=data_dir,
        split="train",
        download=True,
        transform=train_transform
    )
    test_dataset = datasets.GTSRB(
        root=data_dir,
        split="test",
        download=True,
        transform=eval_transform
    )

    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size]
    )

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(data_dir="data", batch_size=BATCH_SIZE):
    # wrap datasets into batches
    # shuffle for training so model doesn’t see data in same order
    train_dataset, val_dataset, test_dataset = get_datasets(data_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # quick check to make sure everything is working
    train_loader, val_loader, test_loader = get_dataloaders()

    images, labels = next(iter(train_loader))
    print("Image batch shape: ", images.shape)
    print("Labels batch shape: ", labels)
    print("Number of classes: ", 43)