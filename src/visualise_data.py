import matplotlib.pyplot as plt

from data import get_dataloaders

def show_batch(images, labels, num_images=8):
    """
    show a small batch if images
    images shape iwll be [batch_size, 3, H, W]
    labels shape will be [batch_size]
    we convert the images into a format that
    matplotlib undestands
    """
    # create a grid of 2 rows and 4 columns
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()

    for i in range(num_images):
        # pytorch uses [C, H, W]
        # whilst matpltlib needs [H, W, C]
        image = images[i].permute(1, 2, 0)

        # convert tensor to a numpy arrayt
        image = image.cpu().numpy()

        image = image.clip(0, 1)

        axes[i].imshow(image)
        axes[i].set_title(f"Class: {labels[i].item()}")
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.savefig("sample_batch.png")
    print("Saved image to sample_batch.png")

def main():
    """
    Load in a batch of training data and display it
    """

    train_loader, _, _ = get_dataloaders()
    
    # get one batch from the training set
    images, labels = next(iter(train_loader))

    print("Batch shape: ", images.shape)
    print("Labels: ", labels)

    show_batch(images, labels)

if __name__ == "__main__":
    main()