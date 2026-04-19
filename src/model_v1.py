import torch
import torch.nn as nn


class TrafficSignCNNv1(nn.Module):
    """
    v1 is a simple CNN for GTSRB traffic sign classification

    The input image shape is [batch_size, 3, 48, 48]
    The output shape is [batch_size, 43]
    """

    def __init__(self, num_classes=43):
        super().__init__()

        # First convolution:
        # in_channels=3 because RGB image has 3 colour channels
        # out_channels=16 means that this layer will learn 16 feature maps
        # kernel_size=3 means a 3x3 filter
        # padding=1 keeps the height and width the same after the convolution
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)

        # ReLU adds non-linearity so the network can learn more complex patterns
        self.relu1 = nn.ReLU()

        # Max pooling reduces spatial size by taking the strongest value in each 2x2 area
        # This halves height and width
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolution:
        # input now has 16 channels because conv1 produced 16 feature maps
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Third convolution:
        # now we go from 32 feature maps to 64
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # After 3 pool layers:
        # 48x48 -> 24x24 -> 12x12 -> 6x6
        # So the tensor shape becomes:
        # [batch_size, 64, 6, 6]

        # Flatten converts the 3D feature maps (channels, height, width)
        # into a 1D vector per image
        # New shape: [batch_size, 2304]
        self.flatten = nn.Flatten()

        # Fully connected layer:
        # 64 feature maps * 6 * 6 = 2304 input features
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.relu_fc1 = nn.ReLU()

        # Final layer outputs one score per class (43 classes for GTSRB)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Defines how the image flows through the network.

        The convolutions extract features, pooling reduces size, and
        then we flatten and use fully connected layers to predict the class.
        """
        # input: [batch_size, 3, 48, 48]

        x = self.conv1(x)
        # shape: [batch_size, 16, 48, 48]

        x = self.relu1(x)

        x = self.pool1(x)
        # shape: [batch_size, 16, 24, 24]

        x = self.conv2(x)
        # shape: [batch_size, 32, 24, 24]

        x = self.relu2(x)

        x = self.pool2(x)
        # shape: [batch_size, 32, 12, 12]

        x = self.conv3(x)
        # shape: [batch_size, 64, 12, 12]

        x = self.relu3(x)

        x = self.pool3(x)
        # shape: [batch_size, 64, 6, 6]

        x = self.flatten(x)
        # shape: [batch_size, 2304]

        x = self.fc1(x)
        # shape: [batch_size, 128]

        x = self.relu_fc1(x)

        x = self.fc2(x)
        # shape: [batch_size, 43]

        return x


if __name__ == "__main__":
    # Quick check that the model works with a fake batch
    model = TrafficSignCNNv1()

    dummy_input = torch.randn(8, 3, 48, 48)
    output = model(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)