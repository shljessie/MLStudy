import numpy as np
from numba import njit, prange

@njit(parallel=True)
def convolve3D(input, kernel, stride=1, padding=1):
    # Add padding
    depth, height, width = input.shape
    k_depth, k_height, k_width = kernel.shape

    # Calculate output dimensions
    padded_input = np.pad(
        input,
        ((padding, padding), (padding, padding), (padding, padding)),
        mode='constant',
        constant_values=0
    )
    output_depth = (depth + 2 * padding - k_depth) // stride + 1
    output_height = (height + 2 * padding - k_height) // stride + 1
    output_width = (width + 2 * padding - k_width) // stride + 1

    # Initialize output tensor
    output = np.zeros((output_depth, output_height, output_width))

    # Perform convolution
    for d in prange(output_depth):  # Parallel loop
        for h in range(output_height):
            for w in range(output_width):
                region = padded_input[
                    d * stride:d * stride + k_depth,
                    h * stride:h * stride + k_height,
                    w * stride:w * stride + k_width
                ]
                output[d, h, w] = np.sum(region * kernel)

    return output


# Example Usage
if __name__ == "__main__":
    # Input volume (Depth x Height x Width)
    input_volume = np.random.rand(8, 8, 8)

    # Kernel (Depth x Height x Width)
    kernel = np.random.rand(3, 3, 3)

    # Convolution parameters
    stride = 1
    padding = 1

    # Perform 3D convolution
    output_volume = convolve3D(input_volume, kernel, stride, padding)
    print("Output Volume Shape:", output_volume.shape)



import torch
input_volume = torch.rand(1, 1, 8, 8, 8)  # Batch x Channel x Depth x Height x Width
kernel = torch.rand(1, 1, 3, 3, 3)
conv3d = torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
output_volume = conv3d(input_volume)
print("Output Volume Shape:", output_volume.shape)
