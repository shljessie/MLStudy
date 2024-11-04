import numpy as np

# Helper functions for convolution, pooling, and upsampling
def conv2d(x, kernel):
    # Naive 2D convolution with padding
    h, w = x.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    x_padded = np.pad(x, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    out = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            out[i, j] = np.sum(x_padded[i:i+kh, j:j+kw] * kernel)
    return out

def relu(x):
    return np.maximum(0, x)

def max_pool2d(x, size=2):
    h, w = x.shape
    out = np.zeros((h // size, w // size))
    for i in range(0, h, size):
        for j in range(0, w, size):
            out[i // size, j // size] = np.max(x[i:i+size, j:j+size])
    return out

def upsample2d(x, scale=2):
    h, w = x.shape
    out = np.zeros((h * scale, w * scale))
    for i in range(h):
        for j in range(w):
            out[i*scale:(i+1)*scale, j*scale:(j+1)*scale] = x[i, j]
    return out

# Basic convolutional block with Conv2D + ReLU
def conv_block(x, kernel):
    return relu(conv2d(x, kernel))

# U-Net Class
class UNet:
    def __init__(self):
        # Define simple 3x3 kernels for convolution
        self.kernels = {
            'conv1': np.random.randn(3, 3),
            'conv2': np.random.randn(3, 3),
            'conv3': np.random.randn(3, 3),
            'conv4': np.random.randn(3, 3),
            'conv5': np.random.randn(3, 3)
        }

    def forward(self, x):
        # Encoder
        c1 = conv_block(x, self.kernels['conv1'])
        p1 = max_pool2d(c1)

        c2 = conv_block(p1, self.kernels['conv2'])
        p2 = max_pool2d(c2)

        # Bottleneck
        c3 = conv_block(p2, self.kernels['conv3'])

        # Decoder
        u1 = upsample2d(c3)
        c4 = conv_block(u1, self.kernels['conv4'])

        u2 = upsample2d(c4)
        c5 = conv_block(u2, self.kernels['conv5'])

        return c5

# Example usage
if __name__ == "__main__":
    # Input image (grayscale, 64x64)
    input_image = np.random.rand(64, 64)

    # Create U-Net and forward pass
    unet = UNet()
    output_image = unet.forward(input_image)

    print("Input Shape:", input_image.shape)
    print("Output Shape:", output_image.shape)
