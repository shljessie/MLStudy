import numpy as np

#numpy.pad(array, pad_width, mode='constant', **kwargs)
# pad width : For example, for a 2D array, it can be ((pad_top, pad_bottom), (pad_left, pad_right)).


def cnn(A, k ,stride, padding=1):
  # set padding
  if padding > 0:
      A = np.pad(A, ((padding, padding), (padding, padding)), mode='constant', constant_values=0)

  #get height and width
  Ah, Aw = A.shape
  kh, kw = k.shape
  output_height = ((Ah - kh + 2*padding) / stride) +1 
  output_width = ((Aw - kw + 2*padding) / stride) +1 

  # initialize output matrix
  output = np.zeros((output_height, output_width))

  #loop through
  #The highest value of i for which the kernel still fits within the matrix is when the kernel’s last row reaches the input’s last row.
  for i in range(0, Ah - kh + 1, stride):
    for j in range(0, Aw - kw + 1, stride):
      sub_matrix = A[i:i+kh, j:j+kw]
      output[i//stride, j//stride] = np.sum(sub_matrix * k)

  return output


def cnn(A,k,stride,padding =1):
  # add padding 
  if padding > 0: 
    A = np.pad(A, ((padding,padding), (padding,padding)),mode="constant", constant_values=0)

  # get height
  Ah, Aw = A.shape
  kh, kw = k.shape
  oh = ((Ah-kh + 2*padding)//stride) +1
  ow = ((Aw-kw + 2*padding)//stride) +1 

  # initialize matrix
  output = np.zeros((oh,ow))

  #loop through 
  for i in range(0,Ah-kh+1, stride):
    for j in range(0,Aw-kw+1, stride):
      sub_matrix = A[i:i+kh, j:j+kw]
      output[i//stride, j //stride] = np.sum(sub_matrix * k)
  
  return output
      

# Example usage
A = np.array([[1, 2, 3, 0],
              [0, 1, 2, 3],
              [3, 1, 0, 2],
              [2, 3, 1, 0]])

k = np.array([[1, 0],
              [0, -1]])

stride = 1
padding = 1 

print(cnn(A, k, stride, padding))

def cnn_backprop(A, k, dL_dout, stride, padding=1):
  if padding > 0:
        A = np.pad(A, ((padding, padding), (padding, padding)), mode='constant', constant_values=0)

  # Get dimensions
  Ah, Aw = A.shape
  kh, kw = k.shape
  output_height, output_width = dL_dout.shape

  # Initialize gradients for A and k
  dL_dA = np.zeros_like(A)
  dL_dk = np.zeros_like(k)

  # Compute gradient with respect to the kernel (dL_dk)
  for i in range(output_height):
      for j in range(output_width):
          sub_matrix = A[i*stride:i*stride + kh, j*stride:j*stride + kw]
          dL_dk += dL_dout[i, j] * sub_matrix

  # Compute gradient with respect to the input (dL_dA)
  k_rotated = np.flip(k)  # Flip kernel for transposed convolution

  for i in range(output_height):
      for j in range(output_width):
          dL_dA[i*stride:i*stride + kh, j*stride:j*stride + kw] += dL_dout[i, j] * k_rotated

  # Remove padding from dL_dA if padding was added
  if padding > 0:
      dL_dA = dL_dA[padding:-padding, padding:-padding]

  return dL_dA, dL_dk

def maxPooling(feature_map, pool_size=2, stride=2 ):
  fh, fw = feature_map.shape

  # Calculate output dimensions
  out_h = (fh - pool_size) // stride + 1
  out_w = (fw - pool_size) // stride + 1

  #initialize output matrix
  output = np.zeros((out_h, out_w))

  # Perform max pooling
  for i in range(0, fh - pool_size + 1, stride):
      for j in range(0, fw - pool_size + 1, stride):
          sub_region = feature_map[i:i+pool_size, j:j+pool_size]
          output[i//stride, j//stride] = np.max(sub_region)

  return output