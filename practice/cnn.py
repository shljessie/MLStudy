import numpy as np

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

print(2//2)



def backprop(A, k, dL_dout, stride, padding ):
  if padding >0:
    A = np.pad(A, ((padding, padding), (padding, padding)), mode='constant', constant_values=0)

  # Get dimensions
  Ah, Aw = A.shape
  kh, kw = k.shape
  output_height, output_width = dL_dout.shape

  # Initialize gradients for A and k
  dL_dA = np.zeros_like(A)
  dL_dk = np.zeros_like(k)

  for i in range(output_height):
    for j in range(output_width):
      sub_matrix = A[i*stride:i*stride + kh, j*stride:j*stride + kw]
      dL_dk += dL_dout[i,j] * sub_matrix

  #comput for dL_dA
  k_r =  np.flip(k)
  for i in range(output_height):
    for j in range(output_width):
      dL_dA[i*stride:i*stride+kh,j*stride: j*stride+kw] += dL_dout[i,j] * k_r

  return dL_dA, dL_dk
  



# Input matrix A (e.g., 5x5) and kernel k (e.g., 3x3)
A = np.array([[1, 2, 3, 0, 1],
              [0, 1, 2, 1, 0],
              [1, 0, 2, 1, 2],
              [2, 1, 0, 1, 1],
              [0, 2, 1, 0, 3]])

k = np.array([[1, -1, 0],
              [0, 1, -1],
              [-1, 0, 1]])

stride = 1
padding = 1

# Forward pass
output = cnn(A, k, stride, padding)
print("Output of CNN:\n", output)

# Assume some gradient of loss with respect to output
dL_dout = np.random.randn(*output.shape)  # Random gradients for demonstration

# Backpropagation
dL_dA, dL_dk = cnn_backprop(A, k, dL_dout, stride, padding)
print("Gradient w.r.t. Input A:\n", dL_dA)
print("Gradient w.r.t. Kernel k:\n", dL_dk)
