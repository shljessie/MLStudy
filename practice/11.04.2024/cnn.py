import numpy as np

def cnn(A, k ,stride, padding=1):
  Ah, Aw = A.shape
  kh,kw =k.shape

  oh = (Ah -kh + 2*padding / stride) +1
  ow = (Aw -kw + 2*padding / stride) +1

  output= np.zeros((oh,ow))

  for i in range(Ah-kh+1 , stride):
    for j in range(Aw-kw+1, stride):
      sub_matrix = A[i+kh:j+kw]
      output[i//stride, j//stride] = np.sum(sub_matrix*k)
  

  return output

def cnn_backprop(A, k, dL_dout, stride, padding=1):
  """
  dL_dA, k rotate 180
  dL_dk
  dL_dout , same size as the desired output dimensions
  """

  Ah, Aw = A.shape
  kh, kw =k.shape

    # Initialize gradients for A and k
  dL_dA = np.zeros_like(A)
  dL_dk = np.zeros_like(k)

  output_height, output_width = dL_dout.shape

  #  dL_dk, matrix A and output 
  for i in range(output_height):
    for j in range(output_width):
      sub_matrix = A[i*stride:i*stride + kh, j*stride:j*stride + kw]
      dl_dk += dL_dout[i,j] * sub_matrix

  #  dL_dA, filter k and output

  k_r = np.flip(k)
  for i in range(output_height):
    for j in range(output_width):
      dl_dA[i*stride:i*stride + kh, j*stride:j*stride + kw] +=  dL_dout[i,j] * k_r


  if padding>0:
    dl_dA= dl_dA[padding:-padding, padding:-padding]

  return dl_dk,dl_dA



