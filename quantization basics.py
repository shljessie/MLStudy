""""

The basic idea behind quantization is quite easy: going from high-precision representation (usually the regular 32-bit floating-point) for weights and activations to a lower precision data type. The most common lower precision data types are:

float16, accumulation data type float16
bfloat16, accumulation data type float32
int16, accumulation data type int32
int8, accumulation data type int32
The accumulation data type specifies the type of the result of accumulating (adding, multiplying, etc) values of the data type in question. For example, let’s consider two int8 values A = 127, B = 127, and let’s define C as the sum of A and B:

Copied
C = A + B
Here the result is much bigger than the biggest representable value in int8, which is 127. Hence the need for a larger precision data type to avoid a huge precision loss that would make the whole quantization process useless.

float32 -> float16 and float32 -> int8.



"""

x = S * (x_q - Z)

x_q = round(x/S + Z)

x_q = clip(round(x/S + Z), round(a/S + Z), round(b/S + Z))