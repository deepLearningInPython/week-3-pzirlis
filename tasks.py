import numpy as np

# Follow the tasks below to practice basic Python concepts.
# Write your code in between the dashed lines.
# Don't import additional packages. Numpy suffices.


# Task 1: Compute Output Size for 1D Convolution
# Instructions:
# Write a function that takes two one-dimensional numpy arrays (input_array, kernel_array) as arguments.
# The function should return the length of the convolution output (assuming no padding and a stride of one).
# The output length can be computed as follows:
# (input_length - kernel_length + 1)

# Your code here:
# -----------------------------------------------

def compute_output_size_1d(input_array, kernel_array):
    return (len(input_array) - len(kernel_array) + 1)


# -----------------------------------------------
# Example:
input_array = np.array([1, 2, 3, 4, 5])
kernel_array = np.array([1, 0, -1])
print(compute_output_size_1d(input_array, kernel_array))


# Task 2: 1D Convolution
# Instructions:
# Write a function that takes a one-dimensional numpy array (input_array) and a one-dimensional kernel array (kernel_array)
# and returns their convolution (no padding, stride 1).

# Your code here:
# -----------------------------------------------

def convolve_1d(input_array, kernel_array):
    # Start by initializing an empty output array
    size = (len(input_array) - len(kernel_array) + 1)
    conv = np.empty(size)

    # Then fill the cells in the array with a loop.
    for i in range(size):
        conv[i] = (input_array[i:i+3] @ kernel_array)

    return conv

# -----------------------------------------------
# Another tip: write test cases like this, so you can easily test your function.
input_array = np.array([1, 2, 3, 4, 5])
kernel_array = np.array([1, 0, -1])
print(convolve_1d(input_array, kernel_array))

# Task 3: Compute Output Size for 2D Convolution
# Instructions:
# Write a function that takes two two-dimensional numpy matrices (input_matrix, kernel_matrix) as arguments.
# The function should return a tuple with the dimensions of the convolution of both matrices.
# The dimensions of the output (assuming no padding and a stride of one) can be computed as follows:
# (input_height - kernel_height + 1, input_width - kernel_width + 1)

# Your code here:
# -----------------------------------------------

def compute_output_size_2d(input_matrix, kernel_matrix):
    # Get heights and widths
    input_height, input_width   = np.shape(input_matrix)
    kernel_height, kernel_width = np.shape(kernel_matrix)
    
    # Dimension of convolution
    dimensions = (input_height - kernel_height + 1, input_width - kernel_width + 1)
    return dimensions

# -----------------------------------------------
# Test case
input_matrix = np.matrix([[1,2,3,4,5],
                          [6,7,8,9,10],
                          [11,12,13,14,15],
                          [16,17,18,19,20],
                          [21,22,23,24,25]])
kernel_matrix = np.matrix([[-1,0,1],[-1,0,1],[-1,0,1]])
compute_output_size_2d(input_matrix, kernel_matrix)

# Task 4: 2D Convolution
# Instructions:
# Write a function that computes the convolution (no padding, stride 1) of two matrices (input_matrix, kernel_matrix).
# Your function will likely use lots of looping and you can reuse the functions you made above.

# Your code here:
# -----------------------------------------------
def convolute_2d(input_matrix, kernel_matrix):
    # Dimensions
    out_h, out_w = compute_output_size_2d(input_matrix, kernel_matrix)
    kH, kW = kernel_matrix.shape
    conv = np.empty(shape = (out_h, out_w))

    # Convolution (nested loop)
    for i in range(out_h):
        for j in range(out_w):
            window = input_matrix[i:i+kH, j:j+kW] # window of input to apply kernel
            conv[i,j] = np.sum(kernel_matrix * window)
    
    return conv

# -----------------------------------------------
# Test case
input_matrix = np.matrix([[1,2,3,4,5],
                          [6,7,8,9,10],
                          [11,12,13,14,15],
                          [16,17,18,19,20],
                          [21,22,23,24,25]])
kernel_matrix = np.matrix([[-1,0,1],[-1,0,1],[-1,0,1]])
convolute_2d(input_matrix, kernel_matrix)
