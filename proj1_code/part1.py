#!/usr/bin/python3

from typing import Tuple

import numpy as np

def create_Gaussian_kernel_1D(ksize: int, sigma: int) -> np.ndarray:
    """Create a 1D Gaussian kernel using the specified filter size and standard deviation.
    
    The kernel should have:
    - shape (k,1)
    - mean = floor (ksize / 2)
    - values that sum to 1
    
    Args:
        ksize: length of kernel
        sigma: standard deviation of Gaussian distribution
    
    Returns:
        kernel: 1d column vector of shape (k,1)
    
    HINT:
    - You can evaluate the univariate Gaussian probability density function (pdf) at each
      of the 1d values on the kernel (think of a number line, with a peak at the center).
    - The goal is to discretize a 1d continuous distribution onto a vector.
    """
    
    """
    Used the concept of scale ratio from Open Cv's implementation
    Reference
    [1] https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gac05a120c1ae92a6060dd0db190a61afa
    """
    ksize = int(ksize)
    sigma = int(sigma)
    mean = (ksize-1) * 0.5
    output = np.arange(0,ksize)
    sqr_sigma = np.power(sigma,2)
    num = output - mean
    sqr_num = np.power(num, 2)
    exp = sqr_num / (-2 * sqr_sigma)
    exponent = np.exp(exp)
    sum = np.sum(exponent)
    scaleRatio = 1 / sum
    kernel = np.multiply(exponent, scaleRatio)
    return kernel.reshape(kernel.shape[0], 1)
    raise NotImplementedError(
        "`create_Gaussian_kernel_1D` function in `part1.py` needs to be implemented"
    )

    
def create_Gaussian_kernel_2D(cutoff_frequency: int) -> np.ndarray:
    """
    Create a 2D Gaussian kernel using the specified filter size, standard
    deviation and cutoff frequency.

    The kernel should have:
    - shape (k, k) where k = cutoff_frequency * 4 + 1
    - mean = floor(k / 2)
    - standard deviation = cutoff_frequency
    - values that sum to 1

    Args:
        cutoff_frequency: an int controlling how much low frequency to leave in
        the image.
    Returns:
        kernel: numpy nd-array of shape (k, k)

    HINT:
    - You can use create_Gaussian_kernel_1D() to complete this in one line of code.
    - The 2D Gaussian kernel here can be calculated as the outer product of two
      1D vectors. In other words, as the outer product of two vectors, each 
      with values populated from evaluating the 1D Gaussian PDF at each 1d coordinate.
    - Alternatively, you can evaluate the multivariate Gaussian probability 
      density function (pdf) at each of the 2d values on the kernel's grid.
    - The goal is to discretize a 2d continuous distribution onto a matrix.
    """
    ############################
    ### TODO: YOUR CODE HERE ###
    ksize = cutoff_frequency * 4 + 1
    sigma = cutoff_frequency
    oned_kernel = create_Gaussian_kernel_1D(ksize, sigma)
    kernel = np.dot(oned_kernel, np.transpose(oned_kernel))
    return kernel

def convolve(matrix: np.ndarray, filter: np.ndarray):
    mul = np.multiply(matrix, filter)
    return np.sum(mul)

def prepConvolve(matrix: np.ndarray, filter: np.ndarray) -> np.ndarray:
    verticalPadding = filter.shape[0] // 2
    horizontalPadding = filter.shape[1] // 2
    updatedMatrix = np.concatenate((np.zeros(matrix.shape[1] * verticalPadding).reshape(verticalPadding, matrix.shape[1]),matrix), axis = 0)
    updatedMatrix = np.concatenate((updatedMatrix,np.zeros(matrix.shape[1] * verticalPadding).reshape(verticalPadding, matrix.shape[1])), axis = 0)
    updatedMatrix = np.concatenate((np.zeros(updatedMatrix.shape[0] * horizontalPadding).reshape(updatedMatrix.shape[0], horizontalPadding),updatedMatrix), axis = 1)
    updatedMatrix = np.concatenate((updatedMatrix,np.zeros(updatedMatrix.shape[0] * horizontalPadding).reshape(updatedMatrix.shape[0], horizontalPadding)), axis = 1)
    output = np.zeros(matrix.shape[0]*matrix.shape[1]).reshape(matrix.shape[0],matrix.shape[1])
    for i in range(0,matrix.shape[0]):
      for j in range(0,matrix.shape[1]):
        output[i,j] = convolve(updatedMatrix[i:i+filter.shape[0],j:j+filter.shape[1]], filter)
    #print(updatedMatrix[verticalPadding:updatedMatrix.shape[0]-verticalPadding,horizontalPadding:updatedMatrix.shape[1]-horizontalPadding])

    return output

def my_conv2d_numpy(image: np.ndarray, filter: np.ndarray) -> np.ndarray:
    """Apply a single 2d filter to each channel of an image. Return the filtered image.
    
    Note: we are asking you to implement a very specific type of convolution.
      The implementation in torch.nn.Conv2d is much more general.

    Args:
        image: array of shape (m, n, c)
        filter: array of shape (k, j)
    Returns:
        filtered_image: array of shape (m, n, c), i.e. image shape should be preserved

    HINTS:
    - You may not use any libraries that do the work for you. Using numpy to
      work with matrices is fine and encouraged. Using OpenCV or similar to do
      the filtering for you is not allowed.
    - We encourage you to try implementing this naively first, just be aware
      that it may take an absurdly long time to run. You will need to get a
      function that takes a reasonable amount of time to run so that the TAs
      can verify your code works.
    - If you need to apply padding to the image, only use the zero-padding
      method. You need to compute how much padding is required, if any.
    - "Stride" should be set to 1 in your implementation.
    - You can implement either "cross-correlation" or "convolution", and the result
      will be identical, since we will only test with symmetric filters.
    """

    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1
    if image.ndim == 2:
        output = np.zeros(image.shape[0]*image.shape[1]).reshape(image.shape[0]*image.shape[1])
        prepConvolve(image,filter)
    else:
        output = np.zeros(image.shape[0]*image.shape[1]*image.shape[2]).reshape(image.shape[0],image.shape[1],image.shape[2])
        for i in range(0, image.shape[2]):
            output[:,:,i] = prepConvolve(image[:,:,i],filter)
    
    return output
    
    ############################
    ### TODO: YOUR CODE HERE ###

    raise NotImplementedError(
        "`my_conv2d_numpy` function in `part1.py` needs to be implemented"
    )

    ### END OF STUDENT CODE ####
    ############################

    return filtered_image



def create_hybrid_image(
    image1: np.ndarray, image2: np.ndarray, filter: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Takes two images and a low-pass filter and creates a hybrid image. Returns
    the low frequency content of image1, the high frequency content of image 2,
    and the hybrid image.

    Args:
        image1: array of dim (m, n, c)
        image2: array of dim (m, n, c)
        filter: array of dim (x, y)
    Returns:
        low_frequencies: array of shape (m, n, c)
        high_frequencies: array of shape (m, n, c)
        hybrid_image: array of shape (m, n, c)

    HINTS:
    - You will use your my_conv2d_numpy() function in this function.
    - You can get just the high frequency content of an image by removing its
      low frequency content. Think about how to do this in mathematical terms.
    - Don't forget to make sure the pixel values of the hybrid image are
      between 0 and 1. This is known as 'clipping'.
    - If you want to use images with different dimensions, you should resize
      them in the notebook code.
    """

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]
    assert filter.shape[0] <= image1.shape[0]
    assert filter.shape[1] <= image1.shape[1]
    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1
    #print(filter)
    #filter = create_Gaussian_kernel_2D(7)
    
    updt_img_one = my_conv2d_numpy(image1,filter)
    updt_img_two = my_conv2d_numpy(image2,filter)
    updt_img_two = np.subtract(image2, updt_img_two)
    hybrid_image = np.add(updt_img_one, updt_img_two)
    hybrid_image = np.clip(hybrid_image,0 ,1)
    return updt_img_one, updt_img_two, hybrid_image
