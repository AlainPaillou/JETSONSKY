"""
Image Conversion Utilities Module

This module provides utility functions for converting between different image formats:
- NumPy <-> CuPy conversions
- RGB image <-> separate RGB channels conversions
- Image processing utilities (blur, negative, quality estimation)

Copyright Alain Paillou 2018-2025
"""

import numpy as np
import cupy as cp
from cupyx.scipy import ndimage
import cv2


def cupy_RGBImage_2_cupy_separateRGB(cupyImageRGB):
    """Convert CuPy RGB image to separate CuPy R, G, B channels."""
    cupy_R = cp.ascontiguousarray(cupyImageRGB[:,:,0], dtype=cp.uint8)
    cupy_G = cp.ascontiguousarray(cupyImageRGB[:,:,1], dtype=cp.uint8)
    cupy_B = cp.ascontiguousarray(cupyImageRGB[:,:,2], dtype=cp.uint8)
    
    return cupy_B, cupy_G, cupy_R


def numpy_RGBImage_2_numpy_separateRGB(numpyImageRGB):
    """Convert NumPy RGB image to separate NumPy R, G, B channels."""
    numpy_R = np.ascontiguousarray(numpyImageRGB[:,:,0], dtype=np.uint8)
    numpy_G = np.ascontiguousarray(numpyImageRGB[:,:,1], dtype=np.uint8)
    numpy_B = np.ascontiguousarray(numpyImageRGB[:,:,2], dtype=np.uint8)
    
    return numpy_R, numpy_G, numpy_B


def numpy_RGBImage_2_cupy_separateRGB(numpyImageRGB):
    """Convert NumPy RGB image to separate CuPy R, G, B channels."""
    cupyImageRGB = cp.asarray(numpyImageRGB)
    cupy_R = cp.ascontiguousarray(cupyImageRGB[:,:,0], dtype=cp.uint8)
    cupy_G = cp.ascontiguousarray(cupyImageRGB[:,:,1], dtype=cp.uint8)
    cupy_B = cp.ascontiguousarray(cupyImageRGB[:,:,2], dtype=cp.uint8)
    
    return cupy_R, cupy_G, cupy_B


def cupy_RGBImage_2_numpy_separateRGB(cupyImageRGB):
    """Convert CuPy RGB image to separate NumPy R, G, B channels."""
    cupy_R = cp.ascontiguousarray(cupyImageRGB[:,:,0], dtype=cp.uint8)
    cupy_G = cp.ascontiguousarray(cupyImageRGB[:,:,1], dtype=cp.uint8)
    cupy_B = cp.ascontiguousarray(cupyImageRGB[:,:,2], dtype=cp.uint8)
    numpy_R = cupy_R.get()
    numpy_G = cupy_G.get()
    numpy_B = cupy_B.get()
    
    return numpy_R, numpy_G, numpy_B


def cupy_separateRGB_2_numpy_RGBimage(cupyR, cupyG, cupyB):
    """Convert separate CuPy R, G, B channels to NumPy RGB image."""
    rgb = (cupyR[..., cp.newaxis], cupyG[..., cp.newaxis], cupyB[..., cp.newaxis])
    cupyRGB = cp.concatenate(rgb, axis=-1, dtype=cp.uint8)
    numpyRGB = cupyRGB.get()
    
    return numpyRGB


def cupy_separateRGB_2_cupy_RGBimage(cupyR, cupyG, cupyB):
    """Convert separate CuPy R, G, B channels to CuPy RGB image."""
    rgb = (cupyR[..., cp.newaxis], cupyG[..., cp.newaxis], cupyB[..., cp.newaxis])
    cupyRGB = cp.concatenate(rgb, axis=-1, dtype=cp.uint8)
    
    return cupyRGB


def numpy_separateRGB_2_numpy_RGBimage(npR, npG, npB):
    """Convert separate NumPy R, G, B channels to NumPy RGB image."""
    rgb = (npR[..., np.newaxis], npG[..., np.newaxis], npB[..., np.newaxis])
    numpyRGB = np.concatenate(rgb, axis=-1, dtype=np.uint8)
    
    return numpyRGB


def gaussianblur_mono(image_mono, niveau_blur):
    """Apply Gaussian blur to monochrome CuPy image."""
    image_gaussian_blur_mono = ndimage.gaussian_filter(image_mono, sigma=niveau_blur)
    
    return image_gaussian_blur_mono


def gaussianblur_colour(im_r, im_g, im_b, niveau_blur):
    """Apply Gaussian blur to separate CuPy R, G, B channels."""
    im_GB_r = ndimage.gaussian_filter(im_r, sigma=niveau_blur)
    im_GB_g = ndimage.gaussian_filter(im_g, sigma=niveau_blur)
    im_GB_b = ndimage.gaussian_filter(im_b, sigma=niveau_blur)
    
    return im_GB_r, im_GB_g, im_GB_b


def image_negative_colour(red, green, blue):
    """Invert color channels (create negative image) for CuPy RGB channels."""
    blue = cp.invert(blue, dtype=cp.uint8)
    green = cp.invert(green, dtype=cp.uint8)
    red = cp.invert(red, dtype=cp.uint8)
    
    return red, green, blue


def Image_Quality(image, IQ_Method):
    """
    Estimate image quality using various methods.
    
    Args:
        image: Input image (NumPy array)
        IQ_Method: Quality estimation method ("Laplacian", "Sobel", or other)
    
    Returns:
        Image quality score (variance of gradient)
    """
    if IQ_Method == "Laplacian":
        image = cv2.GaussianBlur(image, (3, 3), 0)
        # Note: laplacianksize needs to be defined in calling context
        Image_Qual = cv2.Laplacian(image, cv2.CV_64F, ksize=5).var()
    elif IQ_Method == "Sobel":
        image = cv2.GaussianBlur(image, (3, 3), 0)
        # Note: SobelSize needs to be defined in calling context
        Image_Qual = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=5).var()
    else:
        Image_Qual = 0
    
    return Image_Qual
