import sys

import cv2
import numpy as np
from skimage.filters import try_all_threshold
from skimage.measure import label, regionprops
from matplotlib import pyplot as plt


def canny_edge_detection_text_fill(image, threshold, grayscale_lenience, is_light_text=True):
    """
    Uses a canny filter and morphological closing to create a tentative text mask from an image

    :param image: For which we want to receive the text mask
    :param threshold: Used to create a luminance threshold, all values below luminance on light text or below
    luminance on dark text will be excluded from the result
    :param grayscale_lenience: Used to create a grayscale mask, all values with difference in their RGB values larger
    than the lenience will be excluded in the result
    :param is_light_text: If text masked is dark or light
    :return: Image with all background removed, and text (if any) left intact
    """
    # Edge mask is created by using a canny filter then performing morphological closing
    edges = cv2.Canny(image, 150, 135)
    edge_mask = _get_edge_mask(edges)

    # Luminance is above or below given threshold
    luminance_mask = np.all(image > threshold, axis=-1) if is_light_text else np.all(image < threshold, axis=-1)

    # Find maximum and minimum channel across all pixels to find maximum difference in color channels and compare to lenience
    maximum_luminance_channel = np.max(image, axis=-1)
    minimum_luminance_channel = np.min(image, axis=-1)
    difference = maximum_luminance_channel - minimum_luminance_channel
    grayscale_mask = np.where(difference <= grayscale_lenience, True, False)

    # Combine all masks to get a final mask
    text_mask = luminance_mask & edge_mask & grayscale_mask

    transformed = image.copy()
    transformed[text_mask] = 255
    transformed[~text_mask] = 0
    return transformed

def plt_image(binarized, cmap='gray'):
    plt.axis('off')
    plt.box(False)
    plt.imshow(binarized, cmap=cmap), plt.show()

def _get_edge_mask(edges, kernel_dim=2, kernel_shape=cv2.MORPH_RECT, iterations=3, max_area=300):
    closing_kernel = cv2.getStructuringElement(kernel_shape, (kernel_dim, kernel_dim))
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, closing_kernel, iterations=iterations)
    labels = label(closing, connectivity=1, background=255)
    regions = [region for region in regionprops(labels) if region.area < max_area]
    for region in regions:
        closing[labels == region.label] = 255
    if not kernel_dim % 2:
        closing = closing[iterations:, iterations:]
        closing = np.pad(closing, ((0, iterations), (0, iterations)), constant_values=0)
    edge_mask = closing == 255
    return edge_mask

def remove_segments(binary_image, min_area=None, max_area=None, max_dimension=None, background=0):
    labeled_image = label(binary_image, connectivity=2, background=background)
    regions = regionprops(labeled_image)
    minimum = min_area if min_area else 0
    maximum = max_area if max_area else sys.maxsize
    max_dim = max_dimension if max_dimension else sys.maxsize
    regions_to_remove = []
    for region in regions:
        if region.area < minimum or region.area > maximum:
            regions_to_remove.append(region)
        elif region.bbox[3] - region.bbox[1] > max_dim or region.bbox[2] - region.bbox[0] > max_dim:
            regions_to_remove.append(region)
    for region in regions_to_remove:
        binary_image[labeled_image == region.label] = background
    return binary_image


def keep_white_pixels(image: np.ndarray, luminance_threshold=120, grayscale_lenience=100, background=0) -> np.ndarray:
    """
    Transforms an image by maintaining grayscale colors

    :param luminance_threshold: Luminance cutoff
    :param image: RGB colorspace image
    :param grayscale_lenience
    :param background determines the background color of the image, all non-grayscale values
    will be converted to background
    :return Image with only grayscale values maintained
    """
    luminance_threshold_mask = np.full(shape=3, fill_value=luminance_threshold)
    luminance_mask = np.all(image >= luminance_threshold_mask, axis=-1)
    maximum_luminance_channel = np.max(image, axis=-1)
    minimum_luminance_channel = np.min(image, axis=-1)
    difference = maximum_luminance_channel - minimum_luminance_channel
    grayscale_mask = np.where(difference <= grayscale_lenience, True, False)
    mask = luminance_mask & grayscale_mask
    processed_image = image.copy()
    processed_image[~mask] = background
    processed_image[mask] = 255
    return processed_image


def try_all_binarizations(img):
    try_all_threshold(img, verbose=False)
    plt.show()
