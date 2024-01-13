import cv2
import numpy as np
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
