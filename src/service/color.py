import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def color_split(image, is_light_text=True):
    """
    Determine the foreground and background colors, using k-means clustering
    :return: (foreground color, background color) RGB
    """
    df = pd.DataFrame(image.reshape(-1, image.shape[-1]))  # Reshape to match clustering requirements

    # K-means with two clusters will predict a foreground and background
    clustering_algorithm = KMeans(n_clusters=2, n_init='auto')
    clustering_algorithm.fit(df)
    labels = clustering_algorithm.predict(df)
    mask = labels.reshape(image.shape[:-1]).astype(bool)  # Create a mask from two labels

    # Segment A and B can be either background or foreground
    segment_a = image.copy().astype(float)
    segment_a[~mask] = np.nan
    segment_b = image.copy().astype(float)
    segment_b[mask] = np.nan
    # If segment A is brighter on average than segment B and we are using light text we assume A is the foreground
    segment_a_foreground = np.nanmean(segment_a) > np.nanmean(segment_b) if is_light_text else np.nanmean(
        segment_a) < np.nanmean(segment_b)

    # Set foreground based on luminance average
    foreground = segment_a if segment_a_foreground else segment_b
    background = segment_b if segment_a_foreground else segment_a

    # For foreground - we assume this is text so use median, and mean for background
    reference_foreground_color = np.nanmedian(foreground, axis=(0, 1))
    reference_background_color = np.nanmean(background, axis=(0, 1))
    return reference_foreground_color, reference_background_color


def get_luminance_rgb(rgb_color):
    return 0.2126 * rgb_color[0] / 255 + 0.7152 * rgb_color[1] / 255 + 0.0722 * rgb_color[2] / 255
