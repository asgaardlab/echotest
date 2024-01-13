import math
import re
import statistics

import cv2
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pytesseract
import skimage.metrics

from src.models.processing_pipeline import ProcessingPipeline
from src.service.color import get_luminance_rgb, color_split
from src.service.image_transformer import canny_edge_detection_text_fill
from src.service.tokenization import perform_segmentation

plt.rcParams['figure.figsize'] = (19.2, 10.8)


class OCRExtractOptions:
    def __init__(self, confidence_level=80., cleaned=False, text_only=False):
        self.confidence_level = confidence_level
        self.cleaned = cleaned
        self.text_only = text_only


def tsv_to_dict(tsv_string):
    """
    Creates a dictionary from a TSV string
    """
    rows = tsv_string.strip().split('\n')
    headers = rows[0].split('\t')
    data = [row.split('\t') for row in rows[1:]]
    result = []
    for row in data:
        row_dict = {header: value for header, value in zip(headers, row)}
        result.append(row_dict)
    return result


def crop_image(image: np.ndarray, crop):
    """
    Crops an image according to the crop bounding box
    :param image: to crop
    :param crop: (leftmost x, topmost y, rightmost x, bottommost y)
    """
    if not crop:
        return image
    if crop[1] > crop[3] or crop[0] > crop[2]:
        return image
    if any(param < 0 for param in crop):
        return image
    return image[crop[1]:crop[3], crop[0]:crop[2]]


class OpticalCharacterRecognizer:
    """
    Model used to extract text features from images
    """

    def __init__(self):
        self.words = set(nltk.corpus.wordnet.words()) | set(nltk.corpus.words.words())
        self.threshold = None
        self.crop = None
        self.common_mistake_replacements = [('|', 'I'), ('[', 'I')]
        self.DEFAULT_LUMINANCE_THRESHOLD_LIGHT = 180
        self.DEFAULT_LUMINANCE_THRESHOLD_DARK = 60
        self.DEFAULT_GRAYSCALE_THRESHOLD = 50

    def crop_image(self, image):
        """
        Crops an image using the OCR crop settings
        """
        return crop_image(image, self.crop)

    def default_luminance_threshold(self, is_light_text):
        """
        Returns a default luminance threshold given light or dark text
        """
        return self.DEFAULT_LUMINANCE_THRESHOLD_LIGHT if is_light_text else self.DEFAULT_LUMINANCE_THRESHOLD_DARK

    def canny_edge_detection_text_fill(self, image, is_light_text=True):
        """
        Runs canny edge detection text fill using the OCR thresholds
        """
        return canny_edge_detection_text_fill(image, self.threshold[0], self.threshold[1], is_light_text)

    def extract_text(self, image: np.ndarray) -> str:
        """
        Extracts text from an image

        :param image: The image object.
        """
        high_conf_data = self.extract_ocr_data(image, OCRExtractOptions(text_only=True, confidence_level=50.))
        clean_text = self.replace_common_mistakes(' '.join(high_conf_data))
        return clean_text

    def subtitle_color_contrast(self, image: np.ndarray):
        thresholded_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresholded_image = self.optimal_binary_threshold(thresholded_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.optimal_crop(thresholded_image, num_crops=10)
        wcag_ratios = []
        if self.crop:
            for crop in self.crop:
                crop = crop_image(image, crop)
                foreground_color, background_color = color_split(crop)
                luminance_a, luminance_b = get_luminance_rgb(foreground_color), get_luminance_rgb(background_color)
                L1 = max(luminance_a, luminance_b)
                L2 = min(luminance_a, luminance_b)
                wcag_contrast_ratio = (L1 + 0.05) / (L2 + 0.05)
                wcag_ratios.append(wcag_contrast_ratio)
            return statistics.mean(wcag_ratios) if len(wcag_ratios) else None
        else:
            return None

    def set_thresholds_from(self, images, is_light_text=True):
        """
        Uses passed images to set the luminance threshold, saturation threshold and crops
        """
        foreground_luminances, differences, crops = [], [], []
        images_considered = 0
        NUM_CROPS = 10  # Number of crops to take on an image
        CROP_MINIMUM = 3  # Number of crops required to consider this image in metrics
        IMAGE_CUTOFF = 0.6  # Minimum percentage of images to consider
        CROP_CUTOFF = 10  # Minimum number of crops to set thresholds from

        for image in images:
            # Threshold image for better optical capture recognition accuracy
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            thresholded = self.optimal_binary_threshold(gray)

            # Find crops around text
            self.optimal_crop(thresholded, num_crops=NUM_CROPS)
            if self.crop and len(self.crop) >= CROP_MINIMUM:  # We have detected crops around a minimum number of words
                images_considered += 1
                for crop in self.crop:
                    cropped = crop_image(image, crop)
                    crops.append(crop)

                    # Use k-means to find the primary colors for the background and foreground
                    foreground_color, background_color = color_split(cropped, is_light_text)

                    min_luminance = np.min(foreground_color)
                    max_luminance = np.max(foreground_color)
                    foreground_luminances.append(min_luminance if is_light_text else max_luminance)
                    differences.append(max_luminance - min_luminance)

                if (images_considered / len(images)) > IMAGE_CUTOFF or len(foreground_luminances) > CROP_CUTOFF:
                    break

        if images_considered:
            LUMINANCE_DELTA = -15 if is_light_text else 15
            SATURATION_DELTA = 25
            # Using the extracted luminance and differences set the luminance and saturation thresholds
            self.threshold = (statistics.mean(foreground_luminances) + LUMINANCE_DELTA,
                              statistics.mean(differences) + SATURATION_DELTA)

            # Set the global crop from the extracted crops
            crop_min_x = min(crops, key=lambda crop: crop[0])[0]  # Left most crop left value
            crop_min_y = min(crops, key=lambda crop: crop[1])[1]  # Top most crop top value
            crop_max_x = max(crops, key=lambda crop: crop[2])[2]  # Right most crop right value
            crop_max_y = max(crops, key=lambda crop: crop[3])[3]  # Bottom most crop bottom value

            # Add a margin to the crop based on distance between left and right most crop, and top and bottommost crop
            X_MARGIN_PERCENTAGE = 0.65
            Y_MARGIN_PERCENTAGE = 0.2
            x_margin = int((crop_max_x - crop_min_x) * X_MARGIN_PERCENTAGE)
            y_margin = int((crop_max_y - crop_min_y) * Y_MARGIN_PERCENTAGE)

            self.crop = (max(crop_min_x - x_margin, 0),
                         crop_min_y - y_margin,
                         min(crop_max_x + x_margin, images[0].shape[1]),
                         crop_max_y + y_margin)
        else:
            # If we haven't considered any images, just set defaults
            self.threshold = (self.default_luminance_threshold(is_light_text), self.DEFAULT_GRAYSCALE_THRESHOLD)

    def integrate_frames(self, images, processing_pipeline: ProcessingPipeline, rmse_cutoff=0.7):
        """
        Joins similar frames applying pipeline reduction to join, while discarding "empty" frames, uses RMSE
        with a cutoff to discriminate between similar images. Pipeline transformations are applied to images prior
        to comparison to simplify comparison.

        :param images: To integrate
        :param processing_pipeline: Single image transformation pipeline applied on each image prior to checking
        image similarity, reduction callback used to integrate frames
        :param rmse_cutoff: Root mean square error cutoff in similarity analysis, two images with higher RMSE will
        be considered distinct
        :return:
        """
        # First we apply pipeline transformation on images prior to comparison
        transformed = []
        for image in images:
            processed_image = processing_pipeline.apply_on_image(image)
            transformed.append(processed_image)

        h, w, d = processed_image.shape
        left = 0  # Index of leftmost image in considered integration stack
        right = 0  # Index of rightmost image in considered integration stack
        integrated_frames = []
        while right < len(images) - 1:
            candidate_frame = transformed[right + 1]
            comparison_frame = transformed[right]
            # Special case: comparison frame is considered blank, we can just ignore it
            is_comparison_frame_blank = (np.count_nonzero(comparison_frame) / (h * w * d)) < 0.001
            is_candidate_frame_blank = (np.count_nonzero(candidate_frame) / (h * w * d)) < 0.001
            if is_comparison_frame_blank:
                left = right + 1  # We can ignore this frame, and move on to the next

            # Otherwise we would like to integrate all previous frames when either one frame is black and the other is
            # not, or when the RMSE > cutoff
            elif is_comparison_frame_blank != is_candidate_frame_blank or skimage.metrics.normalized_root_mse(
                    comparison_frame, candidate_frame) > rmse_cutoff:
                integrated_frames.append(processing_pipeline.apply_reduction(np.stack(images[left:right + 1])))
                left = right + 1
            right += 1

        # Integrate remaining frames
        integrated_frames.append(processing_pipeline.apply_reduction(np.stack(images[left:right + 1])))
        return integrated_frames

    def mean_fill_reduce(self, images, is_light_text=True):
        mean = self.crop_image(np.mean(images, axis=0))
        mean = np.clip(mean, 0, 255).astype(images.dtype)
        filled = self.canny_edge_detection_text_fill(mean, is_light_text)
        return filled

    def rmse(self, image_a: np.ndarray, image_b: np.ndarray):
        h, w, d = image_a.shape
        diff = cv2.subtract(image_a.astype(int), image_b.astype(int))
        err = np.sum(diff ** 2)
        mse = err / (h * w * d)
        rmse = math.sqrt(mse)
        return rmse

    def optimal_binary_threshold(self, image):
        threshold_results = [(cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY), threshold)
                             for threshold in range(80, 280, 40)]
        highest_mean_confidence = self._ocr_confidence(image)
        highest_confidence_image = image
        for image_result, current_threshold in threshold_results:
            test = image_result[1]
            mean_confidence = self._ocr_confidence(test)
            if mean_confidence >= highest_mean_confidence:
                highest_mean_confidence = mean_confidence
                highest_confidence_image = test
                self.threshold = (
                    current_threshold, self.threshold[1] if self.threshold else self.DEFAULT_GRAYSCALE_THRESHOLD)
        return highest_confidence_image

    def _ocr_confidence(self, image):
        extracted_data = self.extract_ocr_data(image, OCRExtractOptions(confidence_level=10., cleaned=True))
        extracted_data = [data for data in extracted_data if "text" in data and data["text"].strip()]
        confidences = [float(data["conf"]) for data in extracted_data if "text" in data and data["text"].strip()]
        confidence_score = statistics.mean(confidences) if confidences else 0
        return confidence_score

    def optimal_crop(self, image, confidence_level=90., margin=0.1, num_crops=1):
        targeted_ocr_data = self.extract_ocr_data(image, OCRExtractOptions(confidence_level, cleaned=True))
        if num_crops != 1 and targeted_ocr_data:
            sorted_data = sorted(targeted_ocr_data, key=lambda x: float(x["conf"]), reverse=True)
            highest_conf = sorted_data[:num_crops]
            self.crop = self._multi_crop_from_data(highest_conf)
            crops = []
            for crop in self.crop:
                crops.append(crop_image(image, crop))
            return crops
        if targeted_ocr_data:
            self.crop = self._crop_from_data(targeted_ocr_data, margin)
            return crop_image(image, self.crop)
        else:
            self.crop = None
            return image

    def _multi_crop_from_data(self, data):
        crops = []
        for data_point in data:
            highest_y = int(data_point.get("top", 0))
            leftmost_x = int(data_point.get("left", 0))
            width = int(data_point.get("width", 0))
            height = int(data_point.get("height", 0))
            crop = (leftmost_x,
                    highest_y,
                    leftmost_x + width,
                    highest_y + height)
            crops.append(crop)
        return crops

    def _crop_from_data(self, data, margin=0.01):
        highest_y = int(min(data, key=lambda x: int(x["top"])).get("top", 0))
        leftmost_x = int(min(data, key=lambda x: int(x["left"])).get("left", 0))
        lowest = max(data, key=lambda x: int(x["top"]))
        lowest_y = int(lowest.get("top")) + int(lowest.get("height", 0))
        rightmost = max(data, key=lambda x: int(x["left"]))
        rightmost_x = int(rightmost.get("left")) + int(rightmost.get("width", 0))
        width_margin = int((rightmost_x - leftmost_x) * margin)
        height_margin = int((lowest_y - highest_y) * margin)
        crop = (leftmost_x - width_margin,
                highest_y - height_margin,
                rightmost_x + width_margin,
                lowest_y + height_margin)
        return crop

    def extract_ocr_data(self, image, options: OCRExtractOptions):
        """
        Extracts words from the image using OCR and returns a set of words.
        :param options: extraction options
        :param image: The input image.
        :return: Set of extracted words.
        """
        data = tsv_to_dict(pytesseract.image_to_data(image))
        if options.confidence_level:
            data = [data for data in data if float(data["conf"]) > options.confidence_level]
        if options.cleaned:
            data = [data for data in data if "text" in data and self.is_valid_word(data["text"])]
        if options.text_only:
            data = [data["text"] for data in data if "text" in data and data["text"].strip()]
        return data

    def is_valid_word(self, text):
        """
        Rules based check to determine if a word and confidence pair is valid for the OCR model
        """
        punctuation_removed_text = re.sub(r"\W+", "", text)
        case_folded_punctuation_removed_text = punctuation_removed_text.lower()
        segmented = perform_segmentation(case_folded_punctuation_removed_text).split()
        if not text or not punctuation_removed_text:
            return False
        number = punctuation_removed_text.isnumeric()
        tokenized_with_replacements = self.replace_common_mistakes(case_folded_punctuation_removed_text)
        in_dictionary = case_folded_punctuation_removed_text in self.words or tokenized_with_replacements in self.words or all(
            word in self.words for word in segmented)
        potential_proper_noun = not in_dictionary and punctuation_removed_text.isalpha() and punctuation_removed_text.istitle()
        return in_dictionary or potential_proper_noun or number

    def replace_common_mistakes(self, text):
        """
        Returns a modified string with all common mistake replacements completed
        """
        replaced_text = text
        for mistake, replacement in self.common_mistake_replacements:
            replaced_text = replaced_text.replace(mistake, replacement)
        return replaced_text
