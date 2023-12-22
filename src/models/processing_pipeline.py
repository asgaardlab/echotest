import hashlib
from typing import Callable, List, Union

import numpy as np


class ProcessingPipeline:
    """
    Represents an image processing pipeline, with callbacks to be applied to single or a stack of images
    """
    def __init__(self):
        self._init_steps = []
        self._steps = []
        self._reduction_step = None

    def __hash__(self):
        return int(hashlib.sha256(self._id.encode()).hexdigest(), 16)

    @property
    def _id(self):
        """
        Creates a unique string id from this pipeline's steps
        """
        return str(self._steps)

    def add_group_step(self, func: Callable, *args):
        """
        Group steps can be called across all images in a pipeline, valid return values for the function are a stack
        of images or nothing
        :param func: To apply across all images
        :param args: To pass to callback func
        """
        self._init_steps.append((func, args))
        return self

    def add_step(self, func: Callable, *args):
        """
        Init steps can be called across all images in a pipeline, valid return values include an image
        :param func: To apply to an image
        :param args: To pass to callback func
        """
        self._steps.append((func, args))
        return self

    def set_reduction_step(self, func: Callable, *args):
        """
        Reduction step can be called across many images to reduce to a single image
        :param func: To apply to an image
        :param args: To pass to callback func
        """
        self._reduction_step = (func, args)
        return self

    def apply_on_image(self, image):
        """
        Apply this image processing pipeline to transform an image.

        Args:
            image (np.ndarray): The input image.

        Returns:
            np.ndarray: The processed image.
        """
        processed_image = image
        for operation, args in self._steps:
            result = operation(processed_image, *args)
            if isinstance(result, tuple):
                for item in result:
                    if isinstance(item, np.ndarray):
                        processed_image = item
                        break
            if isinstance(result, np.ndarray):
                processed_image = result
        return processed_image

    def apply_on_images(self, images: Union[List[np.ndarray], np.ndarray]):
        """
        Apply this group image processing pipeline on passed images

        Args:
            images: The input images
        """
        for operation, args in self._init_steps:
            result = operation(images, *args)
            if result:
                images = result
        return images

    def apply_reduction(self, images: Union[List[np.ndarray], np.ndarray]):
        """
        Applies reduction function on passed images. If no reduction function is defined in this pipeline
        returns the first image in the stack.

        Args:
            images: The input images
        """
        if self._reduction_step:
            reduction_operation, args = self._reduction_step
            result = reduction_operation(images, *args)
        else:
            result = images[0]
        return result
