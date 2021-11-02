from typing import List, Tuple

import numpy as np

from abstract.stage import Stage

from .dto import Image


class Detection(Stage):
    def __init__(self, *args, **kwargs):
        super(Detection, self).__init__(*args, **kwargs)

    def preprocess(self, images: List[np.ndarray]) -> Tuple[List[Image]]:
        for image in images:
            if not isinstance(image, np.ndarray):
                raise TypeError(f'image must be an instance of ndarray, not {type(image)}.')
            if len(image.shape) != 3:
                raise ValueError(f'image must be a 3d ndarray, not {len(image.shape)}d.')
            if image.shape[2] != 3:
                raise ValueError(f'image must be 3-channel, not {image.shape[2]}-channel.')

        images = [Image(image=image) for image in images]

        return images,
