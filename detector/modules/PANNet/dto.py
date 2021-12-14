from typing import List

import numpy as np

from abstract.dto import DTO


class Point(DTO):
    def __init__(self, x: float, y: float):
        super(Point, self).__init__()
        self.x = x
        self.y = y


class Field(DTO):
    def __init__(self, field: str = None, box: List[Point] = None, confidence: float = None):
        super(Field, self).__init__()
        self.box = box
        self.field = field
        self.confidence = confidence


class Image(DTO):
    def __init__(
        self,
        image: np.ndarray = None,
        text_mask: np.ndarray = None,
        kernel_mask: np.ndarray = None,
        info: List[Field] = None
    ):
        super(Image, self).__init__()
        self.image = image
        self.text_mask = text_mask
        self.kernel_mask = kernel_mask
        self.info = info if info is not None else []
