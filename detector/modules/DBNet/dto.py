from typing import List

import numpy as np

from abstract.dto import DTO


class Point(DTO):
    def __init__(self, x: float, y: float):
        super(Point, self).__init__()
        self.x = x
        self.y = y


class Word(DTO):
    def __init__(self, field: str = None, box: List[Point] = None, confidence: float = None, box_score: float = None):
        super(Word, self).__init__()
        self.box = box
        self.box_score = box_score
        self.field = field
        self.confidence = confidence


class Image(DTO):
    def __init__(self, image: np.ndarray = None, words: List[Word] = None):
        super(Image, self).__init__()
        self.image = image
        self.words = words if words is not None else []
