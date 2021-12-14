from typing import Generator, List, Optional, Tuple, Union

import cv2
import torch
import pyclipper
import numpy as np
from torch import nn
from shapely.geometry import Polygon

import utils
from .dto import Image, Point, Field
from abstract.processor import Processor


def chunks(lst: list, size: Optional[int] = None) -> Union[List, Generator]:
    if size is None:
        yield lst
    else:
        for i in range(0, len(lst), size):
            yield lst[i:i + size]


class Detector(Processor):
    def __init__(
        self,
        model: nn.Module,
        imsize: int = 640,
        batch_size: int = None,
        weight_path: str = None,
        mean: Tuple[float, float, float] = (0., 0., 0.),
        std: Tuple[float, float, float] = (1., 1., 1.),
        device: str = 'cpu',
    ):
        super(Detector, self).__init__()
        self.model = model
        self.device = device
        self.imsize = imsize
        self.batch_size = batch_size
        self.std = torch.tensor(std, dtype=torch.float32, device=device).reshape(1, 3, 1, 1)  # B, C, H, W
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device).reshape(1, 3, 1, 1)  # B, C, H, W

        state_dict = torch.load(f=utils.abs_path(weight_path), map_location='cpu')
        self.model.load_state_dict(state_dict=state_dict)
        self.model.eval().to(device)

    def preprocess(self, images: List[Image]) -> Tuple[List[Image], List[np.ndarray], List[float]]:
        samples = []
        for image in images:
            fs = self.imsize / min(image.image.shape[:2])
            sample = cv2.resize(image.image, None, fx=fs, fy=fs)
            sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
            samples.append(sample)

        return images, samples

    def process(self, images: List[Image], samples: List[np.ndarray]) -> Tuple[List[Image], List[dict]]:
        images_boxes = []

        pairs = list(zip(images, samples))

        for batch in chunks(pairs, size=self.batch_size):
            image_sizes = [pair[0].image.shape[1::-1] for pair in batch]

            samples = [pair[1] for pair in batch]
            samples = [torch.from_numpy(sample) for sample in samples]
            samples = torch.stack(samples, dim=0).to(self.device)
            samples = samples.permute(0, 3, 1, 2).contiguous()
            samples = (samples.float().div(255.) - self.mean) / self.std

            image_boxes = self.model.predict(x=samples, image_sizes=image_sizes)  # N x 6 x H x W
            images_boxes += image_boxes

        return images, images_boxes

    def postprocess(self, images: List[Image], images_boxes: List[dict]) -> Tuple[List[Image]]:
        for image, image_boxes in zip(images, images_boxes):
            if image.image is not None:
                image.info = [
                    Field(box=[Point(x=point[0], y=point[1]) for point in box['points']])
                    for box in image_boxes['text_boxes']
                ]
                image.text_mask = image_boxes['text_mask']
                image.kernel_mask = image_boxes['kernel_mask']

        return images,
