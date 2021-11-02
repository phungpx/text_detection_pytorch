from typing import Generator, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch import nn
import pyclipper

import utils
from abstract.processor import Processor

from .dto import Image, Point, Word
# from .postprocess import decode


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
        weight_path: str = None,
        batch_size: int = None,
        binary_threshold: float = 0.7311,
        mean: Tuple[float, float, float] = (0., 0., 0.),
        std: Tuple[float, float, float] = (1., 1., 1.),
        imsize: int = 736,
        device: str = 'cpu',
    ):
        super(Detector, self).__init__()
        self.model = model
        self.device = device
        self.imsize = imsize
        self.batch_size = batch_size
        self.binary_threshold = binary_threshold
        self.std = torch.tensor(std, dtype=torch.float32, device=device).reshape(1, 3, 1, 1)  # B, C, H, W
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device).reshape(1, 3, 1, 1)  # B, C, H, W

        state_dict = torch.load(f=utils.abs_path(weight_path), map_location='cpu')['state_dict']
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

    def process(
        self, images: List[Image], samples: List[np.ndarray]
    ) -> Tuple[List[Image], List[np.ndarray]]:
        preds = []
        for batch in chunks(samples, size=self.batch_size):
            batch = [torch.from_numpy(image) for image in batch]
            batch = torch.stack(batch, dim=0).to(self.device)
            batch = batch.permute(0, 3, 1, 2).contiguous()
            batch = (batch.float().div(255.) - self.mean) / self.std

            with torch.no_grad():
                batch = self.model(batch)  # N x 6 x H x W
                batch[:2, :, :] = nn.Sigmoid()(batch[:2, :, :])
                preds += [
                    pred.squeeze(dim=0).cpu().numpy()  # 6 x H x W
                    for pred in torch.split(batch, split_size_or_sections=1, dim=0)
                ]

        return images, preds

    def postprocess(self, images: List[Image], preds: List[torch.tensor]) -> Tuple[List[Image]]:
        for image, pred in zip(images, preds):
            if image.image is not None:
                height, width = image.image.shape[:2]
                fx, fy = pred.shape[2] / width, pred.shape[1] / height
                boxes = self.get_boxes(pred, fx=fx, fy=fy)
                image.words = [
                    Word(box=[Point(x=point[0], y=point[1]) for point in box]) for box in boxes
                ]

        return images,

    def get_boxes(
        self, pred: np.ndarray, fx: float = 1, fy: float = 1, min_area: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        text_region = pred[0] > self.binary_threshold
        kernel = (pred[1] > self.binary_threshold) * text_region  # kernel

        num_labels, label = cv2.connectedComponents(kernel.astype(np.uint8), connectivity=4)

        bboxes = []
        for label_id in range(1, num_labels):
            points = np.array(np.where(label == label_id)).transpose((1, 0))[:, ::-1]
            if points.shape[0] < min_area:
                continue

            rect = cv2.minAreaRect(points)
            poly = cv2.boxPoints(rect).astype(np.int)

            d_i = cv2.contourArea(poly) * 1.5 / cv2.arcLength(poly, True)
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            shrinked_poly = np.array(pco.Execute(d_i))
            if shrinked_poly.size == 0:
                continue

            rect = cv2.minAreaRect(shrinked_poly)
            shrinked_poly = cv2.boxPoints(rect).astype(np.int)
            # if cv2.contourArea(shrinked_poly) < 800 / (fx * fy):
            #     continue

            bboxes.append(
                [
                    shrinked_poly[1] / fx,
                    shrinked_poly[2] / fy,
                    shrinked_poly[3] / fx,
                    shrinked_poly[0] / fy,
                ]
            )

        return bboxes
