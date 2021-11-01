from typing import Generator, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch import nn
import pyclipper

import utils
from abstract.processor import Processor

from ..dto import Image, Point, Word
# from .postprocess import decode


def chunks(lst: list, size: Optional[int] = None) -> Union[List, Generator]:
    if size is None:
        yield lst
    else:
        for i in range(0, len(lst), size):
            yield lst[i:i + size]


class WordDetector(Processor):
    def __init__(
        self,
        model: nn.Module,
        weight_path: str = None,
        batch_size: int = None,
        mean: Tuple[float, float, float] = (0., 0., 0.),
        std: Tuple[float, float, float] = (1., 1., 1.),
        imsize: int = 736,
        device: str = 'cpu',
    ):
        super(WordDetector, self).__init__()
        self.model = model
        self.device = device
        self.imsize = imsize
        self.batch_size = batch_size
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
    ) -> Tuple[List[Image], List[np.ndarray], List[np.ndarray], List[float]]:
        for batch in chunks(samples, size=self.batch_size):
            batch = [torch.from_numpy(image) for image in batch]
            batch = torch.stack(batch, dim=0).to(self.device)
            batch = batch.permute(0, 3, 1, 2).contiguous()
            batch = (batch.float().div(255.) - self.mean) / self.std

            with torch.no_grad():
                preds = self.model(batch)
                preds = [pred.squeeze(dim=0) for pred in torch.split(preds, split_size_or_sections=1, dim=0)]

        return images, preds

    def postprocess(self, images: List[Image], preds: List[torch.tensor]) -> Tuple[List[Image]]:
        for image, pred in zip(images, preds):
            if image.image is not None:
                height, width = image.image.shape[:2]
                pred, boxes = self.decode_clip(pred)
                scale = pred.shape[1] / width, pred.shape[0] / height
                image.words = [Word(box=[Point(x=point[0], y=point[1]) for point in box]) for box in boxes / scale]

        return images,

    def decode_clip(self, preds, scale=1, threshold=0.7311, min_area=5):
        preds[:2, :, :] = torch.sigmoid(preds[:2, :, :])
        preds = preds.detach().cpu().numpy()
        text = preds[0] > threshold  # text
        kernel = (preds[1] > threshold) * text  # kernel

        label_num, label = cv2.connectedComponents(kernel.astype(np.uint8), connectivity=4)
        bbox_list = []
        for label_idx in range(1, label_num):
            points = np.array(np.where(label == label_idx)).transpose((1, 0))[:, ::-1]
            if points.shape[0] < min_area:
                continue
            rect = cv2.minAreaRect(points)
            poly = cv2.boxPoints(rect).astype(int)

            d_i = cv2.contourArea(poly) * 1.5 / cv2.arcLength(poly, True)
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            shrinked_poly = np.array(pco.Execute(d_i))
            if shrinked_poly.size == 0:
                continue
            rect = cv2.minAreaRect(shrinked_poly)
            shrinked_poly = cv2.boxPoints(rect).astype(int)
            if cv2.contourArea(shrinked_poly) < 800 / (scale * scale):
                continue

            bbox_list.append([shrinked_poly[1], shrinked_poly[2], shrinked_poly[3], shrinked_poly[0]])
        return label, np.array(bbox_list)
