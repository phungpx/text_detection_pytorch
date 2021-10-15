import math
from collections import OrderedDict
from typing import Generator, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch import nn

import utils
from abstract.processor import Processor

from ..dto import Image, Point, Word


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
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        imsize: int = 1280, device: str = 'cpu', network_ratio: float = 2,
        region_discard_threshold: float = 0.7,
        region_score_theshold: float = 0.4,
        affinity_score_threshold: float = 0.4,
        cc_area_threshold: float = 10
    ):
        super(WordDetector, self).__init__()
        self.model = model
        self.imsize = imsize
        self.device = device
        self.batch_size = batch_size
        self.network_ratio = network_ratio
        self.cc_area_threshold = cc_area_threshold
        self.region_score_theshold = region_score_theshold
        self.region_discard_threshold = region_discard_threshold
        self.affinity_score_threshold = affinity_score_threshold
        self.std = torch.tensor(std, dtype=torch.float32, device=device).reshape(1, 3, 1, 1)  # B, C, H, W
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device).reshape(1, 3, 1, 1)  # B, C, H, W

        state_dict = self._process_state_dict(torch.load(f=utils.abs_path(weight_path), map_location='cpu'))
        self.model.load_state_dict(state_dict=state_dict)
        self.model.eval().to(device)

    def preprocess(self, images: List[Image]) -> Tuple[List[Image], List[np.ndarray], List[float]]:
        resized_images, resized_ratios = [], []
        for image in images:
            padded_image = self.pad_to_square(image.image)
            resized_image = self.resize(padded_image, imsize=self.imsize, divisor=32)
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            resized_images.append(resized_image)
            resized_ratios.append(resized_image.shape[0] / padded_image.shape[0])

        return images, resized_images, resized_ratios

    def process(
        self,
        images: List[Image],
        resized_images: List[np.ndarray],
        resized_ratios: List[float]
    ) -> Tuple[List[Image], List[np.ndarray], List[np.ndarray], List[float]]:
        region_scores, affinity_scores = [], []
        for batch in chunks(resized_images, size=self.batch_size):
            batch = [torch.from_numpy(image) for image in batch]
            batch = torch.stack(batch, dim=0).to(self.device)
            batch = batch.permute(0, 3, 1, 2).contiguous()
            batch = (batch.float().div(255.) - self.mean) / self.std

            with torch.no_grad():
                pred_scores, _ = self.model(batch)
                region_scores += torch.split(pred_scores[:, :, :, 0], split_size_or_sections=1, dim=0)
                affinity_scores += torch.split(pred_scores[:, :, :, 1], split_size_or_sections=1, dim=0)

        for i, (region_score, affinity_score) in enumerate(zip(region_scores, affinity_scores)):
            region_scores[i] = region_score.squeeze(dim=0).cpu().numpy()
            affinity_scores[i] = affinity_score.squeeze(dim=0).cpu().numpy()

        return images, region_scores, affinity_scores, resized_ratios

    def postprocess(
        self,
        images: List[Image],
        region_scores: List[np.ndarray],
        affinity_scores: List[np.ndarray],
        resized_ratios: List[float]
    ) -> Tuple[List[Image]]:
        for image, region_score, affinity_score, resized_ratio in zip(
            images, region_scores, affinity_scores, resized_ratios
        ):
            if image.image is not None:
                boxes = self.get_boxes(image.image, region_score, affinity_score, resized_ratio)
                image.words = [Word(box=[Point(x=point[0], y=point[1]) for point in box]) for box in boxes]

        return images,

    def get_boxes(
        self,
        image: np.ndarray,
        region_score: np.ndarray,
        affinity_score: np.ndarray,
        resized_ratio: float
    ) -> List[List[List[float]]]:

        boxes = []

        region_binary = cv2.threshold(src=region_score, thresh=self.region_score_theshold, maxval=1, type=0)[-1]
        affinity_binary = cv2.threshold(src=affinity_score, thresh=self.affinity_score_threshold, maxval=1, type=0)[-1]
        combined_mask = np.clip(affinity_binary + region_binary, a_min=0., a_max=1.)

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined_mask.astype(np.uint8), connectivity=4)
        for i in range(1, n_labels):
            cc_area = stats[i, cv2.CC_STAT_AREA]
            if cc_area < self.cc_area_threshold:
                continue

            if np.max(region_score[labels == i]) < self.region_discard_threshold:
                continue

            binary_mask = np.zeros(shape=region_score.shape, dtype=np.uint8)
            binary_mask[labels == i] = 255

            # remove affinity area
            binary_mask[np.logical_and(affinity_binary == 1, region_binary == 0)] = 0

            # dilate connected component area
            x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
            w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]

            niter = int(math.sqrt(cc_area * min(w, h) / (w * h)) * 2)

            x1 = int(round(min(x - niter, 0.)))
            y1 = int(round(min(y - niter, 0.)))
            x2 = int(round(max(x + w + niter + 1, region_score.shape[1])))
            y2 = int(round(max(y + h + niter + 1, region_score.shape[0])))
            kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(niter + 1, niter + 1))

            binary_mask[y1:y2, x1:x2] = cv2.dilate(binary_mask[y1:y2, x1:x2], kernel=kernel)

            # get box
            contours = np.roll(a=np.array(np.where(binary_mask != 0)), shift=1, axis=0).transpose()
            box = cv2.boxPoints(box=cv2.minAreaRect(points=contours))
            box = self.order_points(box)

            # rectify diamond_shape box
            w_box, h_box = self.distance(box[0], box[1]), self.distance(box[1], box[2])
            box_ratio = max(w_box, h_box) / (min(w_box, h_box) + 1e-6)
            if abs(box_ratio - 1) <= 0.1:
                x1, y1 = min(contours[:, 0]), min(contours[:, 1])
                x2, y2 = max(contours[:, 0]), max(contours[:, 1])
                box = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

            boxes.append(self.adjust_box_coordinates(box, self.network_ratio, resized_ratio))

        return boxes

    def order_points(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        assert len(points) == 4, 'Length of points must be 4'
        tl = min(points, key=lambda p: p[0] + p[1])
        br = max(points, key=lambda p: p[0] + p[1])
        tr = max(points, key=lambda p: p[0] - p[1])
        bl = min(points, key=lambda p: p[0] - p[1])
        return [tl, tr, br, bl]

    def distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        point1 = np.float64(point1)
        point2 = np.float64(point2)
        return np.linalg.norm(point1 - point2)

    def adjust_box_coordinates(
        self, box: List[Tuple[float, float]], network_ratio: float, scale_ratio: float
    ) -> List[Tuple[float, float]]:
        box = np.float32(box)
        box = box * network_ratio / scale_ratio
        box = [tuple(point) for point in box]
        return box

    def _process_state_dict(self, state_dict):
        if list(state_dict.keys())[0].startswith('module'):
            processed_state_dict = OrderedDict()
            for name, info in state_dict.items():
                name = '.'.join(name.split('.')[1:])
                processed_state_dict[name] = info
        return processed_state_dict

    def pad_to_square(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        max_size = max(height, width)
        image = np.pad(image, ((0, max_size - height), (0, max_size - width), (0, 0)))
        return image

    def resize(self, image: np.ndarray, imsize: int, divisor: int = 32) -> np.ndarray:
        imsize = int(np.ceil(imsize / divisor)) * divisor
        f = imsize / max(image.shape)
        image = cv2.resize(image, (0, 0), fx=f, fy=f)
        return image
