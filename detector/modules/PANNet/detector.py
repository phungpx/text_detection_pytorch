from typing import Generator, List, Optional, Tuple, Union

import cv2
import torch
import pyclipper
import numpy as np
from torch import nn
from shapely.geometry import Polygon

import utils
from .dto import Image, Point, Word
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
        weight_path: str = None,
        batch_size: int = None,
        binary_threshold: float = 0.7311,
        area_threshold: float = 0.,
        mean: Tuple[float, float, float] = (0., 0., 0.),
        std: Tuple[float, float, float] = (1., 1., 1.),
        imsize: int = 736,
        shrink_ratio: float = 0.5,
        device: str = 'cpu',
    ):
        super(Detector, self).__init__()
        self.model = model
        self.device = device
        self.imsize = imsize
        self.batch_size = batch_size
        self.shrink_ratio = shrink_ratio
        self.area_threshold = area_threshold
        self.binary_threshold = binary_threshold
        self.std = torch.tensor(std, dtype=torch.float32, device=device).reshape(1, 3, 1, 1)  # B, C, H, W
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device).reshape(1, 3, 1, 1)  # B, C, H, W

        state_dict = torch.load(f=utils.abs_path(weight_path), map_location='cpu')
        # state_dict = torch.load(f=utils.abs_path(weight_path), map_location='cpu')['state_dict']
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
                fx, fy = width / pred.shape[2], height / pred.shape[1]
                boxes = self.get_boxes(pred, fx=fx, fy=fy)
                image.words = [
                    Word(box=[Point(x=point[0], y=point[1]) for point in box]) for box in boxes
                ]

        return images,

    def get_boxes(
        self, pred: np.ndarray, fx: float = 1, fy: float = 1, min_area: int = 5
    ) -> List[List[Tuple[float, float]]]:
        text_region = pred[0] > self.binary_threshold  # text region
        kernel = (pred[1] > self.binary_threshold) * text_region  # kernel of text region

        num_labels, label = cv2.connectedComponents(kernel.astype(np.uint8), connectivity=4)

        boxes = []
        for i in range(1, num_labels):
            # find bounding box
            contours = cv2.findContours(np.uint8(label == i), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            if len(contours) != 1:
                raise RuntimeError('Found more than one contour in one connected component.')

            box = cv2.boxPoints(cv2.minAreaRect(contours[0])).tolist()
            if not Polygon(box).is_valid:
                # raise ValueError('must be valid polygon.')
                continue

            box = self.unshrink_polygon(box, r=self.shrink_ratio) 
            if not len(box):
                continue

            box = cv2.boxPoints(cv2.minAreaRect(np.array(box)))
            box = self.order_points(box)
            box = [(int(round(x * fx)), int(round(y * fy))) for x, y in box]

            boxes.append(box)

        return boxes

    def unshrink_polygon(self, points, r: float = 0.5):
        offseter = pyclipper.PyclipperOffset()

        poly = Polygon(points)
        d = poly.area * (1 + r ** 2) / poly.length

        points = [tuple(point) for point in points]
        offseter.AddPath(points, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        polys = offseter.Execute(d)

        return polys

    def order_points(self, points: List[List[float]]) -> List[List[float]]:
        tl = min(points, key=lambda p: p[0] + p[1])
        br = max(points, key=lambda p: p[0] + p[1])
        tr = max(points, key=lambda p: p[0] - p[1])
        bl = min(points, key=lambda p: p[0] - p[1])

        return [tl, tr, br, bl]

    # def rm_small_components(self, mask: np.ndarray, class_ratio: float) -> np.ndarray:
    #     num_class, label = cv2.connectedComponents(mask.round().astype(np.uint8))
    #     threshold = self.area_threshold * class_ratio * mask.shape[0] * mask.shape[1]

    #     for i in range(1, num_class):
    #         area = (label == i).sum()
    #         if area < threshold:
    #             mask[label == i] = 0

    #     return mask
