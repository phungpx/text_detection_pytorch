import math
from collections import OrderedDict
from typing import Generator, List, Optional, Tuple, Union
from shapely.geometry import Polygon
import cv2
import numpy as np
import torch
from torch import nn
import utils
from abstract.processor import Processor
import pyclipper

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
        imsize: int = 1280, device: str = 'cpu',
        thresh: float = 0.3,
        box_thresh: float = 0.7,
        max_candidates: int = 1000,
        unclip_ratio: float = 1.5,
        min_size: int = 3
    ):
        super(WordDetector, self).__init__()
        self.model = model
        self.imsize = imsize
        self.device = device
        self.batch_size = batch_size

        self.std = torch.tensor(std, dtype=torch.float32, device=device).reshape(1, 3, 1, 1)  # B, C, H, W
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device).reshape(1, 3, 1, 1)  # B, C, H, W
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.model = self._load_model(model, torch.load(f=utils.abs_path(weight_path), map_location='cpu'))
        self.model.eval().to(device)
        self.min_size = min_size

    def preprocess(self, images: List[Image]) -> Tuple[List[Image], List[np.ndarray], List[float]]:
        resized_images = []
        for image in images:
            resized_image = self.resize(image.image, imsize=self.imsize, divisor=32)
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            resized_images.append(resized_image)

        return images, resized_images


    def process(
        self,
        images: List[Image],
        resized_images: List[np.ndarray],
    ) -> Tuple[List[Image], List[np.ndarray], List[np.ndarray], List[float]]:
        shrink_map, boxes_batch, scores_batch = [], [], []
        for batch in chunks(resized_images, size=self.batch_size):
            batch = [torch.from_numpy(image) for image in batch]
            batch = torch.stack(batch, dim=0).to(self.device)
            batch = batch.permute(0, 3, 1, 2).contiguous()
            batch = (batch.float().div(255.) - self.mean) / self.std
            with torch.no_grad():
                pred = self.model(batch)   #(shrink_map, threshold_map)
                shrink_map += torch.split(pred[:, 0, :, :], split_size_or_sections=1, dim=0)
        for i, pred in enumerate(shrink_map):
            height, width = images[i].image.shape[0], images[i].image.shape[1]
            seg = self.get_binarize(pred)
            boxes, scores = self.boxes_from_bitmap(pred[0], seg[0], width, height)
            boxes_batch.append(boxes)
            scores_batch.append(scores)
        return images, boxes_batch, scores_batch

    def postprocess(
        self,
        images: List[Image],
        boxes_batch: List[np.ndarray],
        scores_batch: List[np.ndarray],
    ) -> Tuple[List[Image]]:
        for image, boxes_batch, scores_batch in zip(
            images, boxes_batch, scores_batch
        ):  
            if image.image is not None:
                idx = boxes_batch.reshape(boxes_batch.shape[0], -1).sum(axis=1) > 0
                boxes = [self.order_points(list(box)) for box in boxes_batch[idx]]
                image.words = [Word(box=[Point(x=point[0], y=point[1]) for point in box]) for box in boxes]
        return images,

    def get_binarize(self, pred):
        return pred > self.thresh


    def polygons_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (H, W),
            whose values are binarized as {0, 1}
        '''

        assert len(_bitmap.shape) == 2
        bitmap = _bitmap.cpu().numpy()  # The first channel
        pred = pred.cpu().detach().numpy()
        height, width = bitmap.shape
        boxes = []
        scores = []

        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours[:self.max_candidates]:
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            # _, sside = self.get_mini_boxes(contour)
            # if sside < self.min_size:
            #     continue
            score = self.box_score_fast(pred, contour.squeeze(1))
            if self.box_thresh > score:
                continue

            if points.shape[0] > 2:
                box = self.unclip(points, unclip_ratio=self.unclip_ratio)
                if len(box) > 1:
                    continue
            else:
                continue
            box = box.reshape(-1, 2)
            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2:
                continue

            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box)
            scores.append(score)
        return boxes, scores
        


    def order_points(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        assert len(points) == 4, 'Length of points must be 4'
        tl = min(points, key=lambda p: p[0] + p[1])
        br = max(points, key=lambda p: p[0] + p[1])
        tr = max(points, key=lambda p: p[0] - p[1])
        bl = min(points, key=lambda p: p[0] - p[1])
        return [tl, tr, br, bl]


    def _load_model(self, model, weight):

        bb_weight = OrderedDict()
        neck_weight = OrderedDict()
        head_weight = OrderedDict()
        for module_name in weight:
            if 'backbone' in module_name and '_offset' not in module_name:
                bb_weight[module_name[22:]] = weight[module_name]
            elif 'binarize' in module_name or 'thresh' in module_name:
                head_weight[module_name[21:]] = weight[module_name]
            elif 'decoder.in' in module_name or 'decoder.out' in module_name:
                neck_weight[module_name[21:]] = weight[module_name]

        bb_weight.pop("fc.weight")
        bb_weight.pop("fc.bias")
        bb_weight.pop("smooth.weight")
        bb_weight.pop("smooth.bias")
        
        model.backbone.load_state_dict(bb_weight)
        model.head.load_state_dict(head_weight)
        model.neck.load_state_dict(neck_weight)
        return model

    def resize(self, image: np.ndarray, imsize: int, divisor: int = 32) -> np.ndarray:
        height, width = image.shape[0], image.shape[1]
        if height < width:
            new_height = imsize
            new_width = new_height / height * width
        else:
            new_width = imsize
            new_height = new_width / width * height
        new_height = int(round(new_height / divisor) * divisor)
        new_width = int(round(new_width / divisor) * divisor)
        image = cv2.resize(image, (new_width, new_height))
        return image

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (H, W),
            whose values are binarized as {0, 1}
        '''

        assert len(_bitmap.shape) == 2
        bitmap = _bitmap.cpu().numpy()  # The first channel
        pred = pred.cpu().detach().numpy()
        height, width = bitmap.shape
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = min(len(contours), self.max_candidates)
        boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
        scores = np.zeros((num_contours,), dtype=np.float32)

        for index in range(num_contours):
            contour = contours[index].squeeze(1)
            points, sside = self.get_mini_boxes(contour)
            if sside < 3:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, contour)
            if self.box_thresh > score:
                continue

            box = self.unclip(points, unclip_ratio=self.unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes[index, :, :] = box.astype(np.int16)
            scores[index] = score
        return boxes, scores

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def unclip(self, box, unclip_ratio=1.5):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded