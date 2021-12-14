import sys
import cv2
import json
import torch
import random
import pyclipper
import numpy as np
import imgaug.augmenters as iaa

from pathlib import Path
from natsort import natsorted
from torch.utils.data import Dataset
from shapely.geometry import Polygon
from typing import List, Tuple, Optional

from .augmenter import Augmenter


class ICDAR2015(Dataset):
    def __init__(
        self,
        dirnames: List[str],
        imsize: int = 640,
        mean: Tuple[float, float, float] = (0., 0., 0.),
        std: Tuple[float, float, float] = (1., 1., 1.),
        shrink_ratio: float = 0.5,
        max_shrink: int = sys.maxsize,
        image_extents: List[str] = ['.jpg'],
        label_extent: str = '.json',
        transforms: Optional[List] = None,
        require_transforms: Optional[List] = None,
        ignore_blur_text: bool = True,
    ) -> None:
        super(ICDAR2015, self).__init__()
        self.imsize = imsize
        self.max_shrink = max_shrink
        self.shrink_ratio = shrink_ratio
        self.ignore_blur_text = ignore_blur_text
        self.transforms = transforms if transforms else []
        self.require_transforms = require_transforms if require_transforms else []

        self.mean = torch.tensor(mean, dtype=torch.float).view(3, 1, 1)  # 3 x 1 x 1
        self.std = torch.tensor(std, dtype=torch.float).view(3, 1, 1)  # 3 x 1 x 1

        self.augmenter = Augmenter()

        image_paths, label_paths = [], []
        for dirname in dirnames:
            for image_extent in image_extents:
                image_paths.extend(list(Path(dirname).glob('**/*{}'.format(image_extent))))

            label_paths.extend(list(Path(dirname).glob('**/*{}'.format(label_extent))))

        image_paths = natsorted(image_paths, key=lambda x: x.stem)
        label_paths = natsorted(label_paths, key=lambda x: x.stem)

        self.data_pairs = [(image, label)for image, label in zip(image_paths, label_paths) if image.stem == label.stem]

        print(f"{', '.join([Path(dirname).stem for dirname in dirnames])} - {len(self.data_pairs)}")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        '''Returns:
            image (Float32Tensor): 3 x H x W, normalized sample (image / 255 - mean) / std
            mask (Float32Tensor): 2 x H x W, combination of kernel map and text map
            effective_mask (Uint8Tensor): H x W, mask with ignored text region with blur text
            image_info (Dictionary): image_path, image_size (w, h), all info of text boxes in label.
        '''
        image_path, label_path = self.data_pairs[idx]

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with label_path.open(mode='r', encoding='utf8') as f:
            label = json.load(f)

        image_info = {
            'image_path': str(image_path),
            'image_size': image.shape[1::-1],
            'boxes': self.get_boxes(label),
        }

        for transform in random.sample(self.transforms, k=random.randint(0, len(self.transforms))):
            image, label = self.augmenter.apply(image=image, label=label, augmenter=transform)

        for require_transform in self.require_transforms:
            image, label = self.augmenter.apply(image=image, label=label, augmenter=require_transform)

        image, label = self.pad_to_square(image=image, label=label, imsize=self.imsize)
        text_map, kernel_map, effective_map = self.generate_map(image=image, label=label)

        image = torch.from_numpy(np.ascontiguousarray(image))  # H x W x 3, tensor.uint8
        text_map = torch.from_numpy(np.ascontiguousarray(text_map))  # H x W, tensor.uint8
        kernel_map = torch.from_numpy(np.ascontiguousarray(kernel_map))  # H x W, tensor.uint8
        effective_map = torch.from_numpy(np.ascontiguousarray(effective_map))  # H x W, tensor.uint8, values 0 or 1

        image = image.permute(2, 0, 1).contiguous().float()  # 3 x H x W, tensor.float32
        mask = torch.stack([text_map, kernel_map], dim=0).float()  # 2 x H x W, tensor.float32
        image = (image.div(255.) - self.mean) / self.std

        return image, mask, effective_map, image_info

    # def resize(self, image: np.ndarray, label: dict, imsize: int = 640) -> Tuple[np.ndarray, dict]:
    #     f = imsize / min(image.shape[:2])

    #     image, label = self.augmenter.apply(
    #         image=image, label=label, augmenter=iaa.Resize(size=f)
    #     )

    #     image, label = self.augmenter.apply(
    #         image=image, label=label,
    #         augmenter=iaa.CropToFixedSize(width=imsize, height=imsize, position='uniform')
    #     )

    #     return image, label

    def pad_to_square(self, image: np.ndarray, label: dict, imsize: int = 640) -> Tuple[np.ndarray, dict]:
        f = imsize / max(image.shape[:2])

        image, label = self.augmenter.apply(
            image=image, label=label, augmenter=iaa.Resize(size=f)
        )

        image, label = self.augmenter.apply(
            image=image, label=label,
            augmenter=iaa.PadToSquare(position='right-bottom')
        )

        return image, label

    def get_boxes(self, label: dict) -> List[dict]:
        boxes = []
        height, width = label['imageHeight'], label['imageWidth']
        for shape in label['shapes']:
            if shape['shape_type'] == 'rectangle':
                points = self.to_4points(shape['points'])
            elif shape['shape_type'] == 'polygon':
                points = shape['points']
            else:
                continue

            points = self.to_valid_poly(points, image_height=height, image_width=width)
            if not Polygon(points).is_valid:
                continue

            # get text boxes after transformers for evaluation
            box = {
                'points': points,
                'text': shape.get('value', '###'),
                'ignore': False,
            }

            if self.ignore_blur_text and (box['text'] == '###'):
                box['ignore'] = True

            boxes.append(box)

        return boxes

    def generate_map(self, image: np.ndarray, label: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        height, width = image.shape[:2]

        text_map = np.zeros(shape=(height, width), dtype=np.uint8)
        kernel_map = np.zeros(shape=(height, width), dtype=np.uint8)

        text_id = 0
        ignored_polys = []
        for shape in label['shapes']:
            if shape['shape_type'] == 'rectangle':
                points = self.to_4points(shape['points'])
            elif shape['shape_type'] == 'polygon':
                points = shape['points']
            else:
                continue

            points = self.to_valid_poly(points, image_height=height, image_width=width)
            if not Polygon(points).is_valid:
                continue

            # text region map
            cv2.fillPoly(img=text_map, pts=[np.int32(points)], color=text_id + 1)

            # shrunk kernel map
            shrunk_polygons = self.shrink_polygon(points, r=self.shrink_ratio, max_shrink=self.max_shrink)
            cv2.fillPoly(img=kernel_map, pts=np.int32(shrunk_polygons), color=text_id + 1)

            # get ignored text regions
            if self.ignore_blur_text and (shape.get('value', '###') == '###'):
                ignored_polys.append(points)

            text_id += 1

        effective_map = self.generate_effective_map(mask_height=height, mask_width=width, ignored_polys=ignored_polys)

        return text_map, kernel_map, effective_map

    def to_valid_poly(
        self, polygon: List[Tuple[float, float]], image_height: int, image_width: int
    ) -> List[Tuple[float, float]]:
        polygon = np.array(polygon)
        polygon[:, 0] = np.clip(polygon[:, 0], a_min=0, a_max=image_width - 1)  # x coord not max w-1, and not min 0
        polygon[:, 1] = np.clip(polygon[:, 1], a_min=0, a_max=image_height - 1)  # y coord not max h-1, and not min 0
        return polygon.tolist()

    def to_4points(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        x1, y1 = points[0][0], points[0][1]
        x2, y2 = points[1][0], points[1][1]
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    def shrink_polygon(
        self, points: List[Tuple[float, float]], r: float = 0.4, max_shrink: int = 20, epsilon: float = 0.001
    ):
        shrinker = pyclipper.PyclipperOffset()

        poly = Polygon(points)
        d = min(round(poly.area * (1 - r ** 2) / (poly.length + epsilon)), max_shrink)

        shrinker.AddPath(points, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        polys = shrinker.Execute(-d)

        return polys

    def generate_effective_map(
        self, mask_height: int, mask_width: int, ignored_polys: List[List[Tuple[float, float]]]
    ) -> np.ndarray:
        """Generate effective mask by setting the ineffective regions to 0 and effective regions to 1.
        Args:
            mask_height (int): the height of mask size.
            mask_width (int): the width of mask size.
            ignored_polys (List[List[Tuple[float, float]]]: The list of ignored text polygons.
        Returns:
            mask (ndarray): The effective mask of (height, width).
        """
        mask = np.ones(shape=(mask_height, mask_width), dtype=np.uint8)
        if len(ignored_polys):
            cv2.fillPoly(img=mask, pts=np.int32(ignored_polys), color=0)

        return mask
