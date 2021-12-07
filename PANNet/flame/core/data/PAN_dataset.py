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


class PANDataset(Dataset):
    def __init__(
        self,
        dirnames: List[str],
        imsize: int = 640,
        mean: Optional[Tuple[float, float, float]] = None,
        std: Optional[Tuple[float, float, float]] = None,
        shrink_ratio: float = 0.5,
        image_extents: List[str] = ['.jpg'],
        label_extent: str = '.json',
        transforms: Optional[List] = None,
        require_transforms: Optional[List] = None,
    ) -> None:
        super(PANDataset, self).__init__()
        self.imsize = imsize
        self.augmenter = Augmenter()
        self.shrink_ratio = shrink_ratio
        self.transforms = transforms if transforms else []
        self.require_transforms = require_transforms if require_transforms else []

        self.mean = torch.tensor(mean, dtype=torch.float).view(3, 1, 1) if mean is not None else None
        self.std = torch.tensor(std, dtype=torch.float).view(3, 1, 1) if std is not None else None

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
        image_path, label_path = self.data_pairs[idx]

        image = cv2.imread(str(image_path))
        with label_path.open(mode='r', encoding='utf8') as f:
            label = json.load(f)

        if (self.mean is not None) and (self.std is not None):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_info = {
            'image_path': str(image_path),
            'image_size': image.shape[1::-1],
            'text_boxes': None
        }

        for transform in random.sample(self.transforms, k=random.randint(0, len(self.transforms))):
            image, label = self.augmenter.apply(image=image, label=label, augmenter=transform)

        for require_transform in self.require_transforms:
            image, label = self.augmenter.apply(image=image, label=label, augmenter=require_transform)

        image, label = self.resize(image=image, label=label, imsize=self.imsize)

        text_map, kernel_map, text_boxes = self.generate_segmap(image=image, label=label, shrink_ratio=self.shrink_ratio)
        image_info['text_boxes'] = text_boxes  # save for evaluation

        image = torch.from_numpy(np.ascontiguousarray(image))  # H x W x 3, tensor.uint8
        text_map = torch.from_numpy(np.ascontiguousarray(text_map))  # H x W, tensor.uint8
        kernel_map = torch.from_numpy(np.ascontiguousarray(kernel_map))  # H x W, tensor.uint8

        image = image.permute(2, 0, 1).contiguous().float()  # 3 x H x W, tensor.float32
        mask = torch.stack([text_map, kernel_map], dim=0).float()  # 2 x H x W, tensor.float32

        if (self.mean is not None) and (self.std is not None):
            image = (image.div(255.) - self.mean) / self.std
        else:
            image = (image - image.mean()) / image.std()

        return image, mask, image_info

    def resize(self, image, label, imsize=640):
        f = imsize / min(image.shape[:2])

        image, label = self.augmenter.apply(
            image=image, label=label,
            augmenter=iaa.Resize(size=f)
        )

        image, label = self.augmenter.apply(
            image=image, label=label,
            augmenter=iaa.CropToFixedSize(width=imsize, height=imsize, position='center')
        )

        return image, label

    def to_valid_poly(self, polygon, image_height, image_width):
        polygon = np.array(polygon)
        polygon[:, 0] = np.clip(polygon[:, 0], a_min=0, a_max=image_width - 1)  # x coord not max w-1, and not min 0
        polygon[:, 1] = np.clip(polygon[:, 1], a_min=0, a_max=image_height - 1)  # y coord not max h-1, and not min 0
        return polygon.tolist()

    def to_4points(self, points):
        x1, y1 = points[0][0], points[0][1]
        x2, y2 = points[1][0], points[1][1]
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    def shrink_polygon(self, points, r: float = 0.4):
        shrinker = pyclipper.PyclipperOffset()

        poly = Polygon(points)
        if not poly.is_valid:
            raise ValueError('must be valid polygon.')

        # d = poly.area * (1 - r ** 2) / poly.length
        d = poly.area * (1 - r) / poly.length

        points = [tuple(point) for point in points]
        shrinker.AddPath(points, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        polys = shrinker.Execute(-d)

        return polys

    def generate_segmap(
        self, image: np.ndarray, label: dict, shrink_ratio: float
    ) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        height, width = image.shape[:2]

        text_map = np.zeros(shape=(height, width), dtype=np.uint8)
        kernel_map = np.zeros(shape=(height, width), dtype=np.uint8)

        text_id = 0
        text_boxes = []
        for shape in label['shapes']:
            if shape['shape_type'] == 'rectangle':
                points = self.to_4points(shape['points'])
            elif shape['shape_type'] == 'polygon':
                points = shape['points']
            else:
                continue

            # get text boxes after transformers for evaluation
            text_boxes.append(
                {
                    'points': [tuple(point) for point in points],
                    'text': shape.get('value', None),
                    'ignore': False,
                }
            )

            # text region map
            polygon = self.to_valid_poly(points, image_height=height, image_width=width)
            cv2.fillPoly(img=text_map, pts=[np.int32(polygon)], color=text_id + 1)

            # shrinked kernel map
            shrinked_polygons = self.shrink_polygon(polygon, r=shrink_ratio)
            cv2.fillPoly(img=kernel_map, pts=np.int32(shrinked_polygons), color=text_id + 1)

            text_id += 1

        return text_map, kernel_map, text_boxes
