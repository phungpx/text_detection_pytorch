import cv2
import torch
import numpy as np
from pathlib import Path
from ignite.engine import Events

from ..module import Module


class RegionPredictor(Module):
    def __init__(self, evaluator_name, output_dir, output_img_ext, output_mask_ext, classes, output_transform=lambda x: x):
        super(RegionPredictor, self).__init__()
        self.classes = classes
        self.evaluator_name = evaluator_name
        self.output_dir = Path(output_dir)
        self.output_img_ext = output_img_ext
        self.output_mask_ext = output_mask_ext
        self._output_transform = output_transform

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

    def init(self):
        assert self.evaluator_name in self.frame, f'The frame does not have {self.evaluator_name}'
        self._attach(self.frame[self.evaluator_name].engine)

    def reset(self):
        pass

    def update(self, output):
        pred_boxes, image_infos = output
        image_names, image_sizes = image_infos
        image_sizes = [(w.item(), h.item()) for w, h in zip(*image_sizes)]

        for pred, image_name, image_size in zip(preds, image_names, image_sizes):
            image_path = '{}/{}{}'.format(self.output_dir, Path(image_name).stem, self.output_img_ext)
            mask_path = '{}/{}_mask{}'.format(self.output_dir, Path(image_name).stem, self.output_mask_ext)

            image = cv2.imread(image_name)
            mask = [
                cv2.resize(
                    self._rm_small_components(pred[i], class_ratio), dsize=image_size, interpolation=cv2.INTER_NEAREST
                )
                for _, i, class_ratio in self.classes.values()
            ]
            color_mask = np.zeros(shape=(*image_size[::-1], 3), dtype=np.uint8)

            for (color, _, _), m in zip(self.classes.values(), mask):
                color_mask[m.astype(np.bool)] = np.array(color, dtype=np.uint8)
                image[m.astype(np.bool)] = (
                    0.7 * image[m.astype(np.bool)].astype(np.float32) + 0.3 * np.array(color, dtype=np.float32)
                ).astype(np.uint8)

            class_contours = [cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2] for m in mask[-1:]]

            thickness = max(image.shape) // 500
            for cnts, (color, _, _) in zip(class_contours, self.classes.values()):
                cv2.drawContours(image=image, contours=cnts, contourIdx=-1, color=color, thickness=thickness)
                cv2.drawContours(image=color_mask, contours=cnts, contourIdx=-1, color=(255, 255, 255), thickness=thickness)

            cv2.imwrite(image_path, image)
            cv2.imwrite(mask_path, color_mask)

    def compute(self):
        pass

    def started(self, engine):
        self.reset()

    @torch.no_grad()
    def iteration_completed(self, engine):
        output = self._output_transform(engine.state.output)
        self.update(output)

    def completed(self, engine):
        self.compute()

    def _attach(self, engine):
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.completed)
        if not engine.has_event_handler(self.started, Events.EPOCH_STARTED):
            engine.add_event_handler(Events.EPOCH_STARTED, self.started)
        if not engine.has_event_handler(self.iteration_completed, Events.ITERATION_COMPLETED):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)

    def _rm_small_components(self, mask, class_ratio):
        mask = mask.copy()
        num_class, label = cv2.connectedComponents(mask)
        for i in range(1, num_class):
            area = (label == i).sum()
            if area < class_ratio * mask.shape[0] * mask.shape[1]:
                mask[label == i] = 0
        return mask
