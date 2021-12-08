import cv2
import torch
import numpy as np
import imgaug.augmenters as iaa

from pathlib import Path
from ignite.engine import Events
from typing import Optional, List, Tuple

from ..module import Module


class Predictor(Module):
    def __init__(
        self, evaluator_name: str, imsize: int, output_dir: str, output_transform=lambda x: x
    ):
        super(Predictor, self).__init__()
        self.imsize = imsize
        self.evaluator_name = evaluator_name
        self.output_dir = Path(output_dir)
        self._output_transform = output_transform

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

    def init(self):
        assert self.evaluator_name in self.frame, f'The frame does not have {self.evaluator_name}'
        self._attach(self.frame[self.evaluator_name].engine)

    def reset(self):
        pass

    def update(self, output):
        preds_boxes, image_infos = output
        trues_boxes = [image_info['text_boxes'] for image_info in image_infos]
        image_names = [image_info['image_path'] for image_info in image_infos]
        image_sizes = [image_info['image_size'] for image_info in image_infos]

        for pred_boxes, true_boxes, image_name, image_size in zip(preds_boxes, trues_boxes, image_names, image_sizes):
            true_image = cv2.imread(image_name)
            true_image = self.resize(true_image, imsize=self.imsize)

            pred_image = cv2.imread(image_name)
            pred_image = self.resize(pred_image, imsize=self.imsize)

            for box in true_boxes:
                self.draw_polygon(image=true_image, points=box['points'], color=(0, 255, 0))
            cv2.imwrite(str(self.output_dir.joinpath(f'{Path(image_name).stem}_gt{Path(image_name).suffix}')), true_image)

            for box in pred_boxes:
                self.draw_polygon(image=pred_image, points=box['points'], color=(0, 0, 255))
            cv2.imwrite(str(self.output_dir.joinpath(f'{Path(image_name).stem}_pred{Path(image_name).suffix}')), pred_image)

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

    def resize(self, image: np.ndarray, imsize: int = 640) -> np.ndarray:
        f = imsize / min(image.shape[:2])
        image = iaa.Resize(size=f)(image=image)
        image = iaa.CropToFixedSize(width=imsize, height=imsize, position='center')(image=image)

        return image

    def draw_polygon(
        self,
        image: np.ndarray,
        points: List[Tuple[int, int]],
        color: Tuple[int, int, int],
        title_position: str = 'bottom_left',  # top_left, bottom_left
        title: Optional[str] = None,
    ) -> None:
        cv2.polylines(
            img=image, pts=[np.int32(points)], isClosed=True, color=color, thickness=max(1, max(image.shape) // 500)
        )

        if title is not None:
            font_scale = max(image.shape) / 1200
            thickness = max(1, max(image.shape) // 600)
            w, h = cv2.getTextSize(title, cv2.FONT_HERSHEY_PLAIN, font_scale, thickness)[0]

            if title_position == 'bottom_left':
                title_box = [(points[0][0], points[0][1]),
                             (points[0][0] + w, points[0][1] + int(1.5 * h))]
                title_pos = (points[0][0], points[0][1] + int(1.3 * h))
            elif title_position == 'top_left':
                title_box = [(points[0][0], points[0][1]),
                             (points[0][0] + w, max(0, points[0][1] - int(1.5 * h)))]
                title_pos = (points[0][0], max(0, points[0][1] - int(0.3 * h)))

            cv2.rectangle(img=image, pt1=title_box[0], pt2=title_box[1], color=color, thickness=-1)

            cv2.putText(
                img=image, text=title, org=title_pos,
                fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=font_scale,
                color=(255, 255, 255), thickness=thickness, lineType=cv2.LINE_AA
            )
