import torch

from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

from .iou import DetectionIoUEvaluator


class IOU(Metric):
    def __init__(self, output_transform=lambda x: x):
        super(IOU, self).__init__(output_transform)
        self.evaluator = DetectionIoUEvaluator()

    def reset(self):
        self.preds = list()
        self.trues = list()

    def update(self, output):
        '''
        Args:
            output: tuple(pred, target, image_infos)
                .pred: N x 1 x H x W
                .target: N x H x W
        Return:
            None
        '''
        pred_boxes, image_infos = output
        true_boxes = image_infos['text_boxes']

        self.trues.append(true_boxes)
        self.preds.append(pred_boxes)

    def compute(self):
        results = [self.evaluator.evaluate_image(true, pred) for true, pred in zip(self.trues, self.preds)]
        metrics = evaluator.combine_results(results)
        print(metrics)
