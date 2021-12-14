import json
import torch

from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

from .iou import DetectionIoUEvaluator


class HMean(Metric):
    def __init__(self, output_transform=lambda x: x):
        super(HMean, self).__init__(output_transform)
        self.evaluator = DetectionIoUEvaluator()

    def reset(self):
        self.preds = list()
        self.trues = list()

    def update(self, output):
        '''
        Args:
            output: tuple(preds, targets, effective_maps, image_infos)
                .preds: N x 6 x H x W
                .targets: N x 2 x H x W
                .effective_maps: N x H x W
                .image_infos: tuple(dictionary)
        Return:
            None
        '''
        pred_boxes, _, _, image_infos = output
        true_boxes = [image_info['text_boxes'] for image_info in image_infos]
        pred_boxes = [pred_box['text_boxes'] for pred_box in pred_boxes]

        self.trues.extend(true_boxes)
        self.preds.extend(pred_boxes)

    def compute(self):
        results = [self.evaluator.evaluate_image(true, pred) for true, pred in zip(self.trues, self.preds)]
        metrics = self.evaluator.combine_results(results)

        print(json.dumps(metrics, ensure_ascii=False, indent=4))

        return metrics['hmean']
