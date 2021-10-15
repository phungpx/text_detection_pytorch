from typing import List

import numpy as np

import utils
from abstract.project import Project


class NameCardTagger(Project):
    def __init__(self, config_path: str = None):
        super(NameCardTagger, self).__init__(config_path)
        self.stages = utils.eval_config(self.config_path)

    def run(self, images: List[np.ndarray]) -> List:
        info = images

        for stage_name in self.stages:
            info, = self.stages[stage_name](info)

        return info
