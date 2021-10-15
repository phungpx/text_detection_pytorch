
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Generator, List, Union

import cv2
import numpy as np

sys.path.append(os.environ['PWD'])

from name_card_tagger import NameCardTagger  # noqa: E402


def module_time(module, module_name, *args):
    start = time.time()
    output = module(*args)
    stop = time.time()
    print('{}: {:.4f}s'.format(module_name, stop - start))
    return output


def chunks(lst: list, size: int = None) -> Union[List, Generator]:
    if size is None:
        yield lst
    else:
        for i in range(0, len(lst), size):
            yield lst[i:i + size]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir', help='image dir.')
    parser.add_argument('--image-chunk', default=1, help='')
    parser.add_argument('--output-dir', help='path to save image')
    parser.add_argument('--image-pattern', help='glob pattern if image_path is a dir.')
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    save_dir = Path(args.output_dir) if args.output_dir else Path('scripts/output/')
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    tagger = NameCardTagger()

    for image_paths in chunks(list(image_dir.glob(args.image_pattern)), size=int(args.image_chunk)):
        images = [cv2.imread(str(image_path)) for image_path in image_paths]

        image_infos = module_time(tagger, 'Tagging', images)

        for image_info, image_path in zip(image_infos, image_paths):
            for i, card in enumerate(image_info.cards):
                if card.words is not None:
                    image_name = str(save_dir.joinpath(image_path.name if i == 0
                                                       else f'{image_path.stem}_{i}.{image_path.suffix}'))

                    words = [[[point.x, point.y] for point in word.box] for word in card.words]

                    card.image = cv2.polylines(img=card.image, pts=np.array(words).astype(np.int32),
                                               isClosed=True, color=(0, 255, 0), thickness=2)

                    cv2.imwrite(image_name, card.image)

            _image = image_info.visualize
            cv2.imwrite(str(save_dir.joinpath(f'{image_path.stem}_visualized.{image_path.suffix}')), _image)
