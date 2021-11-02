import os
import sys
sys.path.append(os.environ['PWD'])

import cv2
import time
import utils
import argparse
import numpy as np
from pathlib import Path
from natsort import natsorted
from typing import List, Tuple
from pdf2image import convert_from_path, pdfinfo_from_path


def pdf_path_to_image_paths(
    pdf_path: str,
    num_per_batch: int = 10,
    dpi: int = 200,
    image_suffix: str = '.jpg'
) -> List[Path]:
    info = pdfinfo_from_path(pdf_path, userpw=None, poppler_path=None)
    maxPages = info['Pages']

    pdf_path = Path(pdf_path)
    save_dir = pdf_path.parent.joinpath(pdf_path.stem)
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    for batch in range(1, maxPages + 1, num_per_batch):
        pages = convert_from_path(
            pdf_path, dpi=dpi, first_page=batch,
            last_page=min(batch + num_per_batch - 1, maxPages)
        )
        for page in pages:
            image_path = str(save_dir.joinpath(f'{pdf_path.stem}-page{pages.index(page) + batch}{image_suffix}'))
            page.save(image_path, 'JPEG')

    return list(save_dir.glob(f'*{image_suffix}'))


def mksavedir(input_dir: Path, output_dir: Path, path: Path) -> Path:
    sub_path_parts = []
    while str(input_dir).strip('/') != str(path).strip('/'):
        sub_path_parts.insert(0, path.name)
        path = path.parent

    output_dir = output_dir.joinpath('/'.join(sub_path_parts)).parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    return output_dir


def module_time(module, module_name, *args):
    start = time.time()
    output = module(*args)
    stop = time.time()
    print('{}: {:.4f}s'.format(module_name, stop - start))
    return output


def draw_polygon(
    title: str,
    image: np.ndarray,
    points: List[Tuple[int, int]],
    color: Tuple[int, int, int],
    title_position: str = 'bottom_left'  # top_left, bottom_left
) -> None:
    cv2.polylines(
        img=image, pts=[points], isClosed=True, color=color,
        thickness=max(1, max(image.shape) // 400)
    )

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
        img=image, text=title, org=title_pos, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=font_scale,
        color=(255, 255, 255), thickness=thickness, lineType=cv2.LINE_AA
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help='image dir.')
    parser.add_argument('--output-dir', type=str, help='path to save image')
    parser.add_argument('--pattern', type=str, help='glob pattern if image_path is a dir.')
    parser.add_argument('--mode', type=str, default='DB')
    parser.add_argument('--start-index', type=int, default=1)
    args = parser.parse_args()

    colors = {
        'word': (0, 255, 0),
    }

    if args.pattern is not None:
        image_paths = natsorted(Path(args.image_path).glob(args.pattern), key=lambda x: x.stem)
    else:
        image_paths = [Path(args.image_path)]

    output_dir = Path(args.output_dir) if args.output_dir else Path('scripts/output/')
    output_dir = output_dir.joinpath(args.mode)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    stages = utils.eval_config(config='config.yaml')

    for image_id, image_path in enumerate(image_paths[int(args.start_index) - 1:], int(args.start_index)):
        print('-' * 50)
        print(f'{image_id} / {len(image_paths)} - {image_path}')

        # save_dir = mksavedir(input_dir=Path(args.image_path), output_dir=output_dir, path=image_path)

        paths = pdf_path_to_image_paths(image_path) if image_path.suffix == '.pdf' else [image_path]
        images = [cv2.imread(str(image_path)) for image_path in paths]

        if args.mode == 'CRAFT':
            doc_infos, = module_time(stages['CRAFT'], 'CRAFT', images)
        elif args.mode == 'PANNet':
            doc_infos, = module_time(stages['PANNet'], 'PANNet', images)
        elif args.mode == 'DB':
            doc_infos, = module_time(stages['DB'], 'DBnet', images)
            
        for i, doc_info in enumerate(doc_infos):
            image = doc_info.image
            for word in doc_info.words:
                points = np.int32([[point.x, point.y] for point in word.box])
                title = f"{word.field}: {word.confidence:.4f}" if word.field else "Word"
                draw_polygon(
                    image=image, points=points,
                    title=title, color=colors.get(word.field, (0, 255, 0)),
                    title_position='top_left',
                )

            image_name = f'{image_path.stem}_{i}'
            image = np.ascontiguousarray(image, dtype=np.uint8)
            cv2.imwrite(str(output_dir.joinpath(f'{image_name}{image_path.suffix}')), image)
