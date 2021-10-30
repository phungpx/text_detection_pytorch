import cv2
import json
import natsort
import argparse

from tqdm import tqdm
from pathlib import Path


def icdar_to_json(txt_path, image_path, output_dir, delim, text_strip_char=None):
    with txt_path.open(mode='r', encoding='cp1252') as f:
        lines = [line.strip() for line in f if line.strip()]
        lines = [line.split(delim) for line in lines]
        for i, line in enumerate(lines):
            if len(line) == 5:
                lines[i] = [
                    int(line[0]), int(line[1]), int(line[2]), int(line[3]),
                    line[4].strip(text_strip_char)
                ]
            elif 5 < len(line) < 9:
                lines[i] = [
                    int(line[0]), int(line[1]), int(line[2]), int(line[3]),
                    (delim.join(line[4:])).strip(text_strip_char)
                ]
            elif len(line) == 9:
                lines[i] = [
                    int(line[0]), int(line[1]), int(line[2]), int(line[3]),
                    int(line[4]), int(line[5]), int(line[6]), int(line[7]),
                    line[8].strip(text_strip_char)
                ]
            elif len(line) > 9:
                lines[i] = [
                    int(line[0]), int(line[1]), int(line[2]), int(line[3]),
                    int(line[4]), int(line[5]), int(line[6]), int(line[7]),
                    (delim.join(line[8:])).strip(text_strip_char)
                ]
            else:
                raise RuntimeError(f'{txt_path} Groundtruth formating is not supported.')

    image = cv2.imread(str(image_path))

    json_info = dict()
    json_info['version'] = "4.5.6"
    json_info['imagePath'] = image_path.name
    json_info['imageData'] = None
    json_info['imageHeight'] = image.shape[0]
    json_info['imageWidth'] = image.shape[1]
    json_info['shapes'] = []

    for line in lines:
        if len(line) in [5, 9]:
            text = line.pop(-1)
            points = [[line[2 * i], line[2 * i + 1]] for i in range(len(line) // 2)]

        text_region = dict()
        text_region['label'] = 'word'
        text_region['value'] = text
        text_region['points'] = points
        text_region['shape_type'] = 'rectangle' if len(points) == 2 else 'polygon'

        json_info['shapes'].append(text_region)

    with open(str(output_dir.joinpath(f'{image_path.stem}.json')), mode='w', encoding='utf8') as f:
        json.dump(json_info, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir')
    parser.add_argument('--output-dir')
    parser.add_argument('--delim', default=',')
    parser.add_argument('--text-strip-char', default=None)
    args = parser.parse_args()

    datadir = Path(args.datadir)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.datadir)

    image_paths = []
    for image_pattern in ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG']:
        image_paths += list(datadir.glob(image_pattern))
    image_paths = natsort.natsorted(image_paths, key=lambda p: p.stem)
    txt_paths = natsort.natsorted(datadir.glob('*.txt'), key=lambda p: p.stem)

    image_names = [image_path.stem for image_path in image_paths]
    txt_names = [txt_path.stem for txt_path in txt_paths]
    for image_name in image_names:
        if image_name not in txt_names:
            print('image path not in txt names:', image_name)
    for txt_name in txt_names:
        if txt_name not in image_names:
            print('txt name not in image names:', txt_name)

    assert image_names == txt_names, f'number of image paths {len(image_paths)} - number of txt paths {len(txt_paths)}'

    for txt_path, image_path in tqdm(list(zip(txt_paths, image_paths))):
        icdar_to_json(txt_path, image_path, output_dir, args.delim, args.text_strip_char)
