import cv2
import json
import argparse
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted


def totaltext_to_json(image_path: Path, label_path: Path, output_dir: Path):
    image = cv2.imread(str(image_path))
    label_info = label_path.open(mode='r', encoding='utf-8')
    ornts = {
        'c': 'curve',
        'h': 'horizontal',
        'm': 'multi_oriented',
        '#': 'do_not_care'
    }

    word_infos = []
    for line in label_info.readlines():
        parts = [
            [part.strip() for part in sub_line.strip().split(':')]
            for sub_line in line.split(',')
        ]

        points, x_coords, y_coords, ornt, content = [], [], [], '#', '#'
        for label, values in parts:
            if label == 'x':
                x_coords = list(map(int, values.strip('[[').strip(']]').split()))
            elif label == 'y':
                y_coords = list(map(int, values.strip('[[').strip(']]').split()))
            elif label == 'ornt':
                ornt = ornts.get(values[3:4].lower(), 'do_not_care')
            elif label == 'transcriptions':
                content = values[3:-2]

        if len(x_coords) != 0 and len(y_coords) != 0:
            points = [(x, y) for x, y in zip(x_coords, y_coords)]
            word_infos.append({'poly': points, 'ornt': ornt, 'content': content})

    json_data = dict()
    json_data['version'] = "4.5.7"
    json_data['imagePath'] = image_path.name
    json_data['imageData'] = None
    json_data['imageHeight'] = image.shape[0]
    json_data['imageWidth'] = image.shape[1]
    json_data['shapes'] = []

    for word_info in word_infos:
        word = dict()
        word['label'] = 'word'
        word['points'] = word_info['poly']
        word['shape_type'] = 'polygon'
        word['value'] = word_info['content'] if word_info['content'] != '#' else ''
        word['ornt'] = word_info['ornt']
        json_data['shapes'].append(word)

    image_path = output_dir.joinpath(image_path.name)
    json_path = image_path.with_suffix('.json')

    with json_path.open(mode='w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)

    cv2.imwrite(str(image_path), image)


if __name__ == '__main__':
    # USAGE
    # python utils/totaltext_to_json.py
    # --image-dir <input/totaltext/Images/Train/>
    # --label-dir <input/totaltext/Labels/Train/>
    # --output-dir <output/totaltext/train/>

    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir')
    parser.add_argument('--label-dir')
    parser.add_argument('--output-dir')
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir)
    output_dir = Path(args.output_dir) if args.output_dir else image_dir
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    image_paths = []
    for image_pattern in ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG']:
        image_paths += list(image_dir.glob(image_pattern))
    image_paths = natsorted(image_paths, key=lambda p: p.stem)
    label_paths = natsorted(label_dir.glob('*.txt'), key=lambda p: p.stem)

    image_names = [image_path.stem for image_path in image_paths]  # image_path: imgxx.txt
    label_names = [label_path.stem.split('_')[-1] for label_path in label_paths]  # label_path: poly_gt_imgxx.txt
    for image_name in image_names:
        if image_name not in label_names:
            print('image path not in label names:', image_name)
    for label_name in label_names:
        if label_name not in image_names:
            print('label name not in image names:', label_name)

    assert image_names == label_names, f'number of image paths {len(image_paths)} - number of label paths {len(label_paths)}'

    data_pairs = [[image_path, label_path] for image_path, label_path in zip(image_paths, label_paths)]

    for image_path, label_path in tqdm(data_pairs):
        totaltext_to_json(image_path, label_path, output_dir)
